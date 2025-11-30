import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from torch.distributions import Categorical

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from embedding_model import EmbeddingModel


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, nhead: int):
        super().__init__()
        self.mha_ab = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, batch_first=True)
        self.mha_ba = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, batch_first=True)
        self.norm_ab = nn.LayerNorm(hidden_dim)
        self.norm_ba = nn.LayerNorm(hidden_dim)
        self.ff_ab = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.ff_ba = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        pad_mask_a: torch.Tensor | None = None,
        pad_mask_b: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # A attends to B
        attn_a, _ = self.mha_ab(query=emb_a, key=emb_b, value=emb_b, key_padding_mask=pad_mask_b)
        a_out = self.norm_ab(emb_a + attn_a)
        a_out = self.norm_ab(a_out + self.ff_ab(a_out))

        # B attends to A
        attn_b, _ = self.mha_ba(query=emb_b, key=emb_a, value=emb_a, key_padding_mask=pad_mask_a)
        b_out = self.norm_ba(emb_b + attn_b)
        b_out = self.norm_ba(b_out + self.ff_ba(b_out))

        return a_out, b_out

class AdaptedPointerNetworkPolicy(nn.Module):
    """
    用于双文本分割的指针网络策略。

    该策略使用交叉注意力Transformer对两个输入文本进行编码，
    并使用一个自回归的LSTM解码器来依次预测两个文本的分割边界。
    """
    def __init__(self,   
                 env, 
                 embedding_dim=768,    
                 hidden_dim=128,        
                 max_segments=6,       
                 nhead=4,               
                 num_encoder_layers=2  
                ):
        super().__init__()
        self.env = env
        self.hidden_dim = hidden_dim
        self.max_segments = max_segments
      
        self.train_decode_type = "sampling"
        self.val_decode_type = "greedy"
        self.test_decode_type = "greedy"
        

        self.input_proj = None

        # --- 编码器 (Encoder) ---
    
        self.encoder_layers = nn.ModuleList([
            CrossAttentionBlock(hidden_dim=hidden_dim, nhead=nhead)
            for _ in range(num_encoder_layers)
        ])

        # --- 解码器 (Decoder) ---
      
        self.decoder_cell = nn.LSTMCell(hidden_dim, hidden_dim) 
        

        self.attention_linear_decoder = nn.Linear(hidden_dim, hidden_dim)
        self.attention_linear_encoder = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=1)
        self.V = nn.Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        nn.init.uniform_(self.V, -1, 1)
        
       
        self.decoder_start_input = nn.Parameter(torch.randn(hidden_dim), requires_grad=True)
        if hasattr(env, 'device'):
            self._device = env.device
        else:
            self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
       
        if hasattr(env, 'reward_lm'):
          
            self.embedding_model = env.reward_lm
            self.tokenizer = env.reward_lm.tokenizer
            self.lm = env.reward_lm.model
           
            self.lm.to(self._device)
            self._device = next(self.lm.parameters()).device
        elif hasattr(env, 'generator') and hasattr(env.generator, 'embedding_model'):
      
            self.embedding_model = env.generator.embedding_model
            self.tokenizer = env.generator.embedding_model.tokenizer
            self.lm = env.generator.embedding_model.model
            self.lm.to(self._device)
            self._device = next(self.lm.parameters()).device
        else:
            
            self.embedding_model = EmbeddingModel()
            self.embedding_model.model.to(self._device)
            for param in self.embedding_model.model.parameters():
                param.requires_grad = False
            self.embedding_model.model.eval()
            self.tokenizer = self.embedding_model.tokenizer
            self.lm = self.embedding_model.model
            self._device = next(self.lm.parameters()).device
    def _init_punctuation_ids(self, device):
        """
        初始化有效的分割token ID集合。
        包含: 基础标点 + [SEP]/[EOS] + 带空格的标点变体。
        """
      
        if hasattr(self, "_valid_split_ids") and self._valid_split_ids is not None:
            if self._valid_split_ids.device == device:
                return
        
      
        punct_chars = {",", ".", "!", "?", ":", ";", "，", "。", "！", "？", "：", "；"}
        valid_ids = set()
        
        if self.tokenizer.sep_token_id is not None:
            valid_ids.add(self.tokenizer.sep_token_id)
        if self.tokenizer.eos_token_id is not None:
            valid_ids.add(self.tokenizer.eos_token_id)
            
  
        for char in punct_chars:
            # Case A: 纯标点
            ids = self.tokenizer.encode(char, add_special_tokens=False)
            if ids: valid_ids.update(ids)
            
            # Case B: 带空格前缀
            ids_space = self.tokenizer.encode(" " + char, add_special_tokens=False)
            if ids_space: valid_ids.update(ids_space)

     
        sorted_ids = sorted(list(valid_ids))
        self._valid_split_ids = torch.tensor(sorted_ids, device=device, dtype=torch.long)
    @property
    def device(self):
        """获取模型所在的设备"""
        if hasattr(self, '_device'):
            return self._device
      
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    def forward(self, td, env=None, phase: str = "train", select_best: bool = False, **kwargs):
        """与 RL4CO 的 REINFORCE 调用约定兼容的入口。
        根据阶段/标志选择采样或贪心解码，返回包含 reward 与 log_likelihood 的字典。
        """
       
        decode_override = kwargs.pop("decode_type", None)
        if decode_override is not None:
            decode_type = decode_override
        else:
            decode_type = "greedy" if (select_best or phase != "train") else "sampling"

        actions, log_likelihood, info = self._forward(td, decode_type=decode_type, **kwargs)
        # 计算奖励
        used_env = env if env is not None else getattr(self, 'env', None)
        if used_env is None:
            raise RuntimeError("Environment instance is required to compute reward.")
        reward = used_env.get_reward(td, actions)

        return {
            "actions": actions,
            "log_likelihood": log_likelihood,
            "reward": reward,
            "info": info,
        }

    def _forward(self, td, decode_type="sampling", **kwargs):
        """
        实现自回归解码的核心逻辑 。
        """
       
     
        target_device = next(self.lm.parameters()).device
        if getattr(self, "_current_module_device", None) != target_device:
            self.to(target_device)
            self._current_module_device = target_device
        self._device = target_device
        
   
        try:
            td = td.to(target_device)
        except Exception:
            for k in ['token_embeddings_a','token_embeddings_b','attention_mask_a','attention_mask_b',
                      'input_ids_a','input_ids_b','length_a','length_b']:
                if k in td and isinstance(td[k], torch.Tensor):
                    td[k] = td[k].to(target_device)
        self._init_punctuation_ids(td['input_ids_a'].device)
        
     
        input_dim = td['token_embeddings_a'].size(-1)
        if self.input_proj is None:
            with torch.inference_mode(False):
                if input_dim == self.hidden_dim:
                    self.input_proj = nn.Identity()
                else:
                    self.input_proj = nn.Linear(input_dim, self.hidden_dim)
            self.input_proj.to(self.device)

        embedded_a = self.input_proj(td['token_embeddings_a'])
        embedded_b = self.input_proj(td['token_embeddings_b'])
        
        mask_a = ~td['attention_mask_a'].bool()
        mask_b = ~td['attention_mask_b'].bool()
        
        batch_size = embedded_a.size(0)
        seq_len_a = embedded_a.size(1)
        seq_len_b = embedded_b.size(1)
        
        # --- 2. Encoder (Cross-Attention) ---
        for layer in self.encoder_layers:
            embedded_a, embedded_b = layer(
                emb_a=embedded_a, 
                emb_b=embedded_b, 
                pad_mask_a=mask_a, 
                pad_mask_b=mask_b
            )
        encoder_outputs_a = embedded_a
        encoder_outputs_b = embedded_b
        
        # --- 3. Decoder 初始化 ---
        decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1)
        h, c = (torch.zeros(batch_size, self.decoder_cell.hidden_size, device=self.device),
                torch.zeros(batch_size, self.decoder_cell.hidden_size, device=self.device))
        
        pointers = []
        log_probs = []
        
        # 初始边界设为 0 (对应 [CLS])
        current_bA = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        current_bB = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # 预计算全局标点 Mask [batch, seq_len]
    
        is_punct_global_a = torch.isin(td['input_ids_a'], self._valid_split_ids)
        is_punct_global_b = torch.isin(td['input_ids_b'], self._valid_split_ids)

        # --- 4. 解码循环 ---
        for step in range(self.max_segments):
            # (a) LSTM Step
            h, c = self.decoder_cell(decoder_input, (h, c))
            query_vec = self.attention_linear_decoder(h)

            #  指针 A 选择
            range_a = torch.arange(seq_len_a, device=self.device).expand(batch_size, -1)
            
            # 1. 必须是标点符号 (预计算结果)
            # 2. 必须严格在当前边界之后 (range_a > current_bA)
            # 3. 必须在有效长度内 (range_a < length_a)
       
            
            is_future_a = range_a > current_bA.unsqueeze(1)
            is_in_length_a = range_a < td['length_a'].unsqueeze(1)
            valid_slots_a = is_punct_global_a & is_future_a & is_in_length_a
            mask_ptr_a = ~valid_slots_a

            inp_a = query_vec.unsqueeze(2)
            ctx_a = self.attention_linear_encoder(encoder_outputs_a.permute(0, 2, 1))
            V_exp = self.V.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)
            attn_scores_a = torch.bmm(V_exp, torch.tanh(inp_a + ctx_a)).squeeze(1)
            
       
            mask_fill_val = float('-inf') 
            attn_scores_a = attn_scores_a.masked_fill(mask_ptr_a, mask_fill_val)

            # === 指针 B 选择 ===
            range_b = torch.arange(seq_len_b, device=self.device).expand(batch_size, -1)
            
            is_future_b = range_b > current_bB.unsqueeze(1)
            is_in_length_b = range_b < td['length_b'].unsqueeze(1)
            
            valid_slots_b = is_punct_global_b & is_future_b & is_in_length_b
            mask_ptr_b = ~valid_slots_b

            inp_b = query_vec.unsqueeze(2)
            ctx_b = self.attention_linear_encoder(encoder_outputs_b.permute(0, 2, 1))
            attn_scores_b = torch.bmm(V_exp, torch.tanh(inp_b + ctx_b)).squeeze(1)
            attn_scores_b = attn_scores_b.masked_fill(mask_ptr_b, mask_fill_val)

            # === 采样/Argmax ===
            pointer_a = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            pointer_b = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            logp_a = torch.zeros(batch_size, device=self.device)
            logp_b = torch.zeros(batch_size, device=self.device)

            has_valid_a = valid_slots_a.any(dim=1)
            has_valid_b = valid_slots_b.any(dim=1)

            # 处理 A
            if has_valid_a.any():
                rows = has_valid_a
                if decode_type == "sampling":
                    dist = Categorical(logits=attn_scores_a[rows])
                    act = dist.sample()
                    pointer_a[rows] = act
                    logp_a[rows] = dist.log_prob(act)
                else:
                    pointer_a[rows] = torch.argmax(attn_scores_a[rows], dim=1)

            if (~has_valid_a).any():
                rows = ~has_valid_a
              
                pointer_a[rows] = td['length_a'][rows] - 1
             

            # 处理 B
            if has_valid_b.any():
                rows = has_valid_b
                if decode_type == "sampling":
                    dist = Categorical(logits=attn_scores_b[rows])
                    act = dist.sample()
                    pointer_b[rows] = act
                    logp_b[rows] = dist.log_prob(act)
                else:
                    pointer_b[rows] = torch.argmax(attn_scores_b[rows], dim=1)
            
            if (~has_valid_b).any():
                rows = ~has_valid_b
                pointer_b[rows] = td['length_b'][rows] - 1

            pointers.append(torch.stack([pointer_a, pointer_b], dim=1))
            log_probs.append(logp_a + logp_b)

        
            feedback_emb_a_list = []
            
            input_ids_a = td['input_ids_a']
            lm_device = next(self.lm.parameters()).device

            for b in range(batch_size):
               
                s_a = current_bA[b].item() # 上一个边界 (包含在上一个片段中)
                e_a = pointer_a[b].item()  # 当前边界 (包含在当前片段中)
                
           
                real_start_a = s_a + 1 if s_a > 0 else 1  
                
              
                real_end_a = e_a + 1
           
                if real_end_a <= real_start_a:
                
                    seg_ids_a = torch.tensor([[self.tokenizer.pad_token_id]], device=lm_device)
                else:
                  
                    seg_ids_a = input_ids_a[b, real_start_a : real_end_a].unsqueeze(0).to(lm_device)

                # 计算 Mask (非 Pad 为 1)
                attn_mask_a_seg = (seg_ids_a != self.tokenizer.pad_token_id).long()
                
                # 调用 LM 提取特征
                with torch.no_grad():
                 
                    if attn_mask_a_seg.sum() == 0:
                        emb_val = torch.zeros(1, self.embedding_model.model.config.hidden_size, device=lm_device)
                    else:
                        outputs = self.lm(seg_ids_a, attention_mask=attn_mask_a_seg)
                       
                        emb_val = outputs.last_hidden_state.mean(dim=1)
                
                feedback_emb_a_list.append(emb_val.squeeze(0))

            # Stack 并放回 Policy 的 device 进行下一步投影
            feedback_tensor_a = torch.stack(feedback_emb_a_list, dim=0).to(self.device)
            
            # 投影作为 LSTM 下一步输入
            decoder_input = self.input_proj(feedback_tensor_a)

            # 更新状态
            current_bA = pointer_a
            current_bB = pointer_b

     
        actions = torch.stack(pointers, 1).view(batch_size, -1)
        log_likelihood = torch.stack(log_probs, 1).sum(dim=1)
        
        return actions, log_likelihood, {}