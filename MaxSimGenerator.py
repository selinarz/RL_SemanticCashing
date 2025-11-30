import torch
import numpy as np
import sys
import os
from tensordict import TensorDict
from rl4co.envs.common.utils import Generator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from embedding_model import EmbeddingModel

class MaxSimGenerator(Generator):
    def __init__(self, prompts, max_len=128, embedding_model=None, lm_model_name_or_path=None):
        super().__init__()
        if prompts is None:
            raise ValueError("Prompts must be provided")
        self.prompts = prompts
        self.num_prompts = len(prompts)
        self.max_len = max_len
        
        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
          
            self.embedding_model = EmbeddingModel(model_name=lm_model_name_or_path)

        self.embedding_model.model.eval()
     
        self.tokenizer = self.embedding_model.tokenizer
        self.lm = self.embedding_model.model

    def _generate(self, batch_size, **kwargs):
        
        if isinstance(batch_size, (list, tuple)):
            bs = int(batch_size[0])
        elif isinstance(batch_size, torch.Size):
            bs = int(batch_size[0])
        else:
            bs = int(batch_size)
            
      
        device = next(self.lm.parameters()).device
        if not hasattr(self, "_dbg_printed"):
            print(f"[GEN] bs={bs} device={device} lm_device={device} max_len={self.max_len}")
            self._dbg_printed = True

        # 随机采样文本对
        indices_a = np.random.randint(0, self.num_prompts, size=bs)
        indices_b = np.random.randint(0, self.num_prompts, size=bs)

        texts_a = [self.prompts[i] for i in indices_a]
        texts_b = [self.prompts[i] for i in indices_b]

        # ==================================================================
        #查看原始文本
        # print(f"--- [DEBUG] Batch Size: {bs}, Device: {device} ---")
        # for i in range(bs):
        #     print(f"  Pair {i+1}: A='{texts_a[i]}', B='{texts_b[i]}'")
        # print("----------------------------------------------------")
        # ==================================================================

        # 使用 EmbeddingModel 获取token级别的嵌入
        token_embeds_a = self.embedding_model.get_token_embeddings(
            texts_a, max_length=self.max_len, device=device
        )
        token_embeds_b = self.embedding_model.get_token_embeddings(
            texts_b, max_length=self.max_len, device=device
        )
        
        embeddings_a = token_embeds_a['last_hidden_state']
        mask_a = token_embeds_a['attention_mask'].to(torch.bool)
        embeddings_b = token_embeds_b['last_hidden_state']
        mask_b = token_embeds_b['attention_mask'].to(torch.bool)
       
        inputs_a = {
            'input_ids': token_embeds_a['input_ids'],
            'attention_mask': token_embeds_a['attention_mask']
        }
        inputs_b = {
            'input_ids': token_embeds_b['input_ids'],
            'attention_mask': token_embeds_b['attention_mask']
        }
        
       
        lengths_a = inputs_a['attention_mask'].sum(dim=1)
        lengths_b = inputs_b['attention_mask'].sum(dim=1)

        td = TensorDict(
            {
               
               "token_embeddings_a": embeddings_a,
                "attention_mask_a": mask_a,  
                "token_embeddings_b": embeddings_b,
                "attention_mask_b": mask_b,

             
                "input_ids_a": inputs_a['input_ids'],
                "input_ids_b": inputs_b['input_ids'],
                "length_a": lengths_a,
                "length_b": lengths_b,
            },
            batch_size=[bs],
            device=device,
        )
        return td