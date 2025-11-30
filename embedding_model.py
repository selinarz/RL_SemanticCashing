import os
from transformers import AutoModel, AutoTokenizer
import torch


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  

class EmbeddingModel:
    """ 计算文本的向量嵌入 """
    def __init__(self, model_name=None):
       
        if model_name is None:
            preferred_abs = "/home/zhengzishan/bge-base-en"
            if os.path.isdir(preferred_abs):
                model_name = preferred_abs
            else:
                env_path = os.environ.get('BGE_MODEL_PATH')
                if env_path and os.path.isdir(env_path):
                    model_name = os.path.normpath(env_path)
                else:
                   
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                   
                    replay_dir = os.path.dirname(current_dir)
                 
                    default_model_path = os.path.join(replay_dir, "LLMCache", "bge-base-en")
                  
                    model_name = os.path.normpath(default_model_path)

       
        if os.path.isdir(model_name):
            tok_json = os.path.join(model_name, "tokenizer.json")
            if os.path.isfile(tok_json):
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, tokenizer_file=tok_json, local_files_only=True)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        
        self.model.eval()

    def get_embedding(self, text):
        """ 获取文本的向量嵌入 """
      
        device = next(self.model.parameters()).device
    
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    def get_token_embeddings(self, texts, max_length=None, device=None):
        """
        获取token级别的嵌入（用于指针网络输入）
        
        Args:
            texts: 字符串列表
            max_length: 最大长度，None则使用模型默认
            device: 指定设备
        
        Returns:
            dict: 包含 'last_hidden_state', 'input_ids', 'attention_mask'
        """
      
        batch_size_total = len(texts)
        if batch_size_total == 0:
            raise ValueError("texts 不能为空")

        chunk_size = min(8, max(1, batch_size_total // 2)) 

        hidden_states_list = []
        input_ids_list = []
        attention_mask_list = []

        with torch.no_grad():
            for start_idx in range(0, batch_size_total, chunk_size):
                end_idx = min(start_idx + chunk_size, batch_size_total)
                texts_chunk = texts[start_idx:end_idx]

                if max_length is not None:
                    inputs = self.tokenizer(
                        texts_chunk, return_tensors="pt", padding='max_length',
                        truncation=True, max_length=max_length
                    )
                else:
                    inputs = self.tokenizer(texts_chunk, return_tensors="pt", padding=True, truncation=True)

                if device is not None:
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                if device is not None and torch.cuda.is_available():
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)

                hidden_states_list.append(outputs.last_hidden_state)
                input_ids_list.append(inputs['input_ids'])
                attention_mask_list.append(inputs.get('attention_mask', None))

                
                del outputs
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

        last_hidden_state = torch.cat(hidden_states_list, dim=0)
        input_ids = torch.cat(input_ids_list, dim=0)
        attention_mask = None
        if all(m is not None for m in attention_mask_list):
            attention_mask = torch.cat(attention_mask_list, dim=0)

        return {
            'last_hidden_state': last_hidden_state,
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def encode(self, texts, convert_to_tensor=False, device=None):
        """
        兼容SentenceTransformer的encode方法
        支持单个字符串或字符串列表
        
        Args:
            texts: 单个字符串或字符串列表
            convert_to_tensor: 是否返回tensor，默认False返回numpy数组
            device: 如果convert_to_tensor=True，指定tensor的设备
        
        Returns:
            如果convert_to_tensor=False: numpy数组 shape=(n, embedding_dim) 或 (embedding_dim,)
            如果convert_to_tensor=True: torch.Tensor shape=(n, embedding_dim) 或 (embedding_dim,)
        """
        # 处理单个字符串输入
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        # 分块编码
        batch_size_total = len(texts)
        chunk_size = min(16, max(1, batch_size_total // 2))
        emb_list = []

        if device is not None:
            self.model.to(device)

        with torch.no_grad():
            for start_idx in range(0, batch_size_total, chunk_size):
                end_idx = min(start_idx + chunk_size, batch_size_total)
                texts_chunk = texts[start_idx:end_idx]
                inputs = self.tokenizer(texts_chunk, return_tensors="pt", padding=True, truncation=True)
                if device is not None:
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                if device is not None and torch.cuda.is_available():
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)

                emb_list.append(outputs.last_hidden_state.mean(dim=1))
                del outputs
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

        embeddings = torch.cat(emb_list, dim=0)
        
        if convert_to_tensor:
            return embeddings.squeeze(0) if single_input else embeddings
        else:
            result = embeddings.cpu().numpy()
            return result.squeeze(0) if single_input else result

if __name__ == "__main__":
    embedder = EmbeddingModel()
    embedding = embedder.get_embedding("What is the capital of France?")
    print(embedding.shape)
