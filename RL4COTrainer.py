import torch
import torch.nn.functional as F
import numpy as np
import argparse 
import random
import os
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from tensordict.tensordict import TensorDict
from rl4co.models.rl import REINFORCE
from rl4co.utils.trainer import RL4COTrainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar

from MaxSimEnv import MaxSimEnv, get_segments_from_pointers
from MaxSimGenerator import MaxSimGenerator
from AdaptedPointerNetworkPolicy import AdaptedPointerNetworkPolicy
from embedding_model import EmbeddingModel


class ResumeFriendlyREINFORCE(REINFORCE):
    
    
    def on_load_checkpoint(self, checkpoint):
    
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
           
            for key in state_dict:
                if isinstance(state_dict[key], torch.Tensor):
                    state_dict[key] = state_dict[key].clone()
        super().on_load_checkpoint(checkpoint)


def build_vocab_and_tokenize(prompts):
  
    word_to_idx = {'<pad>': 0}
    for prompt in prompts:
        tokens = prompt.lower().split()
        for token in tokens:
            if token not in word_to_idx:
                word_to_idx[token] = len(word_to_idx)
    max_len = max(len(p.lower().split()) for p in prompts)
    padded_prompts = []
    for prompt in prompts:
        tokens = prompt.lower().split()
        padding = ['<pad>'] * (max_len - len(tokens))
        ids = [word_to_idx[token] for token in tokens] + [word_to_idx[p] for p in padding]
        padded_prompts.append(ids)
    return torch.tensor(padded_prompts, dtype=torch.long), word_to_idx, max_len


def load_prompts_from_file(filepath: str) -> list:
    """从 txt 文件加载 prompts，每行一个。"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        if not prompts:
            print(f"错误: 文件 {filepath} 为空或只包含空行。")
            exit(1)
        print(f"成功从 {filepath} 加载 {len(prompts)} 条 prompts。")
        return prompts
    except FileNotFoundError:
        print(f"错误: 文件未找到 {filepath}")
        exit(1)
    except Exception as e:
        print(f"读取文件 {filepath} 时发生错误: {e}")
        exit(1)


if __name__ == '__main__':
    # --- 参数解析 ---
    parser = argparse.ArgumentParser(description="使用 RL4CO 训练 MaxSim 模型")
    parser.add_argument(
        '--train_file',
        type=str,
        required=True,
        help='包含训练 prompts 的 txt 文件路径 (例如: descriptions_train.txt)'
    )
    parser.add_argument(
        '--val_file',
        type=str,
        required=True,
        help='包含验证 prompts 的 txt 文件路径 (例如: descriptions_val.txt)'
    )
    parser.add_argument(
        '--test_file',
        type=str,
        required=True,
        help='包含测试 prompts 的 txt 文件路径 (例如: descriptions_test.txt)'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./checkpoints',
        help='模型检查点（ checkpoints）的保存目录'
    )
 
    parser.add_argument(
        '--resume_from_checkpoint',
        type=str,
        default=None,
        help='从指定的 checkpoint 文件路径恢复训练'
    )
    args = parser.parse_args()
    print(f"[ARGS] train_file={args.train_file}")
    print(f"[ARGS] val_file={args.val_file}")
    print(f"[ARGS] test_file={args.test_file}")
    print(f"[ARGS] checkpoint_dir={args.checkpoint_dir}")
    print(f"[ARGS] resume_from_checkpoint={args.resume_from_checkpoint}")
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    def _print_cuda_mem(tag):
        try:
            if torch.cuda.is_available():
                dev = torch.cuda.current_device()
                name = torch.cuda.get_device_name(dev)
                alloc = torch.cuda.memory_allocated(dev) / (1024**2)
                reserved = torch.cuda.memory_reserved(dev) / (1024**2)
                max_alloc = torch.cuda.max_memory_allocated(dev) / (1024**2)
                print(f"[MEM][{tag}] device={dev}({name}) alloc={alloc:.1f}MB reserved={reserved:.1f}MB max_alloc={max_alloc:.1f}MB")
            else:
                print(f"[MEM][{tag}] CUDA not available")
        except Exception as e:
            print(f"[MEM][{tag}] failed: {e}")

    train_prompts = load_prompts_from_file(args.train_file)
    val_prompts = load_prompts_from_file(args.val_file)
    test_prompts = load_prompts_from_file(args.test_file)
    
  
    MAX_LEN = 512 # 文本最大长度
    MAX_SEGMENTS =6  # 最大分割片段数
    TRAIN_DATA_SIZE = len(train_prompts)
    VAL_DATA_SIZE = len(val_prompts)
    BATCH_SIZE = 24 
    MAX_EPOCHS = 50  # 最大训练轮数


    print("Step 1: Instantiating components...")


    embedding_model = EmbeddingModel()

  
    train_generator = MaxSimGenerator(prompts=train_prompts, max_len=MAX_LEN, embedding_model=embedding_model)
    train_env = MaxSimEnv(generator=train_generator, max_segments=MAX_SEGMENTS, embedding_model=embedding_model)

    val_generator = MaxSimGenerator(prompts=val_prompts, max_len=MAX_LEN, embedding_model=embedding_model)
    val_env = MaxSimEnv(generator=val_generator, max_segments=MAX_SEGMENTS, embedding_model=embedding_model)
    test_generator = MaxSimGenerator(prompts=test_prompts, max_len=MAX_LEN, embedding_model=embedding_model)
    test_env = MaxSimEnv(generator=test_generator, max_segments=MAX_SEGMENTS, embedding_model=embedding_model)

  
    policy = AdaptedPointerNetworkPolicy(train_env, embedding_dim=768, hidden_dim=768, max_segments=MAX_SEGMENTS)
    print("Components instantiated.")


    print("Step 2: Setting up RL algorithm (REINFORCE)...")

    model_kwargs = {
        'env': train_env,
        'policy': policy,
        'baseline': 'rollout',
        'train_data_size': 2560,  
        'val_data_size': VAL_DATA_SIZE,
        'batch_size': BATCH_SIZE,
        'dataloader_num_workers': 0,
        'optimizer_kwargs': {'lr': 1e-4},
      
    }
    
  
    model = ResumeFriendlyREINFORCE(**model_kwargs)
    
    model.strict_loading = False
    print("REINFORCE model configured.")


    print("Step 3: Setting up the trainer...")

    early_stopping_callback = EarlyStopping(
        monitor="val/reward",  
        mode="max",           
        patience=5,         
        verbose=True,        
        min_delta=0.01       
    )


    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='{epoch}-{step}',  
        monitor='val/reward',      
        mode='max',
        save_top_k=1,   
        save_last=True, 
        verbose=True
    )
    progress_bar_callback = RichProgressBar()

 
    logger = TensorBoardLogger("lightning_logs", name="maxsim_model")

    trainer = RL4COTrainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=[2],  
        logger=logger, 
        callbacks=[early_stopping_callback, checkpoint_callback, progress_bar_callback], 
        num_sanity_val_steps=1,  
        check_val_every_n_epoch=5,
        accumulate_grad_batches=2,
        reload_dataloaders_every_n_epochs=1,
    )
    print("RL4COTrainer configured with Early Stopping and Checkpointing.")
    try:
        for cb in trainer.callbacks:
            if isinstance(cb, EarlyStopping):
                print(f"[EarlyStopping] monitor={cb.monitor} mode={cb.mode} patience={cb.patience} min_delta={cb.min_delta}")
    except Exception as e:
        print(f"[EarlyStopping] Introspection failed: {e}")

    try:
        if hasattr(logger, "save_dir"):
            print("TB logger save_dir:", logger.save_dir)
        if hasattr(logger, "name"):
            print("TB logger name:", logger.name)
        if hasattr(logger, "version"):
            print("TB logger version:", logger.version)
    except Exception as e:
        print(f"[Logger] Introspection failed: {e}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"trainable={trainable:,}, frozen={frozen:,}, total={trainable+frozen:,}")
    _print_cuda_mem("after_setup")

    _print_cuda_mem("before_fit")
    print("\nStarting training...")
    val_dataset = val_env.dataset(VAL_DATA_SIZE, phase="val")
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0, collate_fn=val_dataset.collate_fn)
    trainer.fit(
        model,
        val_dataloaders=val_dataloader,
        ckpt_path=args.resume_from_checkpoint
    )
    print("Training finished.")
    test_dataset = test_env.dataset(len(test_prompts), phase="test")
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0, collate_fn=test_dataset.collate_fn)
    trainer.test(model, dataloaders=test_dataloader)

    print("\n--- Step 5: Evaluating trained model on a sample from the locked-box test set ---")
    
   
   
    model.to(train_env.device) 
    model.eval()


    if len(test_prompts) >= 2:
        sample_prompts = random.sample(test_prompts, 2)
        test_prompts_a = [sample_prompts[0]]
        test_prompts_b = [sample_prompts[1]]
        print(f"从测试集中随机抽样进行最终评估:\nA: {test_prompts_a[0]}\nB: {test_prompts_b[0]}")
    else:
        print("警告: 测试集样本数小于2，使用默认 prompts 进行评估。")
        test_prompts_a = ["how to learn pytorch for deep learning"]
        test_prompts_b = ["can you give me a tutorial on pytorch tensors"]
    

    test_generator = MaxSimGenerator(prompts=test_prompts, max_len=MAX_LEN, embedding_model=embedding_model)

    test_generator.lm.to(train_env.device)
    
    inputs_a = test_generator.tokenizer(test_prompts_a, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LEN).to(train_env.device)
    inputs_b = test_generator.tokenizer(test_prompts_b, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LEN).to(train_env.device)
    with torch.no_grad():
        embeds_a = test_generator.lm(**inputs_a).last_hidden_state
        embeds_b = test_generator.lm(**inputs_b).last_hidden_state
    
    test_td = TensorDict({
        "token_embeddings_a": embeds_a,
        "token_embeddings_b": embeds_b,
        "attention_mask_a": inputs_a['attention_mask'],
        "attention_mask_b": inputs_b['attention_mask'],
        "length_a": inputs_a['attention_mask'].sum(dim=1),
        "length_b": inputs_b['attention_mask'].sum(dim=1),
        "input_ids_a": inputs_a['input_ids'],
        "input_ids_b": inputs_b['input_ids'],
    }, batch_size=1)

  
    N_SAMPLES = 20 
    print(f"Generating {N_SAMPLES} samples to find the best segmentation...")
    
    with torch.no_grad():
        out = model.policy(test_td.expand(N_SAMPLES), model.env, phase="test", select_best=False, decode_type="sampling")

    actions_candidates = out['actions']
    
  
    print("Evaluating each sample to find the one with the highest MaxSim score...")
    best_reward = -1.0
    best_action = None

   
    for action_candidate in actions_candidates:
        reward = train_env._get_reward(test_td, action_candidate.unsqueeze(0)).item()
        
 
        if reward > best_reward:
            best_reward = reward
            best_action = action_candidate


    print(f"\nBest segmentation found with score: {best_reward:.4f}")

  
    pointers_a = best_action[:MAX_SEGMENTS].tolist()
    pointers_b = best_action[MAX_SEGMENTS:].tolist()

    la = int(inputs_a['attention_mask'][0].sum().item())
    lb = int(inputs_b['attention_mask'][0].sum().item())
    ids_a = inputs_a['input_ids'][0, :la]
    ids_b = inputs_b['input_ids'][0, :lb]
    tok = test_generator.tokenizer

    def _decode_segments(ids, pointers):
        length = ids.size(0)
        pointers = [min(max(0, int(p)), length - 1) for p in pointers]
        bounds = [0] + sorted(set(pointers))
        segs = []
        start = 0
        for p in bounds:
            end = p + 1
            if end > start:
                segs.append(tok.decode(ids[start:end], skip_special_tokens=True).strip())
            start = end
        if start < length:
            segs.append(tok.decode(ids[start:length], skip_special_tokens=True).strip())
        return segs

    segments_a = _decode_segments(ids_a, pointers_a)
    segments_b = _decode_segments(ids_b, pointers_b)

    N_PRINT = 128
    try:
        tokens_a = tok.convert_ids_to_tokens(ids_a.tolist())
    except Exception:
        tokens_a = tok.tokenize(test_prompts_a[0])[:la]
    try:
        tokens_b = tok.convert_ids_to_tokens(ids_b.tolist())
    except Exception:
        tokens_b = tok.tokenize(test_prompts_b[0])[:lb]

    def _print_tokens(label, tokens, pointers, max_n=128):
        n = min(len(tokens), max_n)
        print(f"\nTokens {label} (first {n}):")
        ptr_set = set([min(max(0, int(p)), len(tokens) - 1) for p in pointers])
        for i in range(n):
            mark = "*" if i in ptr_set else " "
            print(f"  {i:>3}{mark}: {tokens[i]}")
        ptr_info = ", ".join([f"{i}:{tokens[i]}" for i in sorted(ptr_set) if i < n])
        print(f"Pointer tokens {label}: {ptr_info}")

    print("\n--- Prompt A ---")
    print(f"Original: '{test_prompts_a[0]}'")
    _print_tokens("A", tokens_a, pointers_a, N_PRINT)
    print("Segments:")
    for i, seg in enumerate(segments_a):
        print(f"  {i+1}: '{seg}'")
        
    print("\n--- Prompt B ---")
    print(f"Original: '{test_prompts_b[0]}'")
    _print_tokens("B", tokens_b, pointers_b, N_PRINT)
    print("Segments:")
    for i, seg in enumerate(segments_b):
        print(f"  {i+1}: '{seg}'")
        
    #     python RL4COTrainer.py \
    # --train_file ../dataset/descriptions_train.txt \
    # --val_file ../dataset/descriptions_val.txt \
    # --test_file ../dataset/descriptions_test.txt \
    # --checkpoint_dir ./my_model_checkpoints
    
    #torchrun --nproc_per_node=4 RL4COTrainer.py --train_file ../dataset/descriptions_train.txt --val_file ../dataset/descriptions_val.txt --test_file ../dataset/descriptions_test.txt --checkpoint_dir ./my_model_checkpoints
    
    # python RL4COTrainer.py \
    # --train_file /path/to/descriptions_train.txt \
    # --val_file /path/to/descriptions_val.txt \
    # --test_file /path/to/descriptions_test.txt \
    # --checkpoint_dir ./my_model_checkpoints \
    # --resume_from_checkpoint ./my_model_checkpoints/last.ckpt 