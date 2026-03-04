"""
Stage 1: Structural Weakness Validation & Embedding Generation
Paper Component: "Figure 1: Motivation Analysis"
Model: Qwen3-Embedding-8B (Instruction-Tuned)
Method: Natural Language Prompting with Multi-Strategy Pooling
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import json
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
import wandb
import csv

warnings.filterwarnings('ignore')

def analyze_structural_weakness(embeddings_dict, labels, save_dir, instruction_mode):
    """
    升级版：遍历所有层计算指标，生成倒U曲线证据 CSV，并为最后一层生成 t-SNE。
    """
    print("\n[Analysis] Calculating Layer-wise Structural Metrics...")
    
    valid_mask = (labels != -1)
    if not np.any(valid_mask):
        print("[Warning] No valid labels found. Skipping analysis.")
        return

    y = labels[valid_mask]
    metrics = []

    # 按层深排序进行评估
    sorted_keys = sorted(embeddings_dict.keys(), key=lambda x: int(x) if int(x) >= 0 else 999)
    last_layer_key = str(sorted_keys[-1])

    for layer_key in sorted_keys:
        X = embeddings_dict[layer_key][valid_mask].numpy()
        
        # 为了应对高维距离诅咒，先降维再算指标能更好反映流形结构
        X_pca = PCA(n_components=50).fit_transform(X) if X.shape[1] > 50 else X
        
        sil_score = silhouette_score(X_pca, y)
        db_score = davies_bouldin_score(X_pca, y) # 已修复旧版函数名问题
        
        print(f"Layer {layer_key:>3s} | Silhouette: {sil_score:.4f} | DB: {db_score:.4f}")
        metrics.append({"layer": layer_key, "silhouette": sil_score, "davies_bouldin": db_score})
        
        if wandb.run is not None:
            wandb.log({
                f"motivation/L{layer_key}_silhouette": sil_score,
                f"motivation/L{layer_key}_davies_bouldin": db_score
            })

    # 保存多层表现 CSV，用于绘制论文里的 "倒 U 型" 曲线
    csv_path = Path(save_dir) / f"layer_metrics_instr_{instruction_mode}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "silhouette", "davies_bouldin"])
        writer.writeheader()
        writer.writerows(metrics)
    print(f"\n[Output] Metric curve data saved to: {csv_path}")

    # =================保留原版的 t-SNE 可视化功能=================
    print(f"[Analysis] Generating t-SNE Plot for Last Target Layer ({last_layer_key})...")
    X_last = embeddings_dict[last_layer_key][valid_mask].numpy()
    X_pca_last = PCA(n_components=50).fit_transform(X_last) if X_last.shape[1] > 50 else X_last
    
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X_pca_last)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_embedded[y==0, 0], X_embedded[y==0, 1], c='#1f77b4', alpha=0.6, label='Human', s=10)
    plt.scatter(X_embedded[y==1, 0], X_embedded[y==1, 1], c='#d62728', alpha=0.6, label='Bot', s=10)
    
    # 动态获取最后一层的指标用于标题
    final_sil = next(m['silhouette'] for m in metrics if m['layer'] == last_layer_key)
    plt.title(f"Qwen3-Embedding Space - Layer {last_layer_key} (Silhouette: {final_sil:.3f})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plot_path = Path(save_dir) / f"motivation_tsne_qwen3_L{last_layer_key}_instr_{instruction_mode}.png"
    plt.savefig(plot_path, dpi=300)
    print(f"[Output] Plot saved to: {plot_path}")


def locate_blocks(model):
    """动态寻找模型的 Transformer 层，以便挂载 Hook"""
    for attr_chain in [("model", "layers"), ("layers",), ("transformer", "h")]:
        cur = model
        ok = True
        for a in attr_chain:
            if not hasattr(cur, a):
                ok = False; break
            cur = getattr(cur, a)
        if ok and isinstance(cur, (torch.nn.ModuleList, list)):
            return cur
    raise RuntimeError("Cannot locate transformer blocks in the provided model")

class Qwen3EmbeddingGenerator:
    """
    Qwen3-Embedding-8B Generator with Scientific Controls
    """
    def __init__(self, model_path='Qwen/Qwen3-Embedding-8B', device='cuda', batch_size=32, pooling='last',target_layers=None, instruction_mode='on', use_wandb=True):
        """
        Args:
            pooling: 'mean' (Average all tokens) or 'last' (EOS token - recommended for some generative models)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.pooling_strategy = pooling
        self.instruction_mode = instruction_mode  # 'on', 'off', 'noise'
        self.use_wandb = use_wandb and wandb is not None

        if self.use_wandb:
            # Check if run is already initialized to avoid errors
            if wandb.run is None:
                wandb.init(project="lmbot-qwen3", name=f"embed_gen_{pooling}")
        
        print(f"[System] Loading Backbone: {model_path}")
        print(f"[System] Pooling Strategy: {self.pooling_strategy.upper()}")
        print(f"[System] Instruction Mode: {self.instruction_mode.upper()}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
            device_map="auto"
        )
        self.model.eval()

        self.blocks = locate_blocks(self.model)
        self.L = len(self.blocks) # 精准读取模型层数 (通常为 36)
        print(f"[Architecture] Detected {self.L} Transformer Blocks.")

        # 自动读取层数与层位解析
        if target_layers == 'sweep':
            # 扫描模式：提取后半段的多个层位，寻找倒 U 曲线的顶点
            self.target_layers_raw = list(range(self.L // 2, self.L, 2))
            if -1 not in self.target_layers_raw and (self.L - 1) not in self.target_layers_raw:
                self.target_layers_raw.append(-1)
        elif target_layers:
            self.target_layers_raw = [int(x.strip()) for x in target_layers.split(",")]
        else:
            # 默认提取 Late-middle 到 Final 层
            self.target_layers_raw = [self.L // 2, int(0.72 * self.L), int(0.85 * self.L), -1]

        # 将负数索引转化为正数索引
        self.layer_ids = [(i if i >= 0 else self.L + i) for i in self.target_layers_raw]
        assert all(0 <= i < self.L for i in self.layer_ids), "Layer index out of bounds."
        print(f"[Config] Target Layers (Raw): {self.target_layers_raw}")
        print(f"[Config] Resolved Target Layers: {self.layer_ids}")
        
        # Scientific Control: Task-Specific Instruction
        self.instruction = "Instruct: Classify this social media user based on their profile and behavior to detect automation.\nInput: "
        
    def _clean_text(self, text):
        """
        Methodology: Text Reconstruction from Legacy Formats
        """
        if not isinstance(text, str): return ""
        
        replacements = {
            " </s> ": "\n",           
            "METADATA:": "## Profile Metadata:\n",
            "DESCRIPTION:": "\n## User Bio:\n",
            "TWEET:": "\n## Recent Tweets:\n",
            "HTTPURL": "[Link]",      
            "#HASHTAG": "[Tag]",
            "EMOJI": "[Emoji]",
            "@USER": "@User"
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text.strip()

    def _mean_pooling(self, hidden_states, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def _last_token_pooling(self, hidden_states, attention_mask):
        """
        Extracts the last valid token (EOS) embedding.
        Often superior for Generative LLMs trained on sequence completion.
        """
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]
        # Gather the last non-padding token
        return hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]

    def encode_batch(self, texts):
        # 1. Apply Instruction + Cleaning
        batch_input = []
        for t in texts:
            cleaned = self._clean_text(t)
            if self.instruction_mode == 'on':
                batch_input.append(self.instruction + cleaned)
            elif self.instruction_mode == 'off':
                batch_input.append(cleaned)
            elif self.instruction_mode == 'noise':
                batch_input.append("Ignore instructions. Random input: \nInput: " + cleaned)
        
        inputs = self.tokenizer(
            batch_input, padding=True, truncation=True, 
            max_length=512, return_tensors='pt'
        ).to(self.device)
        
        # Forward Hook to capture hidden states at target layers
        captured_hiddens = {}
        hooks = []

        def mk_hook(lid):
            def hook_fn(module, inp, out):
                h = out[0] if isinstance(out, (tuple, list)) else out
                captured_hiddens[lid] = h
            return hook_fn

        for lid in set(self.layer_ids):
            hooks.append(self.blocks[lid].register_forward_hook(mk_hook(lid)))
            
        with torch.no_grad():
            _ = self.model(**inputs, return_dict=True)
            
        layer_embeddings = {}
        for raw_id, res_id in zip(self.target_layers_raw, self.layer_ids):
            h = captured_hiddens[res_id]
            
            if self.pooling_strategy == 'last':
                emb = self._last_token_pooling(h, inputs['attention_mask'])
            else:
                emb = self._mean_pooling(h, inputs['attention_mask'])
                
            emb = F.normalize(emb, p=2, dim=1)
            layer_embeddings[str(raw_id)] = emb.cpu()
            
        # 释放资源防止 OOM
        for hk in hooks:
            hk.remove()
        captured_hiddens.clear()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        return layer_embeddings

    def generate_embeddings(self, raw_texts):
        print(f"[Process] Encoding {len(raw_texts)} users...")
        all_embeddings = {str(k): [] for k in self.target_layers_raw}
        
        iterator = range(0, len(raw_texts), self.batch_size)
        for i in tqdm(iterator, desc="Inference"):
            batch_raw = raw_texts[i:i + self.batch_size]
            batch_embs = self.encode_batch(batch_raw)
            
            for k, v in batch_embs.items():
                all_embeddings[k].append(v)
            
            if self.use_wandb and i % (self.batch_size * 10) == 0:
                wandb.log({"progress": i / len(raw_texts)})
            
        for k in all_embeddings.keys():
            all_embeddings[k] = torch.cat(all_embeddings[k], dim=0)
            
        return all_embeddings


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./datasets/TwiBot-20')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--pooling', type=str, default='last', choices=['mean', 'last'], 
                        help="Pooling strategy: 'mean' or 'last' (EOS)")
    parser.add_argument('--no_wandb', action='store_true', help="Disable WandB")
    
    # 新增实验控制参数
    parser.add_argument('--layers', type=str, default=None, 
                        help='Comma-separated layer indices (e.g., "18,26,30,-1") or "sweep"')
    parser.add_argument('--instruction_mode', type=str, default='on', choices=['on', 'off', 'noise'],
                        help="Controls the prompt prefix to test robustness.")
    args = parser.parse_args()
    
    base_path = Path(args.dataset_path)
    
    # 1. Load Texts
    print(f"[Data] Loading texts from {base_path}")
    with open(base_path / 'norm_user_text.json', 'r') as f:
        text_data = json.load(f)
        if isinstance(text_data, dict):
            # Sort keys to ensure alignment
            ids = sorted(text_data.keys(), key=lambda x: int(x))
            texts = [text_data[k] for k in ids]
        else:
            texts = text_data

    # 2. Load Labels (CRITICAL for Analysis)
    # TwiBot-20 stores labels in node.json. We must map them.
    print(f"[Data] Loading labels from {base_path}")
    labels = []
    try:
        with open(base_path / 'node.json', 'r') as f:
            nodes = json.load(f)
            # Map labels: human=0, bot=1, unlabeled=-1
            label_map = {'human': 0, 'bot': 1}
            for n in nodes:
                # Ensure we only check 'user' nodes if the file contains other types
                if 'label' in n:
                    labels.append(label_map.get(n['label'], -1))
                else:
                    labels.append(-1)
        labels = np.array(labels)
    except FileNotFoundError:
        print("[Warning] node.json not found. Creating dummy labels (Analysis will be skipped).")
        labels = np.full(len(texts), -1)

    # 3. Generate
    generator = Qwen3EmbeddingGenerator(
        batch_size=args.batch_size, 
        pooling=args.pooling,
        target_layers=args.layers,
        instruction_mode=args.instruction_mode,
        use_wandb=not args.no_wandb
    )

    embeddings_dict = generator.generate_embeddings(texts)
    
    # 4. Save
    for layer_id, emb in embeddings_dict.items():
        output_filename = f"qwen3_emb_{args.pooling}_L{layer_id}_instr_{args.instruction_mode}.pt"
        output_path = base_path / output_filename
        torch.save(emb, output_path)
        print(f"[Output] Embeddings saved to {output_path}")
    
    # 5. Run Motivation Analysis
    # This is the "Professor's Requirement" - prove the weakness immediately
    analyze_structural_weakness(embeddings_dict, labels, base_path, args.instruction_mode)
    
    if wandb.run is not None:
        wandb.finish()

if __name__ == '__main__':
    main()