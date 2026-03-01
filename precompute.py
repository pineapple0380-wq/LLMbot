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
from sklearn.metrics import silhouette_score, davies_bouldin_index
import warnings
import wandb

warnings.filterwarnings('ignore')

class Qwen3EmbeddingGenerator:
    """
    Qwen3-Embedding-8B Generator with Scientific Controls
    """
    def __init__(self, model_path='Qwen/Qwen3-Embedding-8B', device='cuda', batch_size=32, pooling='last', use_wandb=True):
        """
        Args:
            pooling: 'mean' (Average all tokens) or 'last' (EOS token - recommended for some generative models)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.pooling_strategy = pooling
        self.use_wandb = use_wandb and wandb is not None

        if self.use_wandb:
            # Check if run is already initialized to avoid errors
            if wandb.run is None:
                wandb.init(project="lmbot-qwen3", name=f"embed_gen_{pooling}")
        
        print(f"[System] Loading Backbone: {model_path}")
        print(f"[System] Pooling Strategy: {self.pooling_strategy.upper()}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
        ).to(self.device)
        self.model.eval()
        
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
        batch_input = [self.instruction + self._clean_text(t) for t in texts]
        
        inputs = self.tokenizer(
            batch_input, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # 2. Apply Selected Pooling Strategy
            if self.pooling_strategy == 'last':
                embeddings = self._last_token_pooling(outputs.last_hidden_state, inputs['attention_mask'])
            else:
                embeddings = self._mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
            
            # 3. CRITICAL: L2 Normalization
            # Required for Cosine Similarity to work in GNNs/Clustering
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu()

    def generate_embeddings(self, raw_texts):
        print(f"[Process] Encoding {len(raw_texts)} users...")
        all_embeddings = []
        
        iterator = range(0, len(raw_texts), self.batch_size)
        for i in tqdm(iterator, desc="Inference"):
            batch_raw = raw_texts[i:i + self.batch_size]
            emb = self.encode_batch(batch_raw)
            all_embeddings.append(emb)
            
            if self.use_wandb and i % (self.batch_size * 10) == 0:
                wandb.log({"progress": i / len(raw_texts)})
            
        return torch.cat(all_embeddings, dim=0)

def analyze_structural_weakness(embeddings, labels, save_dir):
    """
    Generates the 'Motivation' metrics for Figure 1.
    """
    print("\n[Analysis] Calculating Structural Metrics...")
    
    # Filter out unlabeled data (-1 or nulls) if necessary
    # Assuming labels are integers 0 (Human) and 1 (Bot)
    valid_mask = (labels != -1)
    if not np.any(valid_mask):
        print("[Warning] No valid labels found. Skipping analysis.")
        return

    X = embeddings[valid_mask].numpy()
    y = labels[valid_mask]
    
    # 1. Metrics
    sil_score = silhouette_score(X, y)
    db_score = davies_bouldin_index(X, y)
    
    print(f"\n{'='*40}")
    print(f"MOTIVATION METRICS")
    print(f"{'='*40}")
    print(f"Silhouette Score: {sil_score:.4f} (Low = Motivation Validated)")
    print(f"Davies-Bouldin:   {db_score:.4f} (High = Motivation Validated)")
    print(f"{'='*40}\n")
    
    if wandb.run is not None:
        wandb.log({
            "motivation/silhouette": sil_score,
            "motivation/davies_bouldin": db_score
        })

    # 2. t-SNE Visualization
    print("[Analysis] Generating t-SNE Plot...")
    # PCA first for speed
    X_pca = PCA(n_components=50).fit_transform(X) if X.shape[1] > 50 else X
    
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X_pca)
    
    plt.figure(figsize=(10, 8))
    # Plot Humans
    plt.scatter(X_embedded[y==0, 0], X_embedded[y==0, 1], c='#1f77b4', alpha=0.6, label='Human', s=10)
    # Plot Bots
    plt.scatter(X_embedded[y==1, 0], X_embedded[y==1, 1], c='#d62728', alpha=0.6, label='Bot', s=10)
    
    plt.title(f"Qwen3-Embedding Space (Silhouette: {sil_score:.3f})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plot_path = Path(save_dir) / "motivation_tsne_qwen3.png"
    plt.savefig(plot_path, dpi=300)
    print(f"[Output] Plot saved to: {plot_path}")
    
    if wandb.run is not None:
        wandb.log({"motivation/plot": wandb.Image(str(plot_path))})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./datasets/TwiBot-20')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--pooling', type=str, default='last', choices=['mean', 'last'], 
                        help="Pooling strategy: 'mean' or 'last' (EOS)")
    parser.add_argument('--no_wandb', action='store_true', help="Disable WandB")
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
        use_wandb=not args.no_wandb
    )
    embeddings = generator.generate_embeddings(texts)
    
    # 4. Save
    output_filename = f"qwen3_emb_{args.pooling}.pt"
    output_path = base_path / output_filename
    torch.save(embeddings, output_path)
    print(f"[Output] Embeddings saved to {output_path}")
    
    # 5. Run Motivation Analysis
    # This is the "Professor's Requirement" - prove the weakness immediately
    analyze_structural_weakness(embeddings, labels, base_path)
    
    if wandb.run is not None:
        wandb.finish()

if __name__ == '__main__':
    main()