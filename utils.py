import torch
import torch.nn as nn
import torch.nn.functional as F
import random, os
import numpy as np
import pandas as pd
import wandb
import json
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from torch_geometric.utils import add_self_loops, remove_self_loops, scatter
from torch_geometric.utils import degree as calc_degree



def compute_feature_homophily(x, edge_index):
    """
    计算每个节点与其邻居的特征余弦相似度均值 (Feature Homophily)
    无需标签，完全合法。
    """
    row, col = edge_index
    
    # 1. 归一化特征 (为了快速计算 Cosine Similarity)
    x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
    
    # 2. 计算每条边的相似度: dot(x_i, x_j)
    edge_sim = (x_norm[row] * x_norm[col]).sum(dim=-1)
    
    # 3. 聚合到节点 (取平均)
    homophily = scatter(edge_sim, col, dim=0, dim_size=x.size(0), reduce='mean')
    
    return homophily.unsqueeze(1)

def load_weights(target_module, state_dict, module_name="Module"):
    """
    智能权重加载器：自动处理 key 前缀不匹配的问题 (vib.encoder vs encoder)
    """
    target_keys = set(target_module.state_dict().keys())
    source_keys = set(state_dict.keys())
    
    # 1. 尝试直接加载
    if target_keys.intersection(source_keys):
        print(f"  [{module_name}] Direct match detected.")
        new_dict = {k: v for k, v in state_dict.items() if k in target_keys}
    
    # 2. 尝试移除前缀 (例如 saved: 'vib.encoder.weight' -> target: 'encoder.weight')
    else:
        print(f"  [{module_name}] Prefix mismatch detected. Attempting auto-fix...")
        new_dict = {}
        for k, v in state_dict.items():
            # 常见的前缀修复逻辑
            clean_k = k
            if k.startswith('vib.'): clean_k = k.replace('vib.', '')
            elif k.startswith('model.'): clean_k = k.replace('model.', '')
            elif k.startswith('fusion.'): clean_k = k.replace('fusion.', '')
            elif k.startswith('gnn.'): clean_k = k.replace('gnn.', '')
            
            # 如果去前缀后在目标里，就由它了
            if clean_k in target_keys:
                new_dict[clean_k] = v
                
    # 3. 执行加载
    if len(new_dict) > 0:
        msg = target_module.load_state_dict(new_dict, strict=False)
        print(f"  ✅ [{module_name}] Loaded {len(new_dict)} layers. Missing: {len(msg.missing_keys)}")
        return True
    else:
        print(f"  ❌ [{module_name}] Failed to match any keys. Target keys example: {list(target_keys)[:3]}")
        return False



def jsd_probs(p, q, eps=1e-9):
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    m = 0.5 * (p + q)
    kl_pm = F.kl_div(m.log(), p, reduction="none").sum(dim=1, keepdim=True)
    kl_qm = F.kl_div(m.log(), q, reduction="none").sum(dim=1, keepdim=True)
    return 0.5 * (kl_pm + kl_qm)

def kl_divergence_dirichlet(alpha, num_classes=2):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=alpha.device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    return first_term + second_term

def analyze_uncertainty_dist(df, split_name='Dataset'):
    """
    [SeGA Statistics] 
    分别统计三个数据集上，不同 Case 的多模态不确定性分布。
    """
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.4f}'.format)

    print("\n" + "-"*80)
    print(f"🧐 [Deep Dive] Uncertainty Distribution on {split_name.upper()} Set")
    print("-"*(80))
    
    # 定义我们要观察的物理量
    metrics = [
        'jsd',             # 冲突度 (Conflict)
        'entropy_text',    # 文本不确定性 (Text Uncertainty)
        'entropy_graph',   # 图不确定性 (Graph Uncertainty)
        'conf_text',       # 文本置信度 (Text Confidence)
        'conf_graph',      # 图置信度 (Graph Confidence)
        'homophily'        # 结构异常度 (Structural Anomaly)
    ]
    
    # 分组计算 Mean 和 Std
    # 这样你能看到分布的中心和离散程度
    stats = df.groupby('case_type')[metrics].agg(['mean', 'std'])
    
    print(stats)
    print("-"*(80))

    # --- 自动生成专家结论 (Auto-Insight) ---
    try:
        # 获取 Graph_Only (Text错, Graph对) 和 Easy (都对) 的均值
        mean_stats = df.groupby('case_type')[metrics].mean()
        
        if 'Graph_Only' in mean_stats.index and 'Easy' in mean_stats.index:
            g_only = mean_stats.loc['Graph_Only']
            easy = mean_stats.loc['Easy']
            
            print("\n💡 [Auto-Insight for Graph_Only Cases]")
            
            # 1. 检查冲突度
            if g_only['jsd'] > easy['jsd'] * 1.5:
                print(f"✅ High Conflict: JSD ({g_only['jsd']:.3f}) is significantly higher than Easy ({easy['jsd']:.3f}).")
                print("   -> Conclusion: The model 'senses' the disagreement. Use JSD for gating!")
            else:
                print(f"⚠️ Low Conflict: JSD ({g_only['jsd']:.3f}) is similar to Easy.")
            
            # 2. 检查 Text 犹豫度
            if g_only['entropy_text'] > easy['entropy_text'] * 1.5:
                print(f"✅ Text Hesitation: Text Entropy ({g_only['entropy_text']:.3f}) is higher than Easy.")
                print("   -> Conclusion: LLM is uncertain when it is wrong. Use Entropy for gating!")
            else:
                print(f"⚠️ Overconfidence: Text Entropy ({g_only['entropy_text']:.3f}) is low even when wrong.")
                print("   -> Conclusion: LLM is hallucinating confidently.")

            # 3. 检查 Graph 自信度
            if g_only['entropy_graph'] < g_only['entropy_text']:
                 print(f"✅ Graph Confidence: Graph is more certain (Ent={g_only['entropy_graph']:.3f}) than Text.")
    
    except Exception as e:
        print(f"Could not generate insights: {e}")
        
    print("="*60 + "\n")
    
def calculate_structural_metrics( x, edge_index):
        """
        [Helper] 计算节点的结构属性，用于深入诊断。
        1. Degree (Log-normalized)
        2. Feature Homophily (邻居语义相似度)
        """
        num_nodes = x.shape[0]
        row, col = edge_index
        
        # --- Metric 1: Degree (入度 + 出度) ---
        # 简单起见，我们计算无向度或总度数
        # 如果是有向图，Bot 检测中 "In-Degree" (粉丝数) 通常更重要，这里我们算 Total
        deg = calc_degree(col, num_nodes=num_nodes, dtype=torch.float)
        # Log degree 用于分析更方便 (Power-law 分布)
        log_deg = torch.log1p(deg)
        
        # --- Metric 2: Feature Homophily (语义同质性) ---
        # 计算每个节点与其邻居的平均 Cosine Similarity
        # Algorithm: Scatter Mean of CosineSim(x_src, x_dst)
        
        # 1. 获取边两端的特征
        x_src = x[row]
        x_dst = x[col]
        
        # 2. 计算每条边的相似度
        edge_sim = F.cosine_similarity(x_src, x_dst, dim=1)
        
        # 3. 聚合到目标节点 (dst)
        # 使用 torch_geometric.utils.scatter (如果版本旧可能在 torch_scatter)
        # 如果没有安装 torch_scatter，可以用简单的 index_add_ 实现
        
        # out[i] = mean(sim(j, i)) for j in neighbors(i)
        # 对于孤立点，结果为 0 (或者我们需要设为 1? 设为 0 表示没有邻居支持)
        node_homophily = scatter(edge_sim, col, dim=0, dim_size=num_nodes, reduce='mean')
        
        return log_deg, node_homophily

def get_stats(logits):
                probs = F.softmax(logits, dim=1)
                conf, preds = probs.max(dim=1)
                # 计算熵 (不确定性)
                log_probs = F.log_softmax(logits, dim=1)
                entropy = -(probs * log_probs).sum(dim=1)
                return preds, conf, entropy

def plot_gate_vs_uncertainty(df, save_path):
    """
    绘制 Gate 响应曲线：检查 Gate 是否随着 Text 不确定性的增加而把权重分给 Graph
    """
    plt.figure(figsize=(10, 6))
    
    # 我们关注 Opportunity 样本，因为这些是 Gate 本应该起作用但没起作用的地方
    sns.scatterplot(
        data=df, 
        x='entropy', 
        y='gate', 
        hue='case_type',
        style='case_type',
        alpha=0.7,
        palette={'Clean':'grey', 'Opportunity (Text Fail)':'red', 'Risk (Graph Fail)':'blue', 'Hard (Both Fail)':'orange'}
    )
    
    # 画出理想的趋势线（示意）或实际的回归线
    sns.regplot(data=df, x='entropy', y='gate', scatter=False, color='black', line_kws={'linestyle':'--'}, label='Trend')
    
    plt.title("Diagnosis: Does Gate Respond to Text Uncertainty?")
    plt.xlabel("Text Entropy (Higher = More Uncertain)")
    plt.ylabel("Gate Value (Higher = Trust Text)")
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"📊 Chart saved to {save_path}")

class EdlLoss(nn.Module):
    """
    Evidential Deep Learning Loss for BotUMC
    Objective: Minimize prediction error + Maximize uncertainty for errors
    """
    def __init__(self, annealing_step=10):
        super().__init__()
        # 移除 self.epoch_num，因为我们会从 forward 中动态传入
        self.annealing_step = annealing_step

    def forward(self, alpha, y, epoch_num, total_epochs=None):
        """
        参数:
        alpha: [Batch, K] - Dirichlet 分布参数
        y: [Batch] - 真实标签
        epoch_num: int - 当前 epoch 数 (用于计算 KL 权重退火)
        total_epochs: int - 总 epoch 数 (虽然传入了，但标准 EDL 只需要当前 epoch)
        """
        
        # S = sum(alpha)
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        # A. Bayesian Risk (Log Likelihood equivalent)
        # L = sum(y_i * (log(S) - log(alpha_i)))
        # 这里的 loss 实际上是 Negative Log-Likelihood of the Marginal Likelihood
        y_hot = F.one_hot(y, num_classes=alpha.shape[1]).float()
        loss_nll = torch.sum(y_hot * (torch.log(S) - torch.log(alpha)), dim=1).mean()
        
        # B. KL Divergence Regularization
        # Force incorrect predictions to have flat Dirichlet distribution (high uncertainty)
        # alpha_tilde: 对正确类别保持原样，对错误类别设为 1 (Uniform)
        alpha_tilde = y_hot + (1 - y_hot) * alpha
        
        # KL(Dir(alpha_tilde) || Dir(1))
        kl = self._kl_divergence(alpha_tilde, alpha.shape[1])
        
        # Annealing: Gradually increase KL weight
        # 使用传入的 epoch_num 计算系数
        annealing_coef = min(1.0, epoch_num / self.annealing_step)
        
        return loss_nll + annealing_coef * kl.mean()

    def _kl_divergence(self, alpha, num_classes):
        beta = torch.ones((1, num_classes), device=alpha.device)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        
        kl = lnB + lnB_uni + torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True)
        return kl

class VariationalTextEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, 2) # Classification

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_var(h)
        
        # 采样潜在变量 z
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z)
        
        # 计算 KL 散度 (作为正则项加到 Loss 里)
        # 限制 z 接近标准正态分布，防止过拟合
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        # 返回 sigma 作为不确定性指标！
        uncertainty = torch.exp(0.5 * logvar).mean(dim=1, keepdim=True)
        
        return logits, kl_loss, uncertainty

class BadCaseAnalyzer:
    def __init__(self, raw_data_path, node_id_map):
        """
        raw_data_path: 原始的 node.json 路径 (包含 description, tweets 等)
        node_id_map: 一个 list 或 dict, 映射 model 的 index 到原始数据的 id
                     例如: dataset.node_ids (test set ordered)
        """
        self.raw_data = self._load_raw_data(raw_data_path)
        self.node_id_map = node_id_map
        
    def _load_raw_data(self, path):
        # 假设是 Twibot-20 标准的 json 格式
        print(f"Loading raw data from {path}...")
        with open(path, 'r') as f:
            data = json.load(f)
        #以此建立 ID -> Data 的快速索引
        return {item['id']: item for item in data}

    def get_user_info(self, idx):
        """获取指定测试集索引的原始用户信息"""
        real_id = self.node_id_map[idx] # 获取真实推特ID
        info = self.raw_data.get(real_id, {})
        
        return {
            "id": real_id,
            "label": "Bot" if info.get('label') == 'bot' else "Human",
            "description": info.get('description', 'N/A'),
            "tweets": info.get('tweet', [])[:3], # 只看前3条
            "followers": info.get('public_metrics', {}).get('follower_count', 0),
            "following": info.get('public_metrics', {}).get('following_count', 0)
        }

    def analyze(self, logits_llm, logits_gnn, labels, indices):
        """
        logits_llm: Tensor [N, 2]
        logits_gnn: Tensor [N, 2]
        labels: Tensor [N]
        indices: 原始测试集在全集中的索引 (Global Indices)
        """
        preds_llm = torch.argmax(logits_llm, dim=1).cpu().numpy()
        preds_gnn = torch.argmax(logits_gnn, dim=1).cpu().numpy()
        targets = labels.cpu().numpy()
        indices = indices.cpu().numpy() # 这里的 indices 是 batch 里的

        # Boolean Masks
        correct_llm = (preds_llm == targets)
        correct_gnn = (preds_gnn == targets)

        # 1. Text 错, Graph 对 (我们最希望 GNN 救回来的)
        # "Opportunity Cases"
        group_1_mask = (~correct_llm) & (correct_gnn)
        
        # 2. Graph 错, Text 对 (融合不当容易被 GNN 带偏的)
        # "Risk Cases"
        group_2_mask = (correct_llm) & (~correct_gnn)

        # 3. 都错 (Hard Samples)
        group_3_mask = (~correct_llm) & (~correct_gnn)

        results = {
            "group1_indices": indices[group_1_mask],
            "group2_indices": indices[group_2_mask],
            "group3_indices": indices[group_3_mask],
        }
        
        print(f"\n=== Bad Case Statistical Overview ===")
        print(f"Total Samples: {len(targets)}")
        print(f"Group 1 (Text Wrong / Graph Right): {len(results['group1_indices'])} (Opportunity)")
        print(f"Group 2 (Text Right / Graph Wrong): {len(results['group2_indices'])} (Risk)")
        print(f"Group 3 (Both Wrong): {len(results['group3_indices'])} (Hard)")
        print("=====================================\n")
        
        return results

    def print_details(self, global_idx, case_type):
        """打印单个用户的详细侦查报告"""
        info = self.get_user_info(global_idx)
        
        print(f"--- [Case Type: {case_type}] ID: {info['id']} ---")
        print(f"GT Label: {info['label']}")
        print(f"Stats: Follower {info['followers']} | Following {info['following']}")
        print(f"Description: {info['description']}")
        print(f"Tweets (Sample):")
        for i, t in enumerate(info['tweets']):
            print(f"  [{i+1}] {t}")
        print("--------------------------------------------------")

def relation_aware_knn_pruning(data_dict, embeddings, k=5, z_score_threshold=0.0):
    """
    [SeGA v4.0 Core] Local Z-score/Top-K Pruning.
    修复了 edge_type 维度爆炸的 Bug，确保边索引和边类型严格对齐。
    """
    print(f"\n[Graph Refinement] Starting Relation-Aware KNN Pruning (K={k})...")
    
    edge_index = data_dict['edge_index']
    edge_type = data_dict['edge_type']
    device = edge_index.device
    num_nodes = embeddings.size(0)

    # 容器用于存放筛选后的边
    refined_edges_list = []
    refined_types_list = []
    
    # 移动 Embedding 到同一设备用于计算
    embeddings = embeddings.to(device)

    mask = torch.zeros(edge_index.shape[1], dtype=torch.bool, device=device)

    # 获取图中包含的所有关系类型 (通常是 0 和 1)
    unique_relations = torch.unique(edge_type)

    total_kept = 0
    
    for r_type in unique_relations:
        # 1. 提取当前关系的所有边
        # mask shape: [E_total], sub_edges shape: [2, E_rel]
        r_mask = (edge_type == r_type)
        r_edges = edge_index[:, r_mask]

        global_indices = torch.where(r_mask)[0]
        
        if r_edges.shape[1] == 0:
            continue
            
        src_nodes = r_edges[0]
        dst_nodes = r_edges[1]
        
        # 2. 计算余弦相似度 (Element-wise)
        emb_src = F.normalize(embeddings[src_nodes], p=2, dim=1)
        emb_dst = F.normalize(embeddings[dst_nodes], p=2, dim=1)
        sims = torch.sum(emb_src * emb_dst, dim=1)

        
        # 3. 执行 Top-K 筛选
        unique_src = torch.unique(src_nodes)
        
        nodes_processed = 0
        edges_kept_local = 0
        
        # 但考虑到 TwiBot-20 规模尚可，此循环是安全的。
        # 如果追求极致速度，可使用 torch_scatter 或 argsort 优化。
        for u in unique_src:
            
            # 找到节点 u 发出的所有边在 sub_edges 中的索引
            loc_idx = torch.where(src_nodes == u)[0]
            u_sims = sims[loc_idx]

            k_actual = min(len(loc_idx), k)
            vals, topk_rel_idx = torch.topk(u_sims, k_actual)
            
            if len(loc_idx) > 2:
                mean = u_sims.mean()
                std = u_sims.std()
                
                # 动态阈值: 必须大于均值
                score_mask = vals > (mean + z_score_threshold * std)
                
                # 至少保留 1 个最好的，防止孤立
                if score_mask.sum() == 0:
                    score_mask[0] = True
                
                topk_rel_idx = topk_rel_idx[score_mask]

            indices = global_indices[loc_idx[topk_rel_idx]]
            mask[indices] = True
            
            nodes_processed += 1
            edges_kept_local += len(indices)
        
        print(f"  - Relation {r_type.item()}: Processed {nodes_processed} nodes. Kept {edges_kept_local}/{len(global_indices)} edges.")


    # 5. 合并所有关系的边
    new_edge_index = edge_index[:, mask]
    new_edge_type = edge_type[mask]

    print(f"[Graph Refinement] Total Pruned: {edge_index.shape[1]} -> {new_edge_index.shape[1]}")
    print(f"[Graph Refinement] Reduction Rate: {1 - new_edge_index.shape[1]/edge_index.shape[1]:.2%}")

    # 5. 添加自环 (关键！防止 RGT 崩溃)
    new_edge_index, new_edge_type = remove_self_loops(new_edge_index, new_edge_type)
    new_edge_index, _ = add_self_loops(new_edge_index, num_nodes=num_nodes)
    
    # 补充 edge_type
    num_self_loops = num_nodes
    # 假设自环是类型 0 (或者你可以定义为 max_type + 1)
    loop_types = torch.zeros(num_self_loops, dtype=torch.long, device=device)
    new_edge_type = torch.cat([new_edge_type, loop_types], dim=0)
    
    data_dict['edge_index'] = new_edge_index
    data_dict['edge_type'] = new_edge_type
    
    return data_dict

def _torch_load(path):
    # 尝试使用 weights_only（若当前 PyTorch 支持），否则回退到普通加载
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


def seed_setting(seed_number):
    random.seed(seed_number)
    os.environ['PYTHONHASHSEED'] = str(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def setup_wandb(args, seed):
    run = wandb.init(
        project=args.project_name,
        name=args.experiment_name + f'_seed_{seed}',
        config=args 
    )
    return run


def load_raw_data(dataset_path, use_GNN=True):
    """
    🚀 **Enhanced TwiBot-20 Dataset Loader**
    
    Robust data loader for TwiBot-20 and similar datasets.
    Handles multiple path formats and ensures consistent file loading.
    
    Args:
        dataset_path (str): Path to dataset directory
            - Support formats: './datasets/TwiBot-20', 'TwiBot-20', './datasets/TwiBot20'
        use_GNN (bool): Whether to load graph structure data (edge_index, edge_type)
    
    Returns:
        dict: Dictionary containing loaded tensors:
            - 'train_idx': Training node indices [N_train] (TwiBot-20: [8278])
            - 'valid_idx': Validation node indices [N_valid] (TwiBot-20: [2365]) 
            - 'test_idx': Test node indices [N_test] (TwiBot-20: [1183])
            - 'labels': Node labels [N_nodes, N_classes] (TwiBot-20: [11826, 2] one-hot)
            - 'edge_index': Edge connectivity [2, N_edges] (TwiBot-20: [2, 16908])
            - 'edge_type': Edge types [N_edges] (TwiBot-20: [16908], values 0/1)
            - 'user_text': Text data from norm_user_text.json (if exists)
    
    File Name Mapping (Based on TwiBot-20 Structure Analysis):
        ✅ train_idx.pt    -> Training indices (consecutive: [0, 1, 2, ...])
        ✅ valid_idx.pt    -> Validation indices (consecutive: [8278, 8279, ...])  
        ✅ test_idx.pt     -> Test indices (consecutive: [10643, 10644, ...])
        ✅ labels.pt       -> One-hot labels [11826, 2] (Human: 5237, Bot: 6589)
        ✅ edge_index.pt   -> Graph edges [2, 16908] (COO format)
        ✅ edge_type.pt    -> Edge types [16908] (binary: 0 or 1)
        ✅ norm_user_text.json -> User text content
    
    Raises:
        FileNotFoundError: If dataset path cannot be resolved
        FileNotFoundError: If required data files are missing
    """
    # === 1. Path Resolution with Robust Error Handling ===
    data_dir = Path(dataset_path)
    print(f"[DataLoader] Resolving dataset path: '{dataset_path}'")

    # Try direct path first
    if not data_dir.exists():
        # Try ./datasets/<name> format
        cand = Path('./datasets') / data_dir.name
        if cand.exists():
            data_dir = cand
            print(f"[DataLoader] ✅ Found dataset at: {data_dir}")
        else:
            # Handle duplicated prefixes (e.g., './datasets/./datasets/TwiBot-20')
            s = str(data_dir).replace('./datasets/', '').replace('datasets/', '')
            cand2 = Path('./datasets') / s
            if cand2.exists():
                data_dir = cand2
                print(f"[DataLoader] ✅ Found dataset at: {data_dir} (cleaned path)")


    data_filepath = data_dir  # Use Path object for robust file operations

    print(f"[DataLoader] Loading data from: {data_filepath}")
    
    # === 2. Load Core Data Files ===
    print("[DataLoader] Loading data split indices...")
    
    # 🔹 Training/Validation/Test Splits (Index Format)
    # Based on analysis: TwiBot-20 uses consecutive indices
    train_idx = _torch_load(data_filepath / 'train_idx.pt')  # [8278] indices
    print(f"[DataLoader] ✅ train_idx: {train_idx.shape} (range: [{train_idx.min()}, {train_idx.max()}])")
    
    valid_idx = _torch_load(data_filepath / 'valid_idx.pt')  # [2365] indices  
    print(f"[DataLoader] ✅ valid_idx: {valid_idx.shape} (range: [{valid_idx.min()}, {valid_idx.max()}])")
    
    test_idx = _torch_load(data_filepath / 'test_idx.pt')    # [1183] indices
    print(f"[DataLoader] ✅ test_idx: {test_idx.shape} (range: [{test_idx.min()}, {test_idx.max()}])")
    
    print("[DataLoader] Loading node labels...")
    
    # 🔹 Labels (One-Hot Format Detection)
    # TwiBot-20: [11826, 2] one-hot -> Human: 5237, Bot: 6589
    labels = _torch_load(data_filepath / 'labels.pt')
    print(f"[DataLoader] ✅ labels: {labels.shape} dtype={labels.dtype}")


    # === 3. Load Graph Structure (if requested) ===
    if use_GNN:
        print("[DataLoader] Loading graph structure data...")
        
        # 🔹 Edge Index (Graph Connectivity)
        edge_index = _torch_load(data_filepath / 'edge_index.pt')
        
        # 🔹 Edge Type (Relation Types) - Optional

        edge_type = _torch_load(data_filepath / 'edge_type.pt')
    
        # 🔹 Text Data (User Metadata and Content) - Optional
        with open(data_filepath / 'norm_user_text.json', 'r', encoding='utf-8') as f:
            user_text = json.load(f)
            
        
        # Return complete dataset with graph structure
        data_dict = {
            'train_idx': train_idx,
            'valid_idx': valid_idx, 
            'test_idx': test_idx,
            'labels': labels,
            'edge_index': edge_index,
            'edge_type': edge_type,
            'user_text': user_text
        }

    else:
        print("[DataLoader] 📝 Graph structure not requested (use_GNN=False)")
        # Return basic dataset without graph structure  
        data_dict = {
            'train_idx': train_idx,
            'valid_idx': valid_idx,
            'test_idx': test_idx, 
            'labels': labels
        }
    
    # === 4. Final Validation ===
    total_nodes = len(train_idx) + len(valid_idx) + len(test_idx)
    expected_nodes = labels.shape[0]
    
    if total_nodes == expected_nodes:
        print(f"[DataLoader] ✅ Data consistency check passed: {total_nodes} total indices = {expected_nodes} labels")
    else:
        print(f"[DataLoader] ⚠️ Data consistency warning: {total_nodes} total indices ≠ {expected_nodes} labels")
    
    print(f"[DataLoader] 🎯 Dataset loaded successfully! Keys: {list(data_dict.keys())}")
    return data_dict


def load_distilled_knowledge(from_which_model, intermediate_data_filepath, iter):
    if from_which_model == 'LM':
        embeddings = torch.load(intermediate_data_filepath / f'embeddings_iter_{iter}.pt')
        soft_labels = torch.load(intermediate_data_filepath / f'soft_labels_iter_{iter}.pt')
        return embeddings, soft_labels
    
    elif from_which_model == 'GNN':
       
        soft_labels = torch.load(intermediate_data_filepath / f'soft_labels_iter_{iter}.pt')
        return soft_labels

    elif from_which_model == 'MLP':
        soft_labels = torch.load(intermediate_data_filepath / f'soft_labels_iter_{iter}.pt')
        return soft_labels
    
    else:
        raise ValueError('"from_which_model" should be "LM", "GNN" or "MLP".')


def prepare_path(experiment_name):
    experiment_path = Path(experiment_name)
    ckpt_filepath = experiment_path / 'checkpoints'
    MLP_ckpt_filepath = ckpt_filepath / 'MLP'
    LM_ckpt_filepath = ckpt_filepath / 'LM'
    GNN_ckpt_filepath = ckpt_filepath / 'GNN'
    MLP_KD_ckpt_filepath = Path('MLP_KD')
    LM_prt_ckpt_filepath = ckpt_filepath / 'LM_pretrain'
    GNN_prt_ckpt_filepath = ckpt_filepath / 'GNN_pretrain'
    LM_prt_ckpt_filepath.mkdir(exist_ok=True, parents=True)
    GNN_prt_ckpt_filepath.mkdir(exist_ok=True, parents=True)
    LM_ckpt_filepath.mkdir(exist_ok=True, parents=True)
    GNN_ckpt_filepath.mkdir(exist_ok=True, parents=True)
    MLP_KD_ckpt_filepath.mkdir(exist_ok=True, parents=True)
    MLP_ckpt_filepath.mkdir(exist_ok=True, parents=True)
    
    LM_intermediate_data_filepath = experiment_path / 'intermediate' / 'LM'
    GNN_intermediate_data_filepath = experiment_path / 'intermediate' / 'GNN'
    MLP_intermediate_data_filepath = experiment_path / 'intermediate' / 'MLP'
    LM_intermediate_data_filepath.mkdir(exist_ok=True, parents=True)
    GNN_intermediate_data_filepath.mkdir(exist_ok=True, parents=True)
    MLP_intermediate_data_filepath.mkdir(exist_ok=True, parents=True)

    return LM_prt_ckpt_filepath, GNN_prt_ckpt_filepath, MLP_KD_ckpt_filepath, LM_ckpt_filepath, GNN_ckpt_filepath, MLP_ckpt_filepath, LM_intermediate_data_filepath, GNN_intermediate_data_filepath, MLP_intermediate_data_filepath
    
def reset_split(n_nodes, ratio):
    idx = torch.randperm(n_nodes)
    split = list(map(int, ratio.split(',')))
    train_ratio = split[0] / sum(split)
    valid_ratio = split[1] / sum(split)

    train_idx = idx[: int(train_ratio * n_nodes)]
    valid_idx = idx[int(train_ratio * n_nodes): int((train_ratio + valid_ratio) * n_nodes)]
    test_idx = idx[int((train_ratio + valid_ratio) * n_nodes):]
    return train_idx, valid_idx, test_idx

def batch_linear_cka(X, Y):
    """计算 Batch 内的线性 CKA 相似度，用于诊断模态鸿沟"""
    
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    
    gram_x = torch.matmul(X, X.t())
    gram_y = torch.matmul(Y, Y.t())
    
    f_x = torch.norm(gram_x, p='fro')
    f_y = torch.norm(gram_y, p='fro')
    
    return torch.sum(gram_x * gram_y) / (f_x * f_y + 1e-8)

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/abs/2004.11362
    这会自动处理 Hard Sample Mining，因为温度系数会放大困难样本的梯度。
    """
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        features: [batch_size, dim] - 必须是归一化后的特征
        labels: [batch_size]
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            features = features.unsqueeze(1) # [batch, 1, dim]

        batch_size = features.shape[0]
        
        # 构造 Mask
        if labels is not None and mask is None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # 计算相似度矩阵
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # 数值稳定性处理
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # 计算 Log-Prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # 计算 Mean Log-Likelihood
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss