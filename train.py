from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.data import Data
from torch_geometric.utils import scatter
from torch.utils.data import DataLoader, TensorDataset

from utils import EdlLoss


@dataclass
class MiniBatch:
    n_id: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    batch_size: int
    x: torch.Tensor
    q8: torch.Tensor
    q16: torch.Tensor
    rob: torch.Tensor

    def to(self, device):
        self.n_id = self.n_id.to(device)
        self.edge_index = self.edge_index.to(device)
        self.x = self.x.to(device)
        self.q8 = self.q8.to(device)
        self.q16 = self.q16.to(device)
        self.rob = self.rob.to(device)
        if self.edge_type is not None:
            self.edge_type = self.edge_type.to(device)
        return self


class NeighborSampler(DataLoader):
    def __init__(self, data: Data, sizes, batch_size, input_nodes, shuffle=True):
        self.data = data
        self.edge_index = data.edge_index
        self.edge_type = data.edge_type
        self.sizes = sizes
        self.num_nodes = int(self.edge_index.max().item()) + 1

        self.adj = [[] for _ in range(self.num_nodes)]
        self.adj_t = [[] for _ in range(self.num_nodes)]
        rows = self.edge_index[0].tolist()
        cols = self.edge_index[1].tolist()
        types = self.edge_type.tolist() if self.edge_type is not None else None
        if types is not None:
            for r, c, t in zip(rows, cols, types):
                self.adj[c].append(r)
                self.adj_t[c].append(t)
        else:
            for r, c in zip(rows, cols):
                self.adj[c].append(r)

        super().__init__(
            TensorDataset(input_nodes.cpu()),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.sample_subgraph,
        )

    def sample_subgraph(self, batch_indices):
        seed_nodes = [item[0].item() for item in batch_indices]
        batch_size = len(seed_nodes)
        n_id = list(seed_nodes)
        n_id_set = set(seed_nodes)
        node_map = {n: i for i, n in enumerate(n_id)}
        edges_src, edges_dst, edges_type = [], [], []

        frontier = seed_nodes
        for size in self.sizes:
            nxt = []
            for target_node in frontier:
                neighbors = self.adj[target_node]
                if not neighbors:
                    continue
                idxs = np.random.choice(len(neighbors), size, replace=False) if len(neighbors) > size else range(len(neighbors))
                for idx in idxs:
                    source_node = neighbors[idx]
                    if source_node not in n_id_set:
                        n_id_set.add(source_node)
                        node_map[source_node] = len(n_id)
                        n_id.append(source_node)
                        nxt.append(source_node)
                    edges_src.append(node_map[source_node])
                    edges_dst.append(node_map[target_node])
                    if self.edge_type is not None:
                        edges_type.append(self.adj_t[target_node][idx])
            frontier = nxt

        n_id = torch.tensor(n_id, dtype=torch.long)
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long) if edges_src else torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.tensor(edges_type, dtype=torch.long) if self.edge_type is not None and edges_type else None

        x = self.data.x[n_id]
        q8 = self.data.q8[n_id]
        q16 = self.data.q16[n_id]
        rob = self.data.rob[n_id]
        return MiniBatch(n_id=n_id, edge_index=edge_index, edge_type=edge_type, batch_size=batch_size, x=x, q8=q8, q16=q16, rob=rob)


class QwenPrecomputedTrainer:
    def __init__(self, data, data_dict, gnn_model, fusion_model, device, epochs, lr, weight_decay, batch_size, eval_patience=10, lambda_conflict=0.1, ckpt_filepath='best_model.pt'):
        self.data = data
        self.data_dict = data_dict
        self.gnn = gnn_model.to(device)
        self.fusion = fusion_model.to(device)
        self.device = device
        self.epochs = epochs
        self.eval_patience = eval_patience
        self.lambda_conflict = lambda_conflict
        self.ckpt_filepath = Path(ckpt_filepath)

        self.labels = data.y.long().to(device)
        self.edl_loss = EdlLoss()
        self.optimizer = torch.optim.AdamW(list(self.gnn.parameters()) + list(self.fusion.parameters()), lr=lr, weight_decay=weight_decay)

        self.train_loader = NeighborSampler(data, sizes=[10, 10], batch_size=batch_size, input_nodes=data_dict['train_idx'], shuffle=True)
        self.val_loader = NeighborSampler(data, sizes=[10, 10], batch_size=batch_size, input_nodes=data_dict['valid_idx'], shuffle=False)
        self.test_loader = NeighborSampler(data, sizes=[10, 10], batch_size=batch_size, input_nodes=data_dict['test_idx'], shuffle=False)

    def _call_gnn(self, x, edge_index, edge_type):
        return self.gnn(x, edge_index, edge_type) if edge_type is not None else self.gnn(x, edge_index)

    def _compute_batch_consistency(self, x, edge_index, batch_size):
        if edge_index.numel() == 0:
            return torch.ones((batch_size, 1), device=x.device)
        row, col = edge_index
        x_norm = F.normalize(x, p=2, dim=-1)
        edge_sim = (x_norm[row] * x_norm[col]).sum(dim=-1)
        consistency_all = scatter(edge_sim, col, dim=0, dim_size=x.size(0), reduce='mean')
        return consistency_all[:batch_size].unsqueeze(1)

    def _train_epoch_joint(self, epoch):
        self.gnn.train()
        self.fusion.train()
        total = 0.0

        for step, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)
            if step == 0:
                assert hasattr(batch, 'q8') and batch.q8.shape[0] == batch.x.shape[0]
                assert batch.rob.shape[1] == 768

            self.optimizer.zero_grad()
            h_sub = self._call_gnn(batch.x, batch.edge_index, batch.edge_type)
            bsz = batch.batch_size
            gnn_emb = h_sub[:bsz]
            consistency = self._compute_batch_consistency(batch.x, batch.edge_index, bsz)

            out = self.fusion(batch.q8[:bsz], batch.q16[:bsz], batch.x[:bsz], batch.rob[:bsz], gnn_emb, consistency)
            y = self.labels[batch.n_id[:bsz]]

            loss_edl = self.edl_loss(out['alpha_final'], y, epoch)
            jsd = out['jsd_tg']
            conflict_penalty = ((1.0 - out['u_text'].detach()) * (1.0 - out['u_graph'].detach()) * jsd).mean()
            loss = loss_edl + self.lambda_conflict * conflict_penalty
            loss.backward()
            self.optimizer.step()
            total += loss.item()

        return total / max(1, len(self.train_loader))

    @torch.no_grad()
    def evaluate(self, split='val'):
        self.gnn.eval()
        self.fusion.eval()
        loader = self.val_loader if split == 'val' else self.test_loader
        preds, targets = [], []
        w_depth_mean, u_text_m, u_graph_m, u_final_m = [], [], [], []

        for batch in loader:
            batch = batch.to(self.device)
            h_sub = self._call_gnn(batch.x, batch.edge_index, batch.edge_type)
            bsz = batch.batch_size
            consistency = self._compute_batch_consistency(batch.x, batch.edge_index, bsz)
            out = self.fusion(batch.q8[:bsz], batch.q16[:bsz], batch.x[:bsz], batch.rob[:bsz], h_sub[:bsz], consistency)
            y = self.labels[batch.n_id[:bsz]]

            preds.append(out['logits'].argmax(dim=-1).cpu())
            targets.append(y.cpu())
            w_depth_mean.append(out['w_depth'].mean(dim=0).cpu())
            u_text_m.append(out['u_text'].mean().item())
            u_graph_m.append(out['u_graph'].mean().item())
            u_final_m.append(out['u_final'].mean().item())

        p = torch.cat(preds)
        t = torch.cat(targets)
        metrics = {
            'f1': f1_score(t, p, average='macro'),
            'acc': accuracy_score(t, p),
            'w_depth': torch.stack(w_depth_mean).mean(dim=0).tolist() if w_depth_mean else [0, 0, 0],
            'u_text': float(np.mean(u_text_m)) if u_text_m else 0.0,
            'u_graph': float(np.mean(u_graph_m)) if u_graph_m else 0.0,
            'u_final': float(np.mean(u_final_m)) if u_final_m else 0.0,
        }
        return metrics

    def train(self):
        best_f1 = -1.0
        bad_epochs = 0
        for epoch in range(1, self.epochs + 1):
            loss = self._train_epoch_joint(epoch)
            val = self.evaluate('val')
            log_obj = {
                'epoch': epoch,
                'train/loss': loss,
                'val/f1': val['f1'],
                'val/acc': val['acc'],
                'val/w_depth_l8': val['w_depth'][0],
                'val/w_depth_l16': val['w_depth'][1],
                'val/w_depth_l32': val['w_depth'][2],
                'val/u_text': val['u_text'],
                'val/u_graph': val['u_graph'],
                'val/u_final': val['u_final'],
            }
            if wandb.run is not None:
                wandb.log(log_obj)

            if val['f1'] > best_f1:
                best_f1 = val['f1']
                bad_epochs = 0
                torch.save({'gnn_state_dict': self.gnn.state_dict(), 'fusion_state_dict': self.fusion.state_dict(), 'best_val_f1': best_f1}, self.ckpt_filepath)
            else:
                bad_epochs += 1
                if bad_epochs >= self.eval_patience:
                    break

        return best_f1
