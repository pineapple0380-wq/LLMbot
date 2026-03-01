import torch
import torch.nn as nn
import torch.nn.functional as F

from model import EvidentialGraphHead
from utils import jsd_probs


class EvidentialMLPHead(nn.Module):
    def __init__(self, din, hidden=512, num_classes=2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(din, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.evidence_layer = nn.Linear(hidden, num_classes)

    def forward(self, x):
        h = self.backbone(x)
        evidence = F.softplus(self.evidence_layer(h))
        alpha = evidence + 1.0
        s = alpha.sum(dim=1, keepdim=True)
        u = alpha.size(1) / s
        p = alpha / s
        return {"alpha": alpha, "u": u, "p": p, "h": h}


class H2EALFusion(nn.Module):
    def __init__(self, qwen_dim=4096, roberta_dim=768, gnn_dim=512, hidden=512, num_classes=2, tau=0.5, gamma_text=2.0, gamma_fuse=2.0):
        super().__init__()
        self.tau = tau
        self.gamma_text = gamma_text
        self.gamma_fuse = gamma_fuse

        self.head_l8 = EvidentialMLPHead(qwen_dim, hidden, num_classes)
        self.head_l16 = EvidentialMLPHead(qwen_dim, hidden, num_classes)
        self.head_l32 = EvidentialMLPHead(qwen_dim, hidden, num_classes)
        self.head_roberta = EvidentialMLPHead(roberta_dim, hidden, num_classes)

        self.gnn_proj = nn.Sequential(nn.Linear(gnn_dim, hidden), nn.ReLU(), nn.Dropout(0.1))
        self.gnn_head = EvidentialGraphHead(hidden, num_classes=num_classes)

    def forward(self, q8, q16, q32, rob, gnn_emb, consistency):
        o8 = self.head_l8(q8)
        o16 = self.head_l16(q16)
        o32 = self.head_l32(q32)
        orob = self.head_roberta(rob)

        u_stack = torch.cat([o8["u"], o16["u"], o32["u"]], dim=1)
        w_depth = F.softmax(-u_stack.detach() / self.tau, dim=1)

        alpha_qwen = 1.0 + (
            w_depth[:, 0:1] * (o8["alpha"] - 1.0)
            + w_depth[:, 1:2] * (o16["alpha"] - 1.0)
            + w_depth[:, 2:3] * (o32["alpha"] - 1.0)
        )
        s_qwen = alpha_qwen.sum(dim=1, keepdim=True)
        u_qwen = alpha_qwen.size(1) / s_qwen
        p_qwen = alpha_qwen / s_qwen

        r_q = torch.exp(-self.gamma_text * u_qwen.detach())
        r_r = torch.exp(-self.gamma_text * orob["u"].detach())
        alpha_text = 1.0 + r_q * (alpha_qwen - 1.0) + r_r * (orob["alpha"] - 1.0)

        s_text = alpha_text.sum(dim=1, keepdim=True)
        u_text = alpha_text.size(1) / s_text
        p_text = alpha_text / s_text

        logits_g, alpha_g, u_g, p_g = self.gnn_head(self.gnn_proj(gnn_emb), consistency)

        jsd_tg = jsd_probs(p_text, p_g)
        r_text = torch.exp(-self.gamma_fuse * (1.0 - u_g.detach()) * jsd_tg.detach())
        r_graph = torch.exp(-self.gamma_fuse * (1.0 - u_text.detach()) * jsd_tg.detach())
        alpha_final = 1.0 + r_text * (alpha_text - 1.0) + r_graph * (alpha_g - 1.0)

        s_final = alpha_final.sum(dim=1, keepdim=True)
        u_final = alpha_final.size(1) / s_final
        p_final = alpha_final / s_final
        logits_final = torch.log(p_final + 1e-9)

        return {
            "alpha_l8": o8["alpha"], "u_l8": o8["u"], "p_l8": o8["p"],
            "alpha_l16": o16["alpha"], "u_l16": o16["u"], "p_l16": o16["p"],
            "alpha_l32": o32["alpha"], "u_l32": o32["u"], "p_l32": o32["p"],
            "alpha_qwen": alpha_qwen, "u_qwen": u_qwen, "p_qwen": p_qwen,
            "alpha_rob": orob["alpha"], "u_rob": orob["u"], "p_rob": orob["p"],
            "alpha_text": alpha_text, "u_text": u_text, "p_text": p_text,
            "logits_graph": logits_g, "alpha_graph": alpha_g, "u_graph": u_g, "p_graph": p_g,
            "alpha_final": alpha_final, "u_final": u_final, "p_final": p_final, "logits": logits_final,
            "w_depth": w_depth, "jsd_tg": jsd_tg, "r_text": r_text, "r_graph": r_graph,
        }
