"""Model components used by H2-EAL training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialGraphHead(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, input_dim)
        self.evidence_layer = nn.Linear(input_dim, num_classes)
        self.evidence_scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, x, consistency=None):
        h = F.relu(self.proj(x))
        raw_evidence = F.softplus(self.evidence_layer(h)) * F.softplus(self.evidence_scale)
        evidence = raw_evidence * consistency if consistency is not None else raw_evidence

        alpha = evidence + 1.0
        s = alpha.sum(dim=1, keepdim=True)
        u = alpha.size(1) / s
        p = alpha / s
        logits = torch.log(p + 1e-9)
        return logits, alpha, u, p
