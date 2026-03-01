import argparse
from pathlib import Path

import torch
import wandb
from torch_geometric.data import Data

from GNNs import build_gnn
from fusion_h2eal import H2EALFusion
from train import QwenPrecomputedTrainer
from utils import load_raw_data, relation_aware_knn_pruning, seed_setting


def parse_args():
    parser = argparse.ArgumentParser(description='H2-EAL training')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--embeddings_path', type=str, required=True, help='Qwen L32 embeddings')
    parser.add_argument('--qwen_l8_path', type=str, required=True)
    parser.add_argument('--qwen_l16_path', type=str, required=True)
    parser.add_argument('--roberta_path', type=str, required=True)

    parser.add_argument('--gnn_type', type=str, default='RGT', choices=['RGCN', 'RGT', 'SimpleHGN', 'HGT'])
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--eval_patience', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb_project', type=str, default='H2EAL')
    parser.add_argument('--exp_name', type=str, default='h2eal')
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--gamma_text', type=float, default=2.0)
    parser.add_argument('--gamma_fuse', type=float, default=2.0)
    parser.add_argument('--lambda_conflict', type=float, default=0.1)
    parser.add_argument('--pruning', action='store_true', default=False)
    parser.add_argument('--neighbor', type=int, default=16)
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
    return parser.parse_args()


def _load_embedding(path):
    emb = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(emb, torch.Tensor):
        emb = torch.tensor(emb)
    return emb.float()


def main():
    args = parse_args()
    seed_setting(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    q32 = _load_embedding(args.embeddings_path)
    q8 = _load_embedding(args.qwen_l8_path)
    q16 = _load_embedding(args.qwen_l16_path)
    rob = _load_embedding(args.roberta_path)

    data_dict = load_raw_data(args.dataset_path, use_GNN=True)
    if args.pruning:
        data_dict = relation_aware_knn_pruning(data_dict, q32, k=args.neighbor)

    labels = data_dict['labels']
    if labels.dim() > 1:
        labels = labels.argmax(dim=1)

    data = Data(
        x=q32,
        q8=q8,
        q16=q16,
        rob=rob,
        y=labels.long(),
        edge_index=data_dict['edge_index'],
        edge_type=data_dict.get('edge_type', None),
    )

    num_relations = int(data.edge_type.max().item()) + 1 if data.edge_type is not None else 1
    gnn = build_gnn(args.gnn_type, {
        'lm_input_dim': q32.size(1),
        'gnn_hidden_dim': args.hidden_dim,
        'n_relations': num_relations,
        'gnn_n_layers': args.n_layers,
        'dropout': args.dropout,
        'heads': args.heads,
        'num_features_dim': 0,
    }).to(device)

    fusion = H2EALFusion(
        qwen_dim=q32.size(1),
        roberta_dim=rob.size(1),
        gnn_dim=args.hidden_dim,
        hidden=args.hidden_dim,
        tau=args.tau,
        gamma_text=args.gamma_text,
        gamma_fuse=args.gamma_fuse,
    ).to(device)

    wandb.init(project=args.wandb_project, name=f"{args.exp_name}_seed{args.seed}", config=vars(args))

    ckpt_dir = Path(f'./saved_models/{args.ckpt_dir}')
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainer = QwenPrecomputedTrainer(
        data=data,
        data_dict=data_dict,
        gnn_model=gnn,
        fusion_model=fusion,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        eval_patience=args.eval_patience,
        lambda_conflict=args.lambda_conflict,
        ckpt_filepath=str(ckpt_dir / f'best_model_seed{args.seed}.pt'),
    )
    trainer.train()


if __name__ == '__main__':
    main()
