'''
train.py — ESMM CVR main entry point.

Quick start (synthetic data, no Tianchi account needed):
  python data/generate_synthetic.py --n_samples 200000
  python train.py --data_dir data/processed --use_amp --log_dir runs/esmm_exp1

Full Ali-CCP:
  bash data/download_aliccp.sh
  python data/preprocess_aliccp.py --data_dir data/raw/aliccp --subsample 0.1
  python train.py --data_dir data/processed --batch_size 8192 --epochs 10

no CTR tower:
  python train.py --no_ctr_tower --log_dir runs/ablation_no_ctr
'''

import argparse
import json
import logging
import os

import torch

from data.dataset import build_dataloaders
from models.esmm import ESMM
from utils.seed import set_seed
from utils.trainer import ESMMTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s — %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description='ESMM CVR Training')
    # Data
    p.add_argument('--data_dir',    default='data/processed')
    p.add_argument('--seq_len',     type=int,   default=50)
    p.add_argument('--batch_size',  type=int,   default=2048)
    p.add_argument('--num_workers', type=int,   default=4)
    p.add_argument('--subset',      type=float, default=None)
    p.add_argument("--streaming",   action="store_true",
                   help="Stream parquet row-by-row (O(batch) RAM, use for large datasets)")
    # Model
    p.add_argument('--embed_dim',   type=int,         default=32)
    p.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128, 64])
    p.add_argument('--dropout',     type=float,       default=0.3)
    p.add_argument('--top_k',       type=int,         default=20)
    p.add_argument('--gsu_mode',    default='hard',   choices=['hard', 'soft'])
    p.add_argument('--no_ctr_tower',action='store_true',
                   help='Ablation: remove CTR tower')
    # Loss
    p.add_argument('--ctr_loss_weight', type=float, default=1.0)
    p.add_argument('--target_cpa',      type=float, default=50.0)
    # Training
    p.add_argument('--epochs',       type=int,   default=30)
    p.add_argument('--lr',           type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-6)
    p.add_argument('--patience',     type=int,   default=3)
    p.add_argument('--use_amp',      action='store_true')
    p.add_argument('--no_cuda',      action='store_true')
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--log_dir',      default='runs/esmm_exp1')
    return p.parse_args()


def main():
    args = parse_args()
    use_ctr_tower = not args.no_ctr_tower

    set_seed(args.seed)
    device = 'cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda'
    logger.info(f'Device: {device}' +
                (f' ({torch.cuda.get_device_name(0)})' if device == 'cuda' else ''))

    # data
    train_loader, val_loader, test_loader, vocab_sizes = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_seq_len=args.seq_len,
        num_workers=args.num_workers,
        subset=args.subset,
        streaming=getattr(args, "streaming", False),
    )
    def _loader_len(loader):
        try:
            return len(loader)
        except TypeError:
            return "?"
    logger.info(f"Train batches: {_loader_len(train_loader)} | "
                f"Val: {_loader_len(val_loader)} | Test: {_loader_len(test_loader)}")

    # model
    model = ESMM(
        vocab_sizes=vocab_sizes['vocab_sizes'],
        embed_dim=args.embed_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        top_k=args.top_k,
        gsu_mode=args.gsu_mode,
        use_ctr_tower=use_ctr_tower,
        ctr_weight=args.ctr_loss_weight,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'ESMM | params={n_params:,} | ctr_tower={use_ctr_tower} | '
                f'top_k={args.top_k} | gsu={args.gsu_mode}')

    os.makedirs(args.log_dir, exist_ok=True)
    cfg = vars(args); cfg['use_ctr_tower'] = use_ctr_tower
    with open(os.path.join(args.log_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)

    # train
    trainer = ESMMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        use_amp=args.use_amp,
        log_dir=args.log_dir,
        device=device,
        ctr_loss_weight=args.ctr_loss_weight,
        target_cpa=args.target_cpa,
    )
    best_ckpt = trainer.train(epochs=args.epochs)
    logger.info(f'Best checkpoint: {best_ckpt}')

    # eval
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    test_metrics = trainer._evaluate(test_loader)

    logger.info('=' * 60 + '\nTEST RESULTS')
    for k, v in test_metrics.items():
        if isinstance(v, float):
            logger.info(f'  {k:22s}: {v:.4f}')
        else:
            logger.info(f'  {k:22s}: {v}')
    logger.info('=' * 60)

    with open(os.path.join(args.log_dir, 'test_results.json'), 'w') as f:
        json.dump({k: (float(v) if isinstance(v, float) else v)
                   for k, v in test_metrics.items()}, f, indent=2)


if __name__ == '__main__':
    main()