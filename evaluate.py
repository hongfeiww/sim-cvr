'''
  python evaluate.py --checkpoint runs/esmm_exp1/best_model.pt --split test
  python evaluate.py --checkpoint runs/esmm_exp1/best_model.pt --split test --calibrate
'''

import argparse
import json
import logging
import os

import numpy as np
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast

from data.dataset import build_dataloaders
from models.esmm import ESMM
from utils.metrics import compute_esmm_metrics
from utils.calibration.cost_calibrator import CostCalibrator, CostMonitor
from utils.seed import set_seed

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def run_inference(model, loader, device):
    model.eval()
    out = {k: [] for k in ['p_ctr','p_cvr','p_ctcvr','click','purchase']}
    with torch.no_grad():
        for batch in tqdm(loader, desc='  infer', leave=False):
            batch    = {k: v.to(device) for k, v in batch.items()}
            click    = batch.pop('click').cpu().numpy()
            purchase = batch.pop('purchase').cpu().numpy()
            p_ctr, p_cvr, p_ctcvr = model(batch)
            out['p_ctr'].append(p_ctr.cpu().numpy())
            out['p_cvr'].append(p_cvr.cpu().numpy())
            out['p_ctcvr'].append(p_ctcvr.cpu().numpy())
            out['click'].append(click)
            out['purchase'].append(purchase)
    return {k: np.concatenate(v) for k, v in out.items()}


def evaluate(args):
    set_seed(38)
    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))

    with open(os.path.join(ckpt_dir, 'config.json')) as f:
        cfg = json.load(f)

    device = 'cpu' if not torch.cuda.is_available() else 'cuda'

    _, val_loader, test_loader, vocab_sizes = build_dataloaders(
        data_dir=cfg['data_dir'],
        batch_size=512,
        max_seq_len=cfg['seq_len'],
        num_workers=2,
    )

    model = ESMM(
        vocab_sizes      = vocab_sizes,
        embed_dim        = cfg['embed_dim'],
        hidden_dims      = cfg['hidden_dims'],
        dropout          = 0.0,
        top_k            = cfg['top_k'],
        gsu_mode         = cfg['gsu_mode'],
        use_ctr_tower    = cfg.get('use_ctr_tower', True),
        ctr_weight       = cfg.get('ctr_loss_weight', 1.0),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    logger.info(f'Loaded epoch={ckpt['epoch']} | '
                f'val_AUC_CVR={ckpt['val_metrics'].get('auc_cvr', 0.0):.4f}')

    target_loader = test_loader if args.split == 'test' else val_loader
    preds = run_inference(model, target_loader, device)

    # Optional calibration
    if args.calibrate:
        logger.info('Fitting isotonic calibration on val set...')
        val_preds  = run_inference(model, val_loader, device)
        calibrator = CostCalibrator()
        calibrator.fit(raw_scores=val_preds['p_cvr'], labels=val_preds['purchase'])
        preds['p_cvr']   = calibrator.transform(preds['p_cvr'])
        preds['p_ctcvr'] = preds['p_ctr'] * preds['p_cvr']   # recompute
        calibrator.save(os.path.join(ckpt_dir, 'calibrator.pkl'))
        logger.info(f'Calibration ECE: {calibrator.fit_stats['raw_ece']:.5f} → '
                    f'{calibrator.fit_stats['calibrated_ece']:.5f}')

    bid     = np.full(len(preds['click']), cfg.get('target_cpa', 50.0), dtype=np.float32)
    metrics = compute_esmm_metrics(
        p_ctr    = preds['p_ctr'],
        p_cvr    = preds['p_cvr'],
        p_ctcvr  = preds['p_ctcvr'],
        click    = preds['click'],
        purchase = preds['purchase'],
        bid      = bid,
    )

    # Cost monitor report
    monitor = CostMonitor()
    monitor.compute(p_cvr=preds['p_cvr'], purchase=preds['purchase'], bid=bid)

    suffix = '_calibrated' if args.calibrate else ''
    logger.info(f'\n{'='*55}')
    logger.info(f'  {args.split.upper()}' +
                (' + isotonic calibration' if args.calibrate else ''))
    logger.info(f'{'='*55}')
    for k, v in metrics.items():
        logger.info(f'  {k:22s}: {float(v):.4f}' if isinstance(v, float)
                    else f'  {k:22s}: {v}')
    logger.info(f'{'='*55}\n')

    out = os.path.join(ckpt_dir, f'{args.split}_metrics{suffix}.json')
    with open(out, 'w') as f:
        json.dump({k: (float(v) if isinstance(v, float) else v)
                   for k, v in metrics.items()}, f, indent=2)
    logger.info(f'Saved → {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split',      default='test', choices=['val','test'])
    p.add_argument('--calibrate',  action='store_true')
    args = p.parse_args()
    evaluate(args)
