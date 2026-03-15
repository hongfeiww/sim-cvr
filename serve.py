'''
FastAPI inference server for ESMM CVR / eCPM prediction.

Deployment detail: production-style serving with batching,
latency logging, and cost-ratio monitoring endpoint.

Usage:
    # Start server
    uvicorn serve:app --host 0.0.0.0 --port 8000

    # Single prediction
    curl -X POST http://localhost:8000/predict \\
      -H 'Content-Type: application/json' \\
      -d '{
        'user_id': 42, 'age_level': 3, 'gender': 1,
        'shopping_level': 2, 'city_level': 1,
        'item_id': 101, 'item_category': 5,
        'item_price_level': 2, 'item_sales_level': 3,
        'ad_id': 7, 'campaign_id': 2, 'customer_id': 1,
        'brand_id': 4, 'pid': 1, 'hour': 14,
        'seq_item_ids': [88, 45, 101, 0, 0],
        'seq_categories': [5, 3, 5, 0, 0],
        'bid': 50.0
      }'

    # Latency benchmark
    curl http://localhost:8000/benchmark
'''

import json
import logging
import os
import time
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from models.esmm import ESMM
from utils.calibration.cost_calibrator import CostCalibrator, CostMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
CHECKPOINT  = os.getenv('ESMM_CHECKPOINT', 'runs/esmm_exp1/best_model.pt')
DATA_DIR    = os.getenv('ESMM_DATA_DIR',   'data/processed')
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_SEQ_LEN = 50

# Load model at startup
app = FastAPI(title='ESMM CVR Prediction Service', version='1.0')

_model: Optional[ESMM] = None
_calibrator: Optional[CostCalibrator] = None
_monitor = CostMonitor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global _model, _calibrator, _monitor
    logger.info("Starting up: loading model...")
    if not os.path.exists(CHECKPOINT):
        logger.warning(f'Checkpoint not found: {CHECKPOINT}. /predict will return 503.')
        _model = None
    else:
        # Load config saved alongside checkpoint
        cfg_path = os.path.join(os.path.dirname(CHECKPOINT), 'config.json')
        with open(cfg_path) as f:
            cfg = json.load(f)

        vocab_path = os.path.join(DATA_DIR, 'vocab_sizes.json')
        with open(vocab_path) as f:
            vocab_sizes = json.load(f)

        _model = ESMM(
            vocab_sizes    = vocab_sizes,
            embed_dim      = cfg['embed_dim'],
            hidden_dims    = cfg['hidden_dims'],
            dropout        = 0.0, # no dropout at inference
            top_k          = cfg['top_k'],
            gsu_mode       = cfg['gsu_mode'],
            use_ctr_tower  = cfg.get('use_ctr_tower', True),
        ).to(DEVICE)

        ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
        _model.load_state_dict(ckpt['model_state'])
        _model.eval()
        logger.info(f"Model loaded from {CHECKPOINT} (epoch={ckpt['epoch']})")

        # Load calibrator if available
        cal_path = os.path.join(os.path.dirname(CHECKPOINT), 'calibrator.pkl')
        if os.path.exists(cal_path):
            _calibrator = CostCalibrator.load(cal_path)
            logger.info('Isotonic calibrator loaded')
        else:
            _calibrator = None

    _monitor = CostMonitor()
    logger.info("Startup complete.")
    yield
    # Shutdown 
    logger.info("Shutting down: releasing resources...")
    # Perform any cleanup here if needed
    _model = None
    _calibrator = None
    _monitor = None
    logger.info("Shutdown complete.")

# Create FastAPI app with lifespan
app = FastAPI(title='ESMM CVR Prediction Service', version='1.0', lifespan=lifespan)

# Request / Response schemas
class PredictRequest(BaseModel):
    # Scalar features
    user_id:          int
    age_level:        int
    gender:           int
    shopping_level:   int
    city_level:       int
    item_id:          int
    item_category:    int
    item_price_level: int
    item_sales_level: int
    ad_id:            int
    campaign_id:      int
    customer_id:      int
    brand_id:         int
    pid:              int
    hour:             int
    # Behavior sequence
    seq_item_ids:   List[int]
    seq_categories: List[int]
    # Bid for eCPM calculation (li, CPA target)
    bid: float = 50000.0


class PredictResponse(BaseModel):
    p_ctr:           float   # P(click | impression)
    p_cvr:           float   # P(purchase | click)
    p_cvr_calibrated:Optional[float]  # isotonic-calibrated pCVR
    p_ctcvr:         float   # P(purchase | impression) = pCTR x pCVR
    ecpm:            float   # bid x pCTCVR x 1000
    latency_ms:      float   # inference latency in milliseconds


class BatchPredictRequest(BaseModel):
    requests: List[PredictRequest]


class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]
    batch_size:  int
    total_latency_ms: float
    avg_latency_ms:   float


# Helpers

def _pad_sequence(seq: List[int], max_len: int) -> List[int]:
    # Right-align: most recent item at the end, pad left with 0.
    seq = seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq


def _request_to_batch(req: PredictRequest) -> dict:
    # Convert a single PredictRequest to a model-ready batch dict.
    scalar_keys = [
        'user_id','age_level','gender','shopping_level','city_level',
        'item_id','item_category','item_price_level','item_sales_level',
        'ad_id','campaign_id','customer_id','brand_id','pid','hour',
    ]
    batch = {k: torch.tensor([getattr(req, k)], dtype=torch.long).to(DEVICE)
             for k in scalar_keys}

    seq_items = _pad_sequence(req.seq_item_ids,   MAX_SEQ_LEN)
    seq_cats  = _pad_sequence(req.seq_categories, MAX_SEQ_LEN)

    batch['seq_item_ids']   = torch.tensor([seq_items], dtype=torch.long).to(DEVICE)
    batch['seq_categories'] = torch.tensor([seq_cats],  dtype=torch.long).to(DEVICE)
    batch['seq_mask']       = (batch['seq_item_ids'] != 0)
    return batch


def _requests_to_batch(reqs: List[PredictRequest]) -> dict:
    # Convert a list of requests to a batched dict.
    scalar_keys = [
        'user_id','age_level','gender','shopping_level','city_level',
        'item_id','item_category','item_price_level','item_sales_level',
        'ad_id','campaign_id','customer_id','brand_id','pid','hour',
    ]
    batch = {k: torch.tensor([getattr(r, k) for r in reqs],
                              dtype=torch.long).to(DEVICE)
             for k in scalar_keys}

    seq_items = [_pad_sequence(r.seq_item_ids,   MAX_SEQ_LEN) for r in reqs]
    seq_cats  = [_pad_sequence(r.seq_categories, MAX_SEQ_LEN) for r in reqs]

    batch['seq_item_ids']   = torch.tensor(seq_items, dtype=torch.long).to(DEVICE)
    batch['seq_categories'] = torch.tensor(seq_cats,  dtype=torch.long).to(DEVICE)
    batch['seq_mask']       = (batch['seq_item_ids'] != 0)
    return batch


# Routes

@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': _model is not None, 'device': DEVICE}


@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')

    batch = _request_to_batch(req)
    bid   = torch.tensor([req.bid], dtype=torch.float).to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        p_ctr, p_cvr, p_ctcvr = _model(batch)
    latency_ms = (time.perf_counter() - t0) * 1000

    p_ctr_val   = float(p_ctr[0])
    p_cvr_val   = float(p_cvr[0])
    p_ctcvr_val = float(p_ctcvr[0])
    ecpm        = float(req.bid * p_ctcvr_val * 1000)

    # Apply isotonic calibration if available
    p_cvr_cal = None
    if _calibrator is not None:
        p_cvr_cal   = float(_calibrator.transform(np.array([p_cvr_val]))[0])
        p_ctcvr_val = p_ctr_val * p_cvr_cal   # recompute with calibrated pCVR
        ecpm        = req.bid * p_ctcvr_val * 1000

    logger.info(
        f'predict | p_ctr={p_ctr_val:.4f} p_cvr={p_cvr_val:.4f} '
        f'ecpm={ecpm:.2f} latency={latency_ms:.2f}ms'
    )

    return PredictResponse(
        p_ctr            = p_ctr_val,
        p_cvr            = p_cvr_val,
        p_cvr_calibrated = p_cvr_cal,
        p_ctcvr          = p_ctcvr_val,
        ecpm             = ecpm,
        latency_ms       = latency_ms,
    )


@app.post('/predict/batch', response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    '''
    Batch inference endpoint.
    In production, the ad server calls this once per auction with all
    candidate ads — typically batch_size = 50-200.
    '''
    if _model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    if not req.requests:
        raise HTTPException(status_code=400, detail='Empty request list')

    batch = _requests_to_batch(req.requests)
    bids  = torch.tensor([r.bid for r in req.requests],
                          dtype=torch.float).to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        p_ctr, p_cvr, p_ctcvr = _model(batch)
    total_ms = (time.perf_counter() - t0) * 1000

    p_ctr_np   = p_ctr.cpu().numpy()
    p_cvr_np   = p_cvr.cpu().numpy()
    p_ctcvr_np = p_ctcvr.cpu().numpy()

    if _calibrator is not None:
        p_cvr_cal_np   = _calibrator.transform(p_cvr_np)
        p_ctcvr_np     = p_ctr_np * p_cvr_cal_np
    else:
        p_cvr_cal_np   = np.full_like(p_cvr_np, fill_value=float('nan'))

    predictions = []
    for i, r in enumerate(req.requests):
        ecpm = float(r.bid * p_ctcvr_np[i] * 1000)
        predictions.append(PredictResponse(
            p_ctr            = float(p_ctr_np[i]),
            p_cvr            = float(p_cvr_np[i]),
            p_cvr_calibrated = float(p_cvr_cal_np[i]) if _calibrator else None,
            p_ctcvr          = float(p_ctcvr_np[i]),
            ecpm             = ecpm,
            latency_ms       = total_ms / len(req.requests),
        ))

    return BatchPredictResponse(
        predictions      = predictions,
        batch_size       = len(req.requests),
        total_latency_ms = total_ms,
        avg_latency_ms   = total_ms / len(req.requests),
    )


@app.get('/benchmark')
def benchmark(n_warmup: int = 10, n_iters: int = 100, batch_size: int = 32):
    if _model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    
    vocab_sizes = {}
    for name, emb in _model.shared_emb.embeddings.items():
        vocab_sizes[name] = emb.num_embeddings

    # Build a random batch
    def make_rand_batch(bs):
        scalar_keys = [
            'user_id','age_level','gender','shopping_level','city_level',
            'item_id','item_category','item_price_level','item_sales_level',
            'ad_id','campaign_id','customer_id','brand_id','pid','hour',
        ]
        b = {}
        for k in scalar_keys:
            max_id = vocab_sizes.get(k, 100)
            b[k] = torch.randint(1, max_id, (bs,)).to(DEVICE)
        max_item_id = vocab_sizes.get('item_id', 100)
        max_cat_id  = vocab_sizes.get('item_category', 20)
        b['seq_item_ids']   = torch.randint(0, max_item_id, (bs, MAX_SEQ_LEN)).to(DEVICE)  # 0 allowed for padding
        b['seq_categories'] = torch.randint(0, max_cat_id,  (bs, MAX_SEQ_LEN)).to(DEVICE)
        b['seq_mask']       = (b['seq_item_ids'] != 0)
        return b

    rand_batch = make_rand_batch(batch_size)

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _model(rand_batch)

    if DEVICE == 'cuda':
        torch.cuda.synchronize()

    # Measure
    latencies = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        with torch.no_grad():
            _model(rand_batch)
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = sorted(latencies)
    result = {
        'device':      DEVICE,
        'batch_size':  batch_size,
        'n_iters':     n_iters,
        'latency_ms': {
            'p50': round(latencies[int(n_iters * 0.50)], 3),
            'p95': round(latencies[int(n_iters * 0.95)], 3),
            'p99': round(latencies[int(n_iters * 0.99)], 3),
            'mean':round(float(np.mean(latencies)), 3),
        },
        'throughput_qps': round(batch_size * 1000 / np.mean(latencies), 1),
    }
    logger.info(f'Benchmark result: {result}')
    return result


@app.get('/cost_monitor')
def cost_monitor_status():
    # Return latest cost ratio monitoring status.
    if not _monitor.history:
        return {'status': 'no_data', 'history_length': 0}
    latest = _monitor.history[-1]
    return {
        'latest':         latest,
        'history_length': len(_monitor.history),
    }
