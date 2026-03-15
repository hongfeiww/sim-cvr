# SIM-CVR: Sequential User Interest Modeling for Conversion Prediction

This repository contains a simplified research-style implementation of conversion rate (CVR) prediction models inspired by industrial recommender systems.

The project reimplements the ESMM (Entire Space Multi-Task Model) framework together with a lightweight SIM-style sequential interest module for modeling user behavior sequences.

The goal is to study how sequence modeling and multi-task learning improve conversion prediction under sample-selection bias.



## Model Overview

```
CTR:  P(click | impression)
CVR:  P(purchase | click)
CTCVR: P(click ∧ purchase | impression)
```
Following the ESMM formulation:

```
P(CTCVR) = P(CTR) × P(CVR)
```

## Architecture

The model consists of three main components.

### 1. Shared Embedding Layer

Categorical features are mapped to dense embeddings:
```
USER_FIELDS = ["user_id", "age_level", "gender", "shopping_level", "city_level"]
ITEM_FIELDS = ["item_id", "item_category", "item_price_level", "item_sales_level"]
AD_FIELDS   = ["ad_id", "campaign_id", "customer_id", "brand_id"]
CTX_FIELDS  = ["pid", "hour"]
```
### 2. CTR Tower

A standard MLP predicting click probability.
```
embedding → MLP → sigmoid → pCTR
```

### 3. CVR Tower (SIM-style)

The CVR tower models user behavioral history.
```
Components:

GSU (General Search Unit)
   ↓
Top-K item retrieval from behavior history

ESU (Exact Search Unit)
   ↓
Attention between target item and retrieved items

MLP decoder
   ↓
pCVR
```
## Dataset

Dataset used in this project:
**Ali-CCP (Alibaba Click Conversion Prediction dataset)** https://tianchi.aliyun.com/dataset/408

Statistics after preprocessing:
```
~8M samples (subset for experimentation)
16 categorical features
two labels: click, purchase
```
Split strategy:
```
train: 70%
validation: 10%
test: 20%
```
Data cleaning included filtering users/items with extremely sparse interactions.

## Training

Example training command:

```
python train.py \
 --data_dir data/processed \
 --embed_dim 18 \
 --hidden_dims 256 128 64 \
 --top_k 20 \
 --gsu_mode hard \
 --lr 1e-3 \
 --batch_size 4096 \
 --epochs 10 \
 --patience 3 \
 --streaming
```

Environment:
```
Python 3.10
PyTorch 2.0
```

Dependencies:
```
requirements.txt
```
## Evaluation

Primary metrics:
```
AUC_CTR
AUC_CVR
AUC_CTCVR
gAUC
```

Example validation result:
```
AUC_CTR   = 0.833
AUC_CVR   = 0.733
AUC_CTCVR = 0.773
```

## Experiment Tracking

Experiments are tracked using MLflow.

Logged information includes:
```
hyperparameters
training loss
AUC metrics
calibration metrics
```

## Inference API

A lightweight inference server is implemented using FastAPI.

Example endpoint:
```
GET /health
GET /benchmark
POST /predict
POST /predict/batch
```

Example Input:
```
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
```

Output:
```
pCTR
pCVR
pCVR_calibrated
pCTCVR
ecpm
latency
```

## Repository Structure
```
train.py
server.py
evaluate.py

models/
  esmm.py
  sim_cvr.py
  layers.py

data/
  preprocessing scripts

tests/
  test_model.py
  test_dataset.py

utils/
  trainer.py
  metrics.py
  profiler.py
  seed.py
  calibration/
    cost_calibrator.py

runs/
  checkpoints
```

## Reproducibility

Random seeds are set for:
```
python
numpy
pytorch
```

## License

MIT License.

Dataset license follows the Tianchi academic research agreement.

## Acknowledgement

This implementation is inspired by research on industrial recommender systems and the ESMM framework proposed for post-click conversion prediction.
