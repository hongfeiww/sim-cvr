# sim-cvr
# SIM-lite: Session-based Interest Model for CVR Prediction

A lightweight re-implementation of **Search-based Interest Model (SIM)** for CVR prediction,
inspired by industrial-scale ranking systems. This project explores how **long-term user
behavior sequences** can be efficiently modeled under strict latency constraints.

> **Background:** In production ad systems (e.g., short-video platforms), a user may have
> thousands of historical interactions. Naively feeding all of them into a DNN is infeasible.
> SIM addresses this with a two-stage retrieval + attention mechanism.

---

## Motivation

At large-scale recommendation platforms, a key challenge is:

- Users have **very long behavior sequences** (5000+ items)
- Model inference must complete in **<30ms**
- Relevant historical behaviors are **sparse** within the full sequence

SIM solves this by:
1. **General Search Unit (GSU):** Fast top-K retrieval from long sequence using category matching or inner product
2. **Exact Search Unit (ESU):** Deep attention over the retrieved K items only

This repo implements SIM-lite on public datasets with full experiment tracking, error analysis, and profiling.

---

inspired by industrial recommender systems.
