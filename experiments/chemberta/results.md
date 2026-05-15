# ChemBERTa run — mean-pool ESM2 + ChemBERTa + protein family embedding

**Date:** 2026-05-15  
**Branch:** `main`  
**Checkpoint:** `checkpoints/chemberta/best.pt`  
**Config:** `config/default.yaml`

---

## Model

| Component | Details |
|-----------|---------|
| Protein encoder | ESM2-650M mean-pool [1280] → CrossAttentionPooling → [256] |
| Ligand encoder | ChemBERTa CLS token [768] → Linear projection → [256] (backbone frozen, precomputed) |
| Family embedding | nn.Embedding(6, 16) — kinase / enzyme / GPCR / nuclear receptor / phosphatase / other |
| Fusion | concat [256 + 256 + 16 = 528] → MLP(512 → 256 → 1) → pIC50 |
| Total parameters | 1,320,801 |

Key config: `normalize_targets=false`, `checkpoint_monitor="spearman"`, `ranking_weight=0.1`, `batch_size=128`, `lr=1e-4`, cosine schedule, `patience=15`.

---

## Training

- Best checkpoint at **epoch 6 / 50** (early stopping on global Spearman)
- `val_spearman=0.409`, `val_mae=0.993`

---

## Evaluation — val and test

| Split | Spearman | Pearson | MAE | RMSE |
|-------|----------|---------|-----|------|
| Val | 0.401 | 0.414 | 0.996 | 1.233 |
| Test | 0.410 | 0.400 | 1.037 | 1.302 |

Val/test gap is small — no overfitting.

### vs baseline

| Metric | Baseline | ChemBERTa | Δ |
|--------|----------|-----------|---|
| Val Spearman | 0.394 | 0.401 | +0.007 |
| Test Spearman | 0.386 | **0.410** | **+0.024** |
| Val MAE | 1.035 | **0.996** | **−0.039** |
| Test MAE | 1.130 | **1.037** | **−0.093** |

ChemBERTa improves global Spearman by +0.024 on the test set and reduces MAE by 0.093 — the stronger chemical encoder adds real signal.

---

## Baseline comparison (test set)

| Model | Spearman |
|-------|----------|
| **Oracle (ChemBERTa)** | **0.410** |
| AEVPLig | 0.357 |
| Vina (neg) | 0.342 |
| OnionNet | 0.338 |
| Vinardo (neg) | 0.312 |

Oracle is the strongest model on global Spearman.

---

## Per-protein Spearman (deployment metric)

| Split | Mean | Median | Proteins with ρ < 0 |
|-------|------|--------|---------------------|
| Val | 0.173 | 0.200 | 87 / 354 (25%) |
| Test | 0.167 | 0.222 | 84 / 350 (24%) |

### vs baseline

| Split | Baseline median | ChemBERTa median | Δ |
|-------|----------------|------------------|---|
| Val | 0.276 | 0.200 | −0.076 |
| Test | 0.278 | 0.222 | −0.056 |

Per-protein Spearman is worse than the baseline despite higher global Spearman. ChemBERTa improves global ranking (driven by 41k samples) but has not improved within-protein discrimination. The model learns between-protein pIC50 scale better, but within-protein ranking stays weak.

### Per-family breakdown (test)

| Family | Proteins | Mean ρ | Median ρ | vs baseline median |
|--------|----------|--------|----------|--------------------|
| kinase | 62 | 0.363 | 0.378 | −0.162 |
| phosphatase | 17 | 0.079 | 0.256 | +0.005 |
| GPCR | 14 | 0.119 | 0.208 | −0.134 |
| other | 62 | 0.157 | 0.198 | −0.069 |
| enzyme | 123 | 0.091 | 0.164 | −0.035 |
| nuclear receptor | 9 | 0.155 | 0.117 | −0.254 |

All families except phosphatase regress on per-protein Spearman. Kinases are still the strongest family (median 0.38). Enzymes remain the weakest (median 0.16) and most numerous (123/287 proteins).

---

## Known limitations

- **Per-protein Spearman regresses vs baseline** despite higher global Spearman: ChemBERTa embeddings are more informative globally but the signal improvement has not translated to within-protein discrimination.
- **Early stopping at epoch 6**: training converged quickly — the model may benefit from a longer cosine cycle or different scheduler.
- **24% of proteins have negative per-protein Spearman**: oracle would actively mislead the GFlowNet for those targets.
- **Constant predictions on some proteins**: ConstantInputWarning during per-protein Spearman computation — zero within-protein variance in predictions.

---

## Potential next directions

- Unfreeze last 1–2 ChemBERTa layers with a much lower learning rate to allow ligand fine-tuning on pIC50 signal
- ECFP4 fingerprints concatenated with ChemBERTa projection as an additional inductive bias for within-protein discrimination
- Per-family loss weighting to up-weight the under-served enzyme class (43% of proteins, weakest median ρ)
- Increase ranking_weight (currently 0.1) — if the deployment goal is within-protein ranking, stronger pairwise ranking supervision may help
- Longer training: `patience=15` on global Spearman may exit before per-protein Spearman has time to improve
