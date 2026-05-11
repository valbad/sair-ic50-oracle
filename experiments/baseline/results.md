# Baseline run — mean-pool ESM2 + GINEConv

**Date:** 2026-04-28  
**Branch:** `main`  
**Checkpoint:** `checkpoints/best.pt`  
**Config:** `config/default.yaml`

---

## Model

| Component | Details |
|-----------|---------|
| Protein encoder | ESM2-650M mean-pool [1280] → linear projection [256] (via CrossAttentionPooling L=1) |
| Ligand encoder | GINEConv × 4, hidden_dim=256, global_mean_pool |
| Fusion | concat [512] → MLP → pIC50 |
| Total parameters | 1,658,113 |

Key config: `ranking_weight=0.1`, `batch_size=128`, `lr=1e-4`, cosine schedule, `patience=15`.

---

## Training

- Stopped at **epoch 27 / 50** (early stopping on per-protein Spearman)
- `val_spearman=0.396`, `val_mae=1.03`

---

## Evaluation — val and test

| Split | Spearman | Pearson | MAE | RMSE |
|-------|----------|---------|-----|------|
| Val | 0.394 | 0.415 | 1.035 | 1.299 |
| Test | 0.386 | 0.387 | 1.130 | 1.414 |

Val/test gap is small — no overfitting.

---

## Baseline comparison (test set)

| Model | Spearman |
|-------|----------|
| **Oracle** | **0.386** |
| AEVPLig | 0.357 |
| Vina (neg) | 0.342 |
| OnionNet | 0.338 |
| Vinardo (neg) | 0.312 |

Oracle beats all SAIR baselines on global Spearman.

---

## Per-protein Spearman (deployment metric)

| Split | Mean | Median | Proteins with ρ < 0 |
|-------|------|--------|---------------------|
| Val | 0.228 | 0.276 | 78 / 354 (22%) |
| Test | 0.244 | 0.278 | 76 / 350 (22%) |

Median per-protein Spearman of **0.28** is the honest GFlowNet deployment number. Global Spearman (0.39) is inflated by between-protein pIC50 scale differences.

### Per-family breakdown (test)

| Family | Proteins | Mean ρ | Median ρ |
|--------|----------|--------|----------|
| kinase | 62 | 0.423 | **0.540** |
| nuclear receptor | 9 | 0.313 | 0.371 |
| GPCR | 14 | 0.286 | 0.342 |
| other | 61 | 0.197 | 0.267 |
| phosphatase | 17 | 0.180 | 0.251 |
| enzyme | 122 | 0.175 | **0.199** |

Kinases are the strongest family (median 0.54). Enzymes are the weakest (median 0.20) and the most common class in the dataset (122/285 test proteins).

---

## Known limitations

- **Mean-pool protein encoding**: protein representation is static regardless of ligand. All within-protein discrimination comes from the ligand encoder alone. Per-residue cross-attention was tested but degraded accuracy — dataset too small.
- **22% of proteins have negative per-protein Spearman**: oracle would actively mislead the GFlowNet for those targets.
- **Constant predictions on some proteins**: ConstantInputWarning during per-protein Spearman computation — ligand encoder contributes zero signal for those proteins.
- **Prediction range [4.7, 8.1]**: model undershoots very potent (<5) and very weak (>9) compounds.

---

## Potential next directions

- 3D graph features (from AlphaFold-docked poses already in SAIR)
- ECFP fingerprints as additional ligand descriptors
- Per-family fine-tuning or family-conditional heads (especially to improve enzymes)
- Larger GNN (hidden_dim 256→512, n_layers 4→6)
- Protein-family-stratified loss weighting to up-weight the weak enzyme family
