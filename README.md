---
title: Offline A/B Test Simulator
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
license: mit
---

# Offline A/B Test Simulator

Estime la performance d'une nouvelle policy de recommandation **sans la déployer**, à partir des logs d'une policy existante.

## Estimateurs

- **IPS** (Inverse Propensity Score) — non biaisé, haute variance
- **SNIPS** (Self-Normalized IPS) — biaisé mais stable
- **Doubly Robust** — reward model + correction IPS avec cross-fitting

## Features

- Diagnostic automatique (effective sample size, overlap des policies, poids extrêmes)
- Intervalles de confiance à 95% (formule fermée pour IPS, bootstrap pour SNIPS)
- Analyse de sensibilité au clipping
- Export CSV du rapport
- Demo intégrée avec ground truth connue

## Format des logs attendus

CSV ou Parquet avec au minimum:

| colonne | description |
|---|---|
| `reward` (ou `click`) | 0 / 1 |
| `propensity_log` (ou `propensity_score`) | proba sous la policy loggée |
| `propensity_new` | proba sous la policy évaluée |
| `product_id` (ou `item_id`) | identifiant de l'action (pour le DR) |

## Références

- Dudík, Langford, Li (2011) — *Doubly Robust Policy Evaluation and Learning*
- Swaminathan, Joachims (2015) — *The Self-Normalized Estimator for Counterfactual Learning*
- [Open Bandit Dataset](https://github.com/st-tech/zr-obp) (ZOZO)