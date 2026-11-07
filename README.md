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
- **Logging Weights & Biases** (optionnel) — historique de chaque run d'estimation

## Notebook : PySpark à l'échelle

[`offline_ab_simulator.ipynb`](offline_ab_simulator.ipynb) contient l'analyse complète :

1. les trois estimateurs expliqués sur un exemple de 6 lignes ;
2. PySpark sur 1M de logs simulés avec ground truth connue, et convergence des estimateurs selon la taille d'échantillon ;
3. benchmark Pandas vs PySpark ;
4. **Open Bandit Dataset (ZOZOTOWN, 12M lignes)** : IPS, SNIPS et Doubly Robust comparés à la ground truth de la policy random, analyse du clipping, intervalles de confiance à 95 %, puis le même calcul en PySpark.

Les données (~2 Go) sont téléchargées par le notebook et ne sont pas versionnées.

## Format des logs attendus

CSV ou Parquet avec au minimum :

| colonne | description |
|---|---|
| `reward` (ou `click`) | 0 / 1 |
| `propensity_log` (ou `propensity_score`) | proba sous la policy loggée |
| `propensity_new` | proba sous la policy évaluée |
| `product_id` (ou `item_id`) | identifiant de l'action (pour le DR) |

## Weights & Biases — usage

Le logging W&B est **désactivé par défaut** et se déclenche uniquement sur un clic explicite (« 📤 Logger ce run sur W&B »). C'est volontaire : Streamlit recalcule l'app à chaque changement de widget, donc sans ce bouton on créerait un run W&B à chaque interaction.

### Setup rapide

```bash
pip install wandb
export WANDB_API_KEY="ta_cle"   # facultatif : sans clé, tombe en mode offline
streamlit run app.py
```

Sans `WANDB_API_KEY`, l'app propose automatiquement le mode **offline** : tout est stocké dans `./wandb/` et peut être synchronisé plus tard avec `wandb sync ./wandb/latest-run`.

### Ce qui est loggé pour chaque run

| Catégorie | Contenu |
|---|---|
| **Config** | source (demo/upload), n, clip, seed, n_bootstrap, colonnes choisies |
| **Metrics** | V̂_IPS, V̂_SNIPS, V̂_DR + IC 95%, ESS, ESS ratio, max/p99/median des poids, `frac_clipped` |
| **Ground truth** (demo) | `true_value`, `err_IPS`, `err_SNIPS`, `err_DR` |
| **Plots** | histogramme des poids, courbe sensibilité au clipping |
| **Tables** | tableau clip → (IPS, SNIPS) |
| **Artifacts** | rapport CSV versionné |

### Usage typique

- Comparer 10 valeurs de `clip` → dashboard trace V̂ en fonction de τ automatiquement
- Comparer 3 seeds pour vérifier la stabilité du bootstrap
- Comparer plusieurs datasets uploadés → tableau W&B tri/filtre par `source`

## Références

- Dudík, Langford, Li (2011) — *Doubly Robust Policy Evaluation and Learning*
- Swaminathan, Joachims (2015) — *The Self-Normalized Estimator for Counterfactual Learning*
- [Open Bandit Dataset](https://github.com/st-tech/zr-obp) (ZOZO)
- [W&B docs](https://docs.wandb.ai/)
