"""
Offline A/B Test Simulator — Streamlit app
Estime la performance d'une nouvelle policy sans la deployer,
a partir des logs d'une policy existante.

Run: streamlit run app.py
"""

import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Offline A/B Test Simulator",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Offline A/B Test Simulator")
st.caption(
    "Estime la performance d'une nouvelle policy sans la déployer — "
    "IPS, SNIPS, Doubly Robust, avec clipping et intervalles de confiance."
)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------
def compute_weights(prop_new: np.ndarray, prop_log: np.ndarray, clip: float | None):
    w = prop_new / prop_log
    if clip is not None:
        w = np.clip(w, a_min=None, a_max=clip)
    return w


def ips(w, r):
    return float((w * r).mean())


def snips(w, r):
    s = w.sum()
    if s == 0:
        return float("nan")
    return float((w * r).sum() / s)


def doubly_robust(w, r, r_hat, direct_term):
    correction = float((w * (r - r_hat)).mean())
    return float(direct_term + correction)


def bootstrap_ci(values_fn, n_boot: int, n: int, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        estimates[i] = values_fn(idx)
    low = float(np.percentile(estimates, 100 * alpha / 2))
    high = float(np.percentile(estimates, 100 * (1 - alpha / 2)))
    return low, high, estimates


def diagnose_weights(w: np.ndarray) -> list[tuple[str, str]]:
    """Return list of (severity, message). severity in {'ok', 'warn', 'error'}."""
    msgs = []
    p99 = float(np.percentile(w, 99))
    p_max = float(w.max())
    frac_extreme = float((w > 100).mean())
    ess = float((w.sum() ** 2) / (w**2).sum())  # effective sample size
    ess_ratio = ess / len(w)

    if p_max > 1000:
        msgs.append(
            ("error", f"Poids max = {p_max:.0f} (>1000). Estimateur IPS très instable — clipping fortement recommandé.")
        )
    elif p_max > 100:
        msgs.append(
            ("warn", f"Poids max = {p_max:.0f} (>100). Envisage SNIPS ou clipping.")
        )

    if frac_extreme > 0.01:
        msgs.append(
            ("warn", f"{frac_extreme*100:.1f}% des poids dépassent 100 — variance élevée.")
        )

    if ess_ratio < 0.1:
        msgs.append(
            ("error", f"Effective sample size = {ess:.0f} ({ess_ratio*100:.1f}% de n). Les policies sont trop différentes.")
        )
    elif ess_ratio < 0.3:
        msgs.append(
            ("warn", f"Effective sample size = {ess:.0f} ({ess_ratio*100:.1f}% de n). Overlap modéré.")
        )
    else:
        msgs.append(("ok", f"Effective sample size = {ess:.0f} ({ess_ratio*100:.1f}% de n). Bon overlap."))

    msgs.append(("ok", f"Poids: min={w.min():.3f}, médiane={np.median(w):.3f}, p99={p99:.2f}, max={p_max:.2f}"))
    return msgs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def make_demo_data(n: int = 50_000, n_products: int = 50, seed: int = 42):
    rng = np.random.default_rng(seed)
    policy_log = rng.dirichlet(np.ones(n_products) * 5)
    policy_new = rng.dirichlet(np.ones(n_products) * 2)
    true_ctr = rng.beta(2, 20, size=n_products)

    products = rng.choice(n_products, size=n, p=policy_log)
    rewards = rng.binomial(1, true_ctr[products])

    df = pd.DataFrame(
        {
            "product_id": products,
            "reward": rewards,
            "propensity_log": policy_log[products],
            "propensity_new": policy_new[products],
        }
    )
    true_value = float(np.sum(policy_new * true_ctr))
    return df, true_value


def load_uploaded(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(file)
    if name.endswith((".csv", ".tsv")):
        sep = "\t" if name.endswith(".tsv") else ","
        return pd.read_csv(file, sep=sep)
    raise ValueError("Format non supporté. Utilise .parquet, .csv ou .tsv.")


# ---------------------------------------------------------------------------
# Sidebar — data + params
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Données")
    source = st.radio(
        "Source des logs",
        ["Demo (50k logs simulés)", "Upload mes logs"],
        index=0,
    )

    df = None
    true_value = None

    if source.startswith("Demo"):
        df, true_value = make_demo_data()
        st.success(f"Demo chargée: {len(df):,} logs — vraie valeur connue = {true_value:.4f}")
    else:
        up = st.file_uploader("Upload logs (.parquet / .csv)", type=["parquet", "csv", "tsv"])
        if up is not None:
            try:
                df = load_uploaded(up)
                st.success(f"Chargé: {len(df):,} lignes")
            except Exception as e:
                st.error(f"Erreur: {e}")

    st.header("2. Colonnes")
    if df is not None:
        cols = list(df.columns)

        def pick(label, candidates, default_idx=0):
            for c in candidates:
                if c in cols:
                    return st.selectbox(label, cols, index=cols.index(c))
            return st.selectbox(label, cols, index=min(default_idx, len(cols) - 1))

        col_reward = pick("Reward (0/1)", ["reward", "click"])
        col_prop_log = pick("Propensity policy loggée", ["propensity_log", "propensity_score", "propensity_a"])
        col_prop_new = pick("Propensity policy évaluée", ["propensity_new", "propensity_b"])
        col_item = pick("Item / product id (pour DR)", ["product_id", "item_id"])

    st.header("3. Paramètres")
    clip_on = st.checkbox("Activer le clipping", value=True)
    clip_val = st.slider("Clip max des poids", 1, 1000, 100, disabled=not clip_on)
    n_boot = st.slider("Bootstrap iterations (CI)", 100, 2000, 500, step=100)
    seed = st.number_input("Random seed", value=0, step=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if df is None:
    st.info("👈 Charge un dataset dans la sidebar pour commencer.")
    st.stop()

# Preview
with st.expander("Aperçu des données"):
    st.dataframe(df.head(20), use_container_width=True)
    st.text(f"Shape: {df.shape}")

# Extract arrays
try:
    r = df[col_reward].to_numpy().astype(float)
    p_log = df[col_prop_log].to_numpy().astype(float)
    p_new = df[col_prop_new].to_numpy().astype(float)
    items = df[col_item].to_numpy()
except Exception as e:
    st.error(f"Erreur sur les colonnes sélectionnées: {e}")
    st.stop()

# Guardrails
if (p_log <= 0).any():
    st.error("Propensity loggée contient des valeurs ≤ 0. IPS est indéfini — nettoie tes logs.")
    st.stop()

clip = clip_val if clip_on else None
w = compute_weights(p_new, p_log, clip)

# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------
st.subheader("🔎 Diagnostic")
diag = diagnose_weights(w)
for sev, msg in diag:
    if sev == "error":
        st.error(msg)
    elif sev == "warn":
        st.warning(msg)
    else:
        st.info(msg)

# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------
st.subheader("📈 Estimations")

n = len(df)
ips_hat = ips(w, r)
snips_hat = snips(w, r)

# Doubly Robust: split moitié pour fit reward model, moitié pour eval (cross-fitting simple)
rng = np.random.default_rng(seed)
mask_train = rng.random(n) < 0.5
r_hat_map = pd.Series(r[mask_train]).groupby(pd.Series(items[mask_train])).mean()
r_hat_full = pd.Series(items).map(r_hat_map).fillna(r_hat_map.mean()).to_numpy()

# Direct term: E_{a ~ policy_new}[r_hat(a)] approximé via reweight
# On approxime en pondérant chaque log par prop_new / uniforme, puis mean * n_items est pas simple
# → approximation raisonnable: mean de r_hat pondéré par prop_new / prop_log (Direct method plug-in)
direct_term = float((p_new / p_log * r_hat_full * mask_train).sum() / max(mask_train.sum(), 1))

w_eval = w.copy()
w_eval[mask_train] = 0  # on n'utilise que la moitié eval pour la correction
n_eval = int((~mask_train).sum())
correction = float((w_eval * (r - r_hat_full)).sum() / max(n_eval, 1))
dr_hat = direct_term + correction

# CI IPS: formule fermée
wr = w * r
ips_ci = 1.96 * wr.std() / np.sqrt(n)

# CI SNIPS: bootstrap
def snips_boot(idx):
    return snips(w[idx], r[idx])

snips_lo, snips_hi, _ = bootstrap_ci(snips_boot, n_boot=n_boot, n=n, seed=seed)

# Display
cols_metrics = st.columns(3)
with cols_metrics[0]:
    st.metric("IPS", f"{ips_hat:.4f}", help="Inverse Propensity Score — non biaisé, haute variance")
    st.caption(f"95% CI: [{ips_hat - ips_ci:.4f}, {ips_hat + ips_ci:.4f}]")
with cols_metrics[1]:
    st.metric("SNIPS", f"{snips_hat:.4f}", help="Self-Normalized IPS — biaisé mais plus stable")
    st.caption(f"95% CI (bootstrap): [{snips_lo:.4f}, {snips_hi:.4f}]")
with cols_metrics[2]:
    st.metric("Doubly Robust", f"{dr_hat:.4f}", help="Reward model + correction IPS (cross-fitted)")
    st.caption("Non biaisé si soit le reward model soit les propensities sont corrects")

if true_value is not None:
    st.success(f"🎯 Vraie valeur (demo): **{true_value:.4f}**")
    err = pd.DataFrame(
        {
            "Estimateur": ["IPS", "SNIPS", "DR"],
            "Estimation": [ips_hat, snips_hat, dr_hat],
            "Erreur absolue": [abs(ips_hat - true_value), abs(snips_hat - true_value), abs(dr_hat - true_value)],
            "Erreur %": [
                abs(ips_hat - true_value) / true_value * 100,
                abs(snips_hat - true_value) / true_value * 100,
                abs(dr_hat - true_value) / true_value * 100,
            ],
        }
    )
    st.dataframe(err, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
st.subheader("📉 Analyse")

tab1, tab2 = st.tabs(["Distribution des poids", "Sensibilité au clipping"])

with tab1:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(np.log10(np.clip(w, 1e-6, None)), bins=60, color="#4C78A8")
    ax.set_xlabel("log10(poids IPS)")
    ax.set_ylabel("fréquence")
    ax.set_title("Distribution des poids d'importance")
    st.pyplot(fig, use_container_width=True)

with tab2:
    clip_grid = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    rows = []
    for c in clip_grid:
        wc = compute_weights(p_new, p_log, c)
        rows.append({"clip": c, "IPS": ips(wc, r), "SNIPS": snips(wc, r)})
    clip_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(clip_df["clip"], clip_df["IPS"], marker="o", label="IPS")
    ax.plot(clip_df["clip"], clip_df["SNIPS"], marker="s", label="SNIPS")
    if true_value is not None:
        ax.axhline(true_value, color="red", ls="--", label="vraie valeur")
    ax.set_xscale("log")
    ax.set_xlabel("clip max")
    ax.set_ylabel("estimation")
    ax.set_title("Sensibilité au clipping")
    ax.legend()
    st.pyplot(fig, use_container_width=True)

    st.dataframe(clip_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
st.subheader("💾 Export")
report = pd.DataFrame(
    {
        "metric": ["IPS", "IPS_CI_low", "IPS_CI_high", "SNIPS", "SNIPS_CI_low", "SNIPS_CI_high", "DR", "n", "clip"],
        "value": [
            ips_hat,
            ips_hat - ips_ci,
            ips_hat + ips_ci,
            snips_hat,
            snips_lo,
            snips_hi,
            dr_hat,
            n,
            clip if clip is not None else "none",
        ],
    }
)
buf = io.StringIO()
report.to_csv(buf, index=False)
st.download_button("Télécharger le rapport (CSV)", buf.getvalue(), "offpolicy_report.csv", "text/csv")

st.caption(
    "Rappel: IPS est non biaisé mais haute variance. SNIPS est biaisé mais stable. "
    "DR combine les deux au prix d'un reward model à entraîner."
)
