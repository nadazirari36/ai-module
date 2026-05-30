"""
streamlit_app.py — Application de recommandation de produits.
Interface utilisateur moderne inspirée des systèmes e-commerce (Amazon, FNAC, etc.)

Lancement : streamlit run streamlit_app.py
"""

import streamlit as st
import os, sys, pickle, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import RecommendationEngine
from config import SAVED_MODELS_DIR

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ShopAI — Recommandations Intelligentes",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════════
# CSS GLOBAL
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Fond et typographie ── */
[data-testid="stAppViewContainer"] {
    background: #f5f6fa;
}
[data-testid="stHeader"] { background: transparent; }

/* ── Barre de navigation simulée ── */
.navbar {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 18px 40px;
    border-radius: 0 0 20px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 30px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.navbar-brand {
    font-size: 28px;
    font-weight: 800;
    color: #fff;
    letter-spacing: -0.5px;
}
.navbar-brand span { color: #e94560; }
.navbar-subtitle {
    font-size: 13px;
    color: #a8b2d8;
    margin-top: 2px;
}
.navbar-stats {
    display: flex;
    gap: 30px;
    align-items: center;
}
.nav-stat { text-align: center; }
.nav-stat-value { font-size: 20px; font-weight: 700; color: #e94560; }
.nav-stat-label { font-size: 11px; color: #a8b2d8; text-transform: uppercase; letter-spacing: 1px; }

/* ── Carte de saisie ── */
.search-card {
    background: white;
    border-radius: 20px;
    padding: 32px 36px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    margin-bottom: 28px;
}
.search-title {
    font-size: 22px;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 6px;
}
.search-subtitle {
    font-size: 14px;
    color: #6c757d;
    margin-bottom: 24px;
}

/* ── Chips de clients exemples ── */
.chip-container { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 8px; }
.chip {
    background: #e8f0fe;
    color: #1967d2;
    border: 1px solid #c5d8f8;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    display: inline-block;
}

/* ── Bouton principal ── */
div.stButton > button[kind="primary"], div.stButton > button {
    background: linear-gradient(135deg, #e94560, #c0392b) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 15px rgba(233,69,96,0.4) !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(233,69,96,0.5) !important;
}

/* ── Section résultats ── */
.results-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 20px;
}
.results-title {
    font-size: 24px;
    font-weight: 800;
    color: #1a1a2e;
}
.strategy-badge {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border-radius: 20px;
    padding: 6px 16px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* ── Carte produit ── */
.product-card {
    background: white;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 14px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border-left: 5px solid #e94560;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    gap: 20px;
}
.product-card:hover { box-shadow: 0 6px 24px rgba(0,0,0,0.12); transform: translateX(4px); }
.product-rank {
    font-size: 28px;
    font-weight: 900;
    color: #e94560;
    min-width: 36px;
    text-align: center;
    line-height: 1;
}
.product-icon {
    font-size: 36px;
    min-width: 50px;
    text-align: center;
}
.product-info { flex: 1; }
.product-name {
    font-size: 16px;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 4px;
    line-height: 1.3;
}
.product-id {
    font-size: 12px;
    color: #adb5bd;
    font-family: monospace;
}
.product-score-wrap { text-align: right; min-width: 90px; }
.product-score-value {
    font-size: 22px;
    font-weight: 800;
}
.product-score-label { font-size: 11px; color: #adb5bd; text-transform: uppercase; letter-spacing: 1px; }
.score-A { color: #2ecc71; }
.score-B { color: #f39c12; }
.score-C { color: #e74c3c; }

/* ── Barre de score ── */
.score-bar-wrap { margin-top: 8px; background: #f0f2f6; border-radius: 4px; height: 6px; }
.score-bar { height: 6px; border-radius: 4px; background: linear-gradient(90deg, #e94560, #c0392b); }

/* ── Cartes métriques ── */
.metric-row { display: flex; gap: 16px; margin-bottom: 28px; }
.metric-card {
    flex: 1;
    background: white;
    border-radius: 16px;
    padding: 22px 20px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
}
.metric-card.lstm::before  { background: linear-gradient(90deg, #e94560, #c0392b); }
.metric-card.ens::before   { background: linear-gradient(90deg, #667eea, #764ba2); }
.metric-card.ae::before    { background: linear-gradient(90deg, #11998e, #38ef7d); }
.metric-card.ncf::before   { background: linear-gradient(90deg, #f7971e, #ffd200); }
.metric-label { font-size: 13px; color: #6c757d; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
.metric-value { font-size: 32px; font-weight: 900; color: #1a1a2e; line-height: 1; }
.metric-sub   { font-size: 12px; color: #adb5bd; margin-top: 4px; }
.metric-delta-pos { font-size: 12px; color: #2ecc71; font-weight: 600; margin-top: 4px; }
.metric-delta-neg { font-size: 12px; color: #e74c3c; font-weight: 600; margin-top: 4px; }

/* ── Section performances ── */
.perf-section {
    background: white;
    border-radius: 20px;
    padding: 32px 36px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    margin-bottom: 28px;
}
.section-title {
    font-size: 22px;
    font-weight: 800;
    color: #1a1a2e;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* ── Barre de performance ── */
.perf-row { margin-bottom: 18px; }
.perf-row-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.perf-model { font-size: 14px; font-weight: 700; color: #1a1a2e; }
.perf-pct   { font-size: 14px; font-weight: 800; }
.perf-bar-bg { background: #f0f2f6; border-radius: 8px; height: 12px; }
.perf-bar-fill { height: 12px; border-radius: 8px; }

/* ── Tableau ── */
.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
}
.styled-table th {
    background: #1a1a2e;
    color: white;
    padding: 12px 16px;
    text-align: left;
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.styled-table td { padding: 12px 16px; border-bottom: 1px solid #f0f2f6; color: #495057; }
.styled-table tr:last-child td { border-bottom: none; }
.styled-table tr:hover td { background: #f8f9ff; }
.best-val { color: #e94560; font-weight: 800; }

/* ── Footer ── */
.footer {
    background: #1a1a2e;
    color: #a8b2d8;
    border-radius: 20px;
    padding: 30px 40px;
    margin-top: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.footer-brand { font-size: 20px; font-weight: 800; color: white; }
.footer-brand span { color: #e94560; }
.footer-copy { font-size: 12px; color: #6c757d; margin-top: 4px; }
.footer-team { font-size: 13px; color: #a8b2d8; text-align: right; line-height: 1.8; }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    border-radius: 10px !important;
    border: 2px solid #e9ecef !important;
    font-size: 15px !important;
    transition: border-color 0.2s !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #e94560 !important;
    box-shadow: 0 0 0 3px rgba(233,69,96,0.1) !important;
}
.stSlider > div { padding-top: 6px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# CHARGEMENT DES MODÈLES
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource
def load_engine():
    eng = RecommendationEngine()
    eng.load()
    return eng

engine = load_engine()

# ═══════════════════════════════════════════════════════════════════
# BARRE DE NAVIGATION
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="navbar">
  <div>
    <div class="navbar-brand">Shop<span>AI</span> &nbsp;🛒</div>
    <div class="navbar-subtitle">Moteur de recommandation intelligent — Deep Learning Ensemble</div>
  </div>
  <div class="navbar-stats">
    <div class="nav-stat">
      <div class="nav-stat-value">4 339</div>
      <div class="nav-stat-label">Clients</div>
    </div>
    <div class="nav-stat">
      <div class="nav-stat-value">3 665</div>
      <div class="nav-stat-label">Produits</div>
    </div>
    <div class="nav-stat">
      <div class="nav-stat-value">397K</div>
      <div class="nav-stat-label">Transactions</div>
    </div>
    <div class="nav-stat">
      <div class="nav-stat-value">30.6%</div>
      <div class="nav-stat-label">Précision@5</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# SECTION SAISIE + RÉSULTATS
# ═══════════════════════════════════════════════════════════════════

col_input, col_results = st.columns([1, 1.6], gap="large")

with col_input:
    st.markdown("""
    <div class="search-card">
      <div class="search-title">🔍 Trouver des produits</div>
      <div class="search-subtitle">Entrez un identifiant client et ses achats récents pour obtenir des recommandations personnalisées.</div>
    </div>
    """, unsafe_allow_html=True)

    customer_id = st.text_input(
        "Identifiant client",
        value="17850",
        placeholder="Ex : 17850",
        help="CustomerID du client"
    )

    st.markdown("**Clients exemples à tester :**")
    st.markdown("""
    <div class="chip-container">
      <span class="chip">👤 17850 — Décoration lumineuse</span>
      <span class="chip">👤 15168 — Décoration intérieure</span>
      <span class="chip">👤 12792 — Fournitures de fête</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    purchased_input = st.text_area(
        "Articles déjà achetés (séparés par des virgules)",
        value="85123A, 71053",
        height=90,
        placeholder="Ex : 85123A, 71053, 84406B",
        help="Codes StockCode des produits déjà achetés par ce client"
    )

    st.markdown("**Produits exemples :**")
    st.markdown("""
    <div class="chip-container">
      <span class="chip">85123A — Set thé blanc</span>
      <span class="chip">71053 — Sac fourre-tout</span>
      <span class="chip">84406B — Décoration florale</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    n_recs = st.slider("Nombre de recommandations", 1, 10, 5)

    st.markdown("<br>", unsafe_allow_html=True)
    get_recs = st.button("🚀 Générer les recommandations", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# PANNEAU RÉSULTATS
# ═══════════════════════════════════════════════════════════════════

ICONS = ["🎁","🛍️","🏠","🌸","🕯️","🎨","🎪","💎","🌿","⭐"]

with col_results:
    if get_recs:
        purchased_items = [i.strip().upper() for i in purchased_input.split(",") if i.strip()]

        with st.spinner("Analyse en cours…"):
            try:
                recs, strategy = engine.recommend(
                    user_id=customer_id,
                    purchased_items=purchased_items,
                    top_n=n_recs
                )

                strategy_labels = {
                    'ensemble_autoencoder_ncf_lstm': '🔀 Ensemble IA (AE + NCF + LSTM)',
                    'single_model':                  '🔷 Modèle unique',
                    'popular_fallback':               '⭐ Tendances populaires',
                }
                strat_label = strategy_labels.get(strategy, strategy)

                st.markdown(f"""
                <div class="results-header">
                  <div class="results-title">✨ Recommandations pour <em>{customer_id}</em></div>
                  <div class="strategy-badge">{strat_label}</div>
                </div>
                """, unsafe_allow_html=True)

                if recs:
                    for i, rec in enumerate(recs):
                        score = float(rec['score'])
                        pct   = int(min(score * 100, 100))
                        icon  = ICONS[i % len(ICONS)]

                        if score >= 0.30:
                            score_cls, score_txt = "score-A", "Excellent"
                        elif score >= 0.15:
                            score_cls, score_txt = "score-B", "Bon"
                        else:
                            score_cls, score_txt = "score-C", "Moyen"

                        name = rec['description'][:65] + ("…" if len(rec['description']) > 65 else "")

                        st.markdown(f"""
                        <div class="product-card">
                          <div class="product-rank">#{i+1}</div>
                          <div class="product-icon">{icon}</div>
                          <div class="product-info">
                            <div class="product-name">{name}</div>
                            <div class="product-id">ID : {rec['product_id']}</div>
                            <div class="score-bar-wrap">
                              <div class="score-bar" style="width:{pct}%"></div>
                            </div>
                          </div>
                          <div class="product-score-wrap">
                            <div class="product-score-value {score_cls}">{score:.3f}</div>
                            <div class="product-score-label">{score_txt}</div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("Aucune recommandation trouvée pour ce client.")

            except Exception as e:
                st.error(f"❌ Erreur : {e}")

    else:
        # Placeholder avant la première recherche
        st.markdown("""
        <div style="background:white; border-radius:20px; padding:60px 40px;
                    text-align:center; box-shadow:0 4px 24px rgba(0,0,0,0.08);
                    border: 2px dashed #e9ecef;">
          <div style="font-size:64px; margin-bottom:20px;">🎯</div>
          <div style="font-size:22px; font-weight:800; color:#1a1a2e; margin-bottom:10px;">
            Prêt à recommander
          </div>
          <div style="font-size:15px; color:#6c757d; max-width:320px; margin:0 auto;">
            Renseignez un client et ses articles dans le formulaire,
            puis cliquez sur <strong>Générer les recommandations</strong>.
          </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# SECTION PERFORMANCES
# ═══════════════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="section-title">📊 Performances des modèles</div>
""", unsafe_allow_html=True)

# Cartes métriques
st.markdown("""
<div class="metric-row">
  <div class="metric-card lstm">
    <div class="metric-label">🏆 LSTM</div>
    <div class="metric-value">30.61%</div>
    <div class="metric-sub">Précision@5</div>
    <div class="metric-delta-pos">↑ Meilleur modèle</div>
  </div>
  <div class="metric-card ens">
    <div class="metric-label">🔀 Ensemble</div>
    <div class="metric-value">29.28%</div>
    <div class="metric-sub">Précision@5</div>
    <div class="metric-delta-neg">↓ -1.33% vs LSTM</div>
  </div>
  <div class="metric-card ae">
    <div class="metric-label">📦 AutoEncodeur</div>
    <div class="metric-value">6.26%</div>
    <div class="metric-sub">Précision@5</div>
    <div class="metric-delta-pos">↑ +131% amélioré</div>
  </div>
  <div class="metric-card ncf">
    <div class="metric-label">🧠 NCF</div>
    <div class="metric-value">3.84%</div>
    <div class="metric-sub">Précision@5</div>
    <div class="metric-delta-neg">↓ Cold-start users</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Barres de performance + tableau côte à côte
col_bars, col_table = st.columns([1, 1.4], gap="large")

with col_bars:
    st.markdown('<div class="perf-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Précision@5</div>', unsafe_allow_html=True)

    models_perf = [
        ("LSTM",         30.61, "#e94560", "linear-gradient(90deg,#e94560,#c0392b)"),
        ("Ensemble",     29.28, "#764ba2", "linear-gradient(90deg,#667eea,#764ba2)"),
        ("AutoEncodeur",  6.26, "#11998e", "linear-gradient(90deg,#11998e,#38ef7d)"),
        ("NCF",           3.84, "#f7971e", "linear-gradient(90deg,#f7971e,#ffd200)"),
    ]
    max_val = 35.0

    for name, val, color, grad in models_perf:
        bar_w = int(val / max_val * 100)
        st.markdown(f"""
        <div class="perf-row">
          <div class="perf-row-header">
            <span class="perf-model">{name}</span>
            <span class="perf-pct" style="color:{color}">{val}%</span>
          </div>
          <div class="perf-bar-bg">
            <div class="perf-bar-fill" style="width:{bar_w}%; background:{grad};"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with col_table:
    st.markdown('<div class="perf-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Tableau complet des métriques</div>', unsafe_allow_html=True)

    csv_path = os.path.join(SAVED_MODELS_DIR, 'evaluation_results.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        best_p = df.groupby('Model')['Precision@K'].max()
        html = """
        <table class="styled-table">
          <thead><tr>
            <th>Modèle</th><th>K</th>
            <th>Précision</th><th>Rappel</th><th>NDCG</th>
          </tr></thead><tbody>
        """
        for _, row in df.iterrows():
            is_best = float(row['Precision@K']) == float(best_p.max())
            cls = ' class="best-val"' if is_best else ''
            html += f"""
            <tr>
              <td><strong>{row['Model']}</strong></td>
              <td>@{int(row['K'])}</td>
              <td{cls}>{float(row['Precision@K']):.4f}</td>
              <td>{float(row['Recall@K']):.4f}</td>
              <td>{float(row['NDCG@K']):.4f}</td>
            </tr>"""
        html += "</tbody></table>"
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.info("Lancez `python train_pipeline.py` pour générer les résultats.")

    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# SECTION AMÉLIORATIONS
# ═══════════════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)

with st.expander("🔧 Améliorations apportées aux modèles"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        **📦 AutoEncodeur**
        | Paramètre | Avant | Après |
        |---|---|---|
        | pos_weight | 40 | **15** |
        | Bruit | 0.30 | **0.15** |
        | Goulot | 128D | **256D** |
        | Époques | 80 | **100** |
        *2.71 % → **6.26 %** (+131 %)*
        """)
    with c2:
        st.markdown("""
        **🧠 NCF**
        | Paramètre | Avant | Après |
        |---|---|---|
        | neg_ratio | 2 | **5** |
        | Époques | 25 | **50** |
        | Couches MLP | 3 | **4 + BN** |
        | Patience | 5 | **8** |
        *Négatifs pondérés par popularité*
        """)
    with c3:
        st.markdown("""
        **⚙️ Pipeline**
        | Paramètre | Avant | Après |
        |---|---|---|
        | Split | 80/10/10 | **70/10/20** |
        | Test users | 435 | **869** |
        | LSTM units | 128 | **256** |
        | Dataset | sample | **training.xlsx** |
        *+99 % d'utilisateurs évalués*
        """)

with st.expander("📖 Architecture du système"):
    st.markdown("""
    ```
    Client + Historique d'achats
           │
           ├──► AutoEncodeur  ──► Score de goût général    (×0.10)
           │     3665→512→256→256→256→512→3665
           │
           ├──► NCF / NeuMF   ──► Score de compatibilité    (×0.35)
           │     GMF: emb_user × emb_item
           │     MLP: 512→256→128→64 + BatchNorm
           │
           └──► LSTM          ──► Score séquentiel          (×0.55)
                 Embedding(64) → LSTM(256) → LSTM(128) → Dense(3665)
                        │
                        ▼
              Fusion pondérée + Normalisation min-max
                        │
                        ▼
              Top-N recommandations (articles non encore achetés)
    ```
    """)

# ═══════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="footer">
  <div>
    <div class="footer-brand">Shop<span>AI</span> 🛒</div>
    <div class="footer-copy">Système de recommandation par ensemble Deep Learning</div>
    <div class="footer-copy" style="margin-top:4px">
      Training.xlsx · 4 339 clients · 3 665 produits · 397 016 transactions
    </div>
  </div>
  <div class="footer-team">
    <strong style="color:white">Équipe</strong><br>
    Nada Zirari · Ziyad Belahmar<br>
    Karim Bekkali · Youssef El Azami<br>
    <span style="color:#6c757d">Projet Module IA — 2024-2025</span>
  </div>
</div>
""", unsafe_allow_html=True)
