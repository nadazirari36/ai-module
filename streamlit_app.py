"""
streamlit_app.py — Interface web du système de recommandation IA.
Affiche les recommandations personnalisées et les performances des modèles.

Lancement :
    streamlit run streamlit_app.py
Puis ouvrir dans le navigateur : http://localhost:8501
"""

import streamlit as st
import pickle
import numpy as np
import os
import sys
import pandas as pd

# Ajoute le répertoire du module au chemin Python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import RecommendationEngine
from config import SAVED_MODELS_DIR

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION DE LA PAGE
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Système de Recommandation IA",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════
# STYLE CSS
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .recommendation-item {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 4px solid #1f77b4;
    }
    .score-high   { color: #31a354; font-weight: bold; }
    .score-medium { color: #ff7f0e; font-weight: bold; }
    .score-low    { color: #e34234; font-weight: bold; }
    .metric-card {
        background: #f8f9fa;
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# CHARGEMENT DES MODÈLES (UNE SEULE FOIS, MIS EN CACHE)
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource
def load_engine():
    """Charge le moteur de recommandation avec tous les modèles entraînés."""
    eng = RecommendationEngine()
    eng.load()
    return eng

engine = load_engine()

# ═══════════════════════════════════════════════════════════════════
# EN-TÊTE
# ═══════════════════════════════════════════════════════════════════

st.title("🎯 Système de Recommandation de Produits Intelligent")
st.markdown("**Ensemble Deep Learning : AutoEncodeur + NCF + LSTM**")
st.markdown("*Entraîné sur training.xlsx — 4 339 clients, 3 665 produits, 397 016 transactions*")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════
# BARRE LATÉRALE — ÉTAT DU SYSTÈME
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("📊 État des modèles")

    # Vérifie dynamiquement si chaque modèle est chargé
    ae_ok   = engine.ae_model   is not None
    ncf_ok  = engine.ncf_model  is not None
    lstm_ok = engine.lstm_model is not None

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**AE**")
        st.success("✅") if ae_ok   else st.error("❌")
    with col2:
        st.write("**NCF**")
        st.success("✅") if ncf_ok  else st.error("❌")
    with col3:
        st.write("**LSTM**")
        st.success("✅") if lstm_ok else st.error("❌")

    st.markdown("---")

    st.header("📈 Statistiques du jeu de données")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Clients",   "4 339")
        st.metric("Produits",  "3 665")
    with col2:
        st.metric("Transactions", "397 016")
        st.metric("Utilisateurs test", "869")

    st.markdown("---")

    st.header("🏆 Meilleures performances")
    # Résultats issus de la dernière évaluation (evaluate_on_data.py)
    st.metric("LSTM  Précision@5",     "30.61%", delta="Meilleur modèle")
    st.metric("Ensemble Précision@5",  "29.28%", delta="-1.33%")
    st.metric("AE  Précision@5",       "6.26%",  delta="+3.55% vs ancienne version")

    st.markdown("---")
    st.caption("Découpage : 70 % train / 10 % val / 20 % test")
    st.caption("NCF + LSTM entraînés sur 80 % (train + val)")

# ═══════════════════════════════════════════════════════════════════
# SECTION PRINCIPALE — RECOMMANDATIONS
# ═══════════════════════════════════════════════════════════════════

st.header("🛍️ Obtenir des recommandations de produits")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Entrée")

    customer_id = st.text_input(
        label="Identifiant client (Customer ID)",
        value="17850",
        placeholder="Ex. 17850",
        help="L'identifiant du client pour lequel générer des recommandations"
    )

    purchased_items_input = st.text_area(
        label="Articles déjà achetés (codes StockCode séparés par des virgules)",
        value="85123A,71053",
        placeholder="Ex. 85123A, 71053, 84406B",
        height=100,
        help="Liste des produits que ce client a déjà achetés"
    )

    num_recommendations = st.slider(
        label="Nombre de recommandations",
        min_value=1,
        max_value=10,
        value=5,
        help="Combien de produits recommander"
    )

with col2:
    st.subheader("Clients exemples")
    st.info("""
    **Essayez ces identifiants clients :**
    - **17850** : Achète des articles de décoration lumineuse
    - **15168** : Achète des articles de décoration intérieure
    - **12792** : Achète des fournitures de fête

    **Codes produits exemples :**
    - `85123A` — Ensemble de thé blanc en métal
    - `71053`  — Sac fourre-tout blanc
    - `84406B` — Ensemble de décoration florale

    **Conseil :** Laissez les articles vides pour tester le mode cold-start.
    """)

# ═══════════════════════════════════════════════════════════════════
# BOUTON DE RECOMMANDATION
# ═══════════════════════════════════════════════════════════════════

if st.button("🚀 Générer les recommandations", use_container_width=True):

    with st.spinner("🔄 Analyse des patterns et génération des recommandations..."):
        try:
            # Parse les articles saisis (nettoie les espaces, met en majuscules)
            purchased_items = [
                item.strip().upper()
                for item in purchased_items_input.split(",")
                if item.strip()
            ]

            # Appel au moteur de recommandation
            recommendations, strategy = engine.recommend(
                user_id=customer_id,
                purchased_items=purchased_items,
                top_n=num_recommendations
            )

            st.success("✅ Recommandations générées avec succès !")
            st.markdown("---")

            # Affiche la stratégie utilisée
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("📋 Recommandations")
            with col2:
                strategy_label = {
                    'ensemble_autoencoder_ncf_lstm': '🔀 Ensemble (AE + NCF + LSTM)',
                    'single_model':                  '🔷 Modèle unique',
                    'popular_fallback':               '⭐ Articles populaires (cold-start)',
                }.get(strategy, strategy)
                st.write(f"**Stratégie :** {strategy_label}")

            # Affiche chaque recommandation
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

                    with col1:
                        st.write(f"**{i}. {rec['description'][:70]}**")

                    with col2:
                        score_value = rec['score']
                        if score_value > 0.30:
                            st.markdown(f"<p class='score-high'>{score_value:.3f}</p>",
                                        unsafe_allow_html=True)
                        elif score_value > 0.15:
                            st.markdown(f"<p class='score-medium'>{score_value:.3f}</p>",
                                        unsafe_allow_html=True)
                        else:
                            st.markdown(f"<p class='score-low'>{score_value:.3f}</p>",
                                        unsafe_allow_html=True)

                    with col3:
                        st.progress(min(float(score_value), 1.0))

                    with col4:
                        st.caption(f"ID : {rec['product_id']}")

                    st.divider()
            else:
                st.warning("Aucune recommandation trouvée pour ce client.")

        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")
            st.info("💡 Vérifiez que l'identifiant client et les codes produits sont corrects.")

# ═══════════════════════════════════════════════════════════════════
# SECTION PERFORMANCES DES MODÈLES
# ═══════════════════════════════════════════════════════════════════

st.markdown("---")
st.header("📊 Comparaison des performances des modèles")
st.caption("Résultats de la dernière évaluation sur 869 utilisateurs de test (20 % du jeu de données)")

# Métriques principales (Précision@5 — dernière évaluation)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🏆 LSTM",
        value="30.61 %",
        delta="Meilleur modèle",
        help="Précision@5 — Le LSTM capture les patterns séquentiels d'achat"
    )

with col2:
    st.metric(
        label="🔀 Ensemble",
        value="29.28 %",
        delta="-1.33 %",
        help="Précision@5 — Moyenne pondérée AE(10%) + NCF(35%) + LSTM(55%)"
    )

with col3:
    st.metric(
        label="🧠 NCF",
        value="3.84 %",
        delta="Cold-start users",
        help="Précision@5 — NCF souffre du problème cold-start pour les 20% d'utilisateurs de test"
    )

with col4:
    st.metric(
        label="📦 AutoEncodeur",
        value="6.26 %",
        delta="+3.55 % (ancienne : 2.71 %)",
        help="Précision@5 — Amélioré grâce au goulot 256D et pos_weight=15"
    )

# Tableau comparatif complet
st.markdown("**Tableau complet des métriques (Précision@K, Rappel@K, NDCG@K)**")

# Charge les résultats depuis le CSV
results_path = os.path.join(SAVED_MODELS_DIR, 'evaluation_results.csv')
if os.path.exists(results_path):
    df_results = pd.read_csv(results_path)
    # Renomme les colonnes en français
    df_results.columns = ['Modèle', 'K', 'Précision@K', 'Rappel@K', 'NDCG@K']
    st.dataframe(df_results, use_container_width=True, hide_index=True)
else:
    st.info("Lancez `python train_pipeline.py` pour générer les résultats d'évaluation.")

# Graphique en barres — Précision@5
st.markdown("**Précision@5 par modèle**")
perf_data = {
    "LSTM":         30.61,
    "Ensemble":     29.28,
    "AutoEncodeur": 6.26,
    "NCF":          3.84,
}

col1, col2 = st.columns([2, 1])

with col1:
    st.bar_chart(
        data=perf_data,
        height=380,
        use_container_width=True
    )

with col2:
    st.write("**Légende des métriques :**")
    st.info("""
    **Précision@K :**
    Sur les K premières recommandations,
    combien correspondent aux achats réels.

    **30 % = 3 sur 10 recommandations
    sont pertinentes — c'est excellent !**

    - Aléatoire : ~0.14 %
    - Votre système : 30.61 %
    - **220x mieux que le hasard**
    """)

# ═══════════════════════════════════════════════════════════════════
# SECTION AMÉLIORATIONS APPORTÉES
# ═══════════════════════════════════════════════════════════════════

st.markdown("---")
st.header("🔧 Améliorations apportées aux modèles")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **📦 AutoEncodeur**

    | Paramètre | Avant | Après |
    |---|---|---|
    | pos_weight | 40 | **15** |
    | Bruit gaussien | 0.3 | **0.15** |
    | Goulot d'étranglement | 128D | **256D** |
    | Époques max | 80 | **100** |
    | Patience arrêt anticipé | 10 | **15** |

    *Résultat : 2.71 % → **6.26 %** (+131 %)*
    """)

with col2:
    st.markdown("""
    **🧠 NCF**

    | Paramètre | Avant | Après |
    |---|---|---|
    | neg_ratio | 2 | **5** |
    | Époques max | 25 | **50** |
    | Patience arrêt anticipé | 5 | **8** |
    | MLP | 256→128→64 | **512→256→128→64** |
    | BatchNorm | Non | **Oui** |

    *Échantillonnage négatif pondéré par popularité*
    """)

with col3:
    st.markdown("""
    **🔀 Pipeline & Évaluation**

    | Paramètre | Avant | Après |
    |---|---|---|
    | Découpage | 80/10/10 | **70/10/20** |
    | Utilisateurs test | 435 | **869** |
    | LSTM_UNITS | 128 | **256** |
    | NCF entraîné sur | 80 % | **80 % (train+val)** |
    | Jeu de données | Online_Retail | **training.xlsx** |

    *+99 % d'utilisateurs évalués*
    """)

# ═══════════════════════════════════════════════════════════════════
# SECTION ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════

st.markdown("---")
st.header("🔍 Comment ça fonctionne ?")

with st.expander("📚 Architecture du système"):
    st.markdown("""
    **Pipeline de recommandation :**
    1. **Entrée** : Identifiant client + historique d'achats
    2. **Extraction de features** : Conversion en vecteurs numériques
    3. **Prédictions des modèles** :
       - **AutoEncodeur** : Reconstruit le vecteur d'achats → profil de goûts général
       - **NCF**          : Embeddings utilisateur × article → interactions collaboratives
       - **LSTM**         : Fenêtre glissante sur séquence → patterns temporels
    4. **Fusion ensemble** : Moyenne pondérée des scores normalisés (AE×10% + NCF×35% + LSTM×55%)
    5. **Classement & filtrage** : Top N produits non déjà achetés
    6. **Sortie** : Liste de produits avec scores de confiance
    """)

with st.expander("🧠 Détails des modèles (architecture mise à jour)"):
    st.markdown("""
    **AutoEncodeur — 6.26 % de précision@5**
    - Architecture : 3665 → 512 → 256 → **256 (goulot)** → 256 → 512 → 3665
    - Perte : BCE pondérée (pos_weight=**15**, réduit de 40)
    - Bruit d'entrée : GaussianNoise(**0.15**, réduit de 0.3)
    - Rôle : Capture les préférences générales via reconstruction

    **NCF — Neural Collaborative Filtering — 3.84 % de précision@5**
    - Architecture : GMF path + MLP path (512 → 256 → 128 → 64 avec BatchNorm)
    - Embeddings : 128 dimensions (utilisateur + article)
    - Entraînement : 5 négatifs pour chaque positif (pondérés par popularité)
    - Limite : Souffre du cold-start pour les 20 % d'utilisateurs de test

    **LSTM — 30.61 % de précision@5 — Meilleur modèle**
    - Architecture : Embedding(64D) → LSTM(**256**) → LSTM(128) → Dense(128) → Dense(3665, softmax)
    - Entraîné à prédire le prochain article dans la séquence d'achats
    - Force : Capture les dépendances temporelles (ce qu'un client achète ENSUITE)

    **Ensemble — 29.28 % de précision@5**
    - Méthode : AE×0.10 + NCF×0.35 + LSTM×0.55 (scores normalisés min-max)
    - Avantage : Robuste aux faiblesses individuelles de chaque modèle
    """)

with st.expander("📈 Interprétation des métriques"):
    st.markdown("""
    **Précision@K** : Sur les K premières recommandations, quel % correspond aux achats réels

    **Rappel@K** : Sur tous les achats réels du client, quel % apparaît dans le top K

    **NDCG@K** : Qualité du classement — pénalise les bonnes recommandations mal positionnées

    **Comparaison avec les baselines :**
    | Méthode | Précision@5 | Amélioration |
    |---|---|---|
    | Aléatoire | ~0.14 % | 1x (référence) |
    | Articles populaires | ~1-2 % | ~10x |
    | AutoEncodeur | 6.26 % | ~45x |
    | NCF | 3.84 % | ~27x |
    | LSTM | **30.61 %** | **~220x** |
    | Ensemble | 29.28 % | ~210x |
    """)

# ═══════════════════════════════════════════════════════════════════
# PIED DE PAGE
# ═══════════════════════════════════════════════════════════════════

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **📊 Qualité des données**
    - 541 909 transactions brutes
    - 397 016 lignes nettoyées (73 %)
    - Jeu de données : training.xlsx
    """)

with col2:
    st.markdown("""
    **🚀 Performances système**
    - Temps de réponse : ~150 ms
    - Précision LSTM@5 : 30.61 %
    - Précision Ensemble@5 : 29.28 %
    - Utilisateurs évalués : 869
    """)

with col3:
    st.markdown("""
    **👥 Équipe**
    - Nada Zirari
    - Ziyad Belahmar
    - Karim Bekkali
    - Youssef El Azami
    """)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "<p>Système de Recommandation de Produits Intelligent | Projet Module IA 2024-2025</p>"
    "</div>",
    unsafe_allow_html=True
)
