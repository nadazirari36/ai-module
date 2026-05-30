"""
STREAMLIT APP FOR RECOMMENDATION SYSTEM
File: streamlit_app.py

Installation:
  pip install streamlit

Run:
  streamlit run streamlit_app.py

Then opens in browser: http://localhost:8501
"""

import streamlit as st
import pickle
import numpy as np
import json
import sys
import os

# Ensure the module directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import RecommendationEngine

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Recommendation System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════
# STYLING
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .recommendation-item {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 4px solid #1f77b4;
    }
    .score-high {
        color: #31a354;
        font-weight: bold;
    }
    .score-medium {
        color: #ff7f0e;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# LOAD MODELS (ONCE)
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource
def load_engine():
    """Load recommendation engine with all models"""
    eng = RecommendationEngine()
    eng.load()
    return eng

engine = load_engine()

# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════

st.title("🎯 Intelligent Product Recommendation System")
st.markdown("**Ensemble Deep Learning (AutoEncoder + NCF + LSTM)**")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR - SYSTEM INFO
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("📊 System Status")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**AutoEncoder**")
        st.success("Loaded")
    with col2:
        st.write("**NCF**")
        st.success("Loaded")

    st.write("**LSTM**")
    st.success("Loaded")

    st.markdown("---")

    st.header("📈 Dataset Stats")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Customers", "4,339")
    with col2:
        st.metric("Products", "3,665")
    with col3:
        st.metric("Transactions", "397,016")

    st.markdown("---")

    st.header("🏆 Best Performance")
    st.metric("LSTM Precision@5",     "29.57%", "Best model")
    st.metric("Ensemble Precision@5", "29.25%", "Nearly identical")

# ═══════════════════════════════════════════════════════════════════
# MAIN CONTENT - RECOMMENDATION SECTION
# ═══════════════════════════════════════════════════════════════════

st.header("🛍️ Get Product Recommendations")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")

    customer_id = st.text_input(
        label="Customer ID",
        value="17850",
        placeholder="Enter customer ID",
        help="The ID of the customer to get recommendations for"
    )

    purchased_items_input = st.text_area(
        label="Previously Purchased Items",
        value="85123A,71053",
        placeholder="Enter product IDs separated by commas",
        height=100,
        help="List of products this customer has already bought"
    )

    num_recommendations = st.slider(
        label="Number of Recommendations",
        min_value=1,
        max_value=10,
        value=5,
        help="How many products to recommend"
    )

with col2:
    st.subheader("Dataset Overview")

    st.info("""
    **📌 Example Customers:**
    - Customer 17850: Bought lighting items
    - Customer 15168: Bought home décor
    - Customer 12792: Bought party supplies

    **💡 Try these examples to see how the system works!**
    """)

# ═══════════════════════════════════════════════════════════════════
# GET RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════

if st.button("🚀 Get Recommendations", use_container_width=True):

    with st.spinner("🔄 Analyzing patterns and generating recommendations..."):
        try:
            # Parse input
            purchased_items = [
                item.strip().upper()
                for item in purchased_items_input.split(",")
                if item.strip()
            ]

            # Get recommendations
            recommendations, strategy = engine.recommend(
                user_id=customer_id,
                purchased_items=purchased_items,
                top_n=num_recommendations
            )

            # Display results
            st.success("Recommendations generated successfully!")

            st.markdown("---")

            # Strategy info
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("📋 Recommendations")
            with col2:
                st.write(f"**Strategy:** {strategy}")

            # Display each recommendation
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    with st.container():
                        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

                        with col1:
                            st.write(f"**{i}. {rec['description'][:60]}**")
                        with col2:
                            score_value = rec['score']
                            if score_value > 0.30:
                                st.markdown(f"<p class='score-high'>{score_value:.3f}</p>",
                                            unsafe_allow_html=True)
                            else:
                                st.markdown(f"<p class='score-medium'>{score_value:.3f}</p>",
                                            unsafe_allow_html=True)
                        with col3:
                            st.progress(min(score_value, 1.0))
                        with col4:
                            st.caption(f"ID: {rec['product_id']}")

                        st.divider()
            else:
                st.warning("No recommendations found for this customer.")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Make sure the customer ID and product IDs are correct.")

# ═══════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE SECTION
# ═══════════════════════════════════════════════════════════════════

st.markdown("---")
st.header("📊 Model Performance Comparison")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="LSTM",
        value="29.57%",
        delta="Best Model",
        help="Precision@5 - LSTM captures temporal patterns"
    )

with col2:
    st.metric(
        label="Ensemble",
        value="29.25%",
        delta="-0.32%",
        help="Precision@5 - Weighted average AE+NCF+LSTM"
    )

with col3:
    st.metric(
        label="NCF",
        value="8.81%",
        delta="+4.97% improved",
        help="Precision@5 - User-from-Items cold-start inference"
    )

with col4:
    st.metric(
        label="AutoEncoder",
        value="3.06%",
        delta="Sparse data",
        help="Precision@5 - Denoising AutoEncoder"
    )

# Performance chart
st.markdown("**Precision@5 Comparison**")
performance_data = {
    "LSTM":        29.57,
    "Ensemble":    29.25,
    "NCF":          8.81,
    "AutoEncoder":  3.06,
}

col1, col2 = st.columns([2, 1])

with col1:
    st.bar_chart(
        data=performance_data,
        height=400,
        use_container_width=True
    )

with col2:
    st.write("**Metrics Legend:**")
    st.info("""
    - **Precision@5**: Out of top 5 recommendations, how many match customer interest
    - **Higher is better**: 29% means 3 out of 10 recommendations match
    - **Random baseline**: ~0.14% (guessing)
    - **Your system**: 29.57% (211x better!)
    """)

# ═══════════════════════════════════════════════════════════════════
# SYSTEM DETAILS
# ═══════════════════════════════════════════════════════════════════

st.markdown("---")
st.header("🔍 How It Works")

with st.expander("📚 System Architecture"):
    st.markdown("""
    **Pipeline:**
    1. **Data Input**: Customer ID + Purchase History
    2. **Feature Extraction**: Convert to vectors
    3. **Model Predictions**:
       - AutoEncoder: General preferences
       - NCF: User-item interactions
       - LSTM: Temporal patterns
    4. **Ensemble Fusion**: Weighted average of normalized scores (AE×10% + NCF×35% + LSTM×55%)
    5. **Ranking & Filtering**: Return top N products not already purchased
    6. **Output**: Product recommendations with confidence scores
    """)

with st.expander("🧠 Model Details"):
    st.markdown("""
    **AutoEncoder (6.26% precision)**
    - Architecture: 3665 → 512 → 256 → **256 (bottleneck)** → 256 → 512 → 3665
    - Improvements: pos_weight 40→**15**, noise 0.3→**0.15**, bottleneck 128→**256**
    - Role: Captures general user taste profile via reconstruction

    **NCF - Neural Collaborative Filtering (3.84% precision)**
    - Architecture: GMF path + MLP path (**512→256→128→64** with BatchNorm)
    - Embeddings: 128-dimensional user & item vectors
    - neg_ratio: 2→**5** with popularity-weighted hard negatives
    - Limit: Cold-start issue for held-out test users

    **LSTM - Long Short-Term Memory (30.61% precision)**
    - Architecture: Embedding(64) → LSTM(**256**) → LSTM(128) → Dense(3665, softmax)
    - Learns: Sequential purchase patterns (next-item prediction)
    - Strength: Temporal dependencies — what a customer buys NEXT

    **Ensemble (29.28% precision)**
    - Method: Weighted average of min-max normalized scores
    - Weights: AE×0.10 + NCF×0.35 + LSTM×0.55
    - Benefit: Robust to individual model weaknesses
    """)

with st.expander("📈 Evaluation Metrics"):
    st.markdown("""
    **Precision@5**: Out of top 5 recommendations, % that match customer interest

    **Recall@5**: Out of customer's actual purchases, % captured in top 5

    **NDCG@5**: Ranking quality (penalises good items ranked too low)

    **Latest results on 869 test users (20% split):**
    | Model | Precision@5 | Recall@5 | NDCG@5 |
    |---|---|---|---|
    | LSTM | **29.57%** | 5.95% | **31.71%** |
    | Ensemble | 29.25% | 5.69% | 31.68% |
    | NCF | 8.81% | 1.21% | 10.08% |
    | AutoEncoder | 3.06% | 0.49% | 3.49% |

    **Comparison to baselines:**
    - Random guessing: ~0.14% precision
    - Your system (LSTM): 29.57% precision (**211x better!**)
    """)

# ═══════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **📊 Data Quality**
    - 540,710 raw transactions
    - 397,016 clean rows (73%)
    - Dataset: training.xlsx
    """)

with col2:
    st.markdown("""
    **🚀 Performance**
    - Response time: ~150ms
    - Accuracy: 29.57% (LSTM)
    - Test users evaluated: 869
    """)

with col3:
    st.markdown("""
    **👥 Team**
    - Nada Zirari
    - Ziyad Belahmar
    - Karim Bekkali
    - Youssef El Azami
    """)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "<p>Intelligent Product Recommendation System | AI Module Project 2024</p>"
    "</div>",
    unsafe_allow_html=True
)
