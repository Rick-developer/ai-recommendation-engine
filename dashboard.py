import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="AI Recommendation Engine Dashboard", layout="wide")

BASE_DIR = os.path.dirname(__file__)
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

@st.cache_data
def load_data():
    results_path = os.path.join(REPORTS_DIR, 'evaluation_results.csv')
    recs_path = os.path.join(REPORTS_DIR, 'ai_recommendations.csv')
    
    if not os.path.exists(results_path) or not os.path.exists(recs_path):
        return None, None
        
    results_df = pd.read_csv(results_path)
    recs_df = pd.read_csv(recs_path)
    return results_df, recs_df

st.title("🛍️ AI Recommendation Engine Dashboard")
st.markdown("Interactive comparison of our Semantic AI model vs a simple Popularity Baseline.")

results_df, recs_df = load_data()

if results_df is None:
    st.error("No reports found! Please run the offline evaluation first.")
else:
    # 1. Headline Metrics
    st.header("1. Core Performance Metrics")
    
    summary = results_df.groupby('model_type')['hit'].mean() * 100
    precision = summary / 5.0
    
    base_hit = summary.get('baseline', 0)
    ai_hit = summary.get('ai_model', 0)
    
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Baseline Hit Rate@5", f"{base_hit:.2f}%")
    col2.metric("AI Model Hit Rate@5", f"{ai_hit:.2f}%", f"+{(ai_hit-base_hit):.2f}% vs Baseline", delta_color="normal")
    col3.metric("AI Model Precision@5", f"{precision.get('ai_model', 0):.2f}%")
    
    st.divider()
    
    # 2. Charts
    colA, colB = st.columns(2)
    
    with colA:
        st.subheader("Category Distribution")
        cat_counts = recs_df['category'].value_counts()
        st.bar_chart(cat_counts, color="#673AB7")
        
    with colB:
        st.subheader("Hit Rate Trend Line")
        window = st.slider("Smoothing Window Size (Moving Average)", min_value=10, max_value=200, value=50, step=10)
        
        baseline = results_df[results_df['model_type'] == 'baseline']['hit'].rolling(window=window).mean() * 100
        ai_model = results_df[results_df['model_type'] == 'ai_model']['hit'].rolling(window=window).mean() * 100
        
        trend_df = pd.DataFrame({
            "AI Model": ai_model.values,
            "Baseline": baseline.values
        })
        st.line_chart(trend_df.dropna(), color=["#FF6F00", "#9E9E9E"])
    
    st.divider()
    
    # 3. Explore Raw Data
    st.header("Explore Raw Recommendations")
    user_id = st.selectbox("Select User ID to inspect:", results_df['user_id'].unique())
    
    user_res = results_df[results_df['user_id'] == user_id]
    st.write("Evaluation Result:")
    st.dataframe(user_res, use_container_width=True, hide_index=True)
    
    st.write(f"Top 5 Recommendations for {user_id}:")
    user_recs = recs_df[recs_df['user_id'] == user_id]
    st.dataframe(user_recs, use_container_width=True, hide_index=True)
