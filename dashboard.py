import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Set page configuration
st.set_page_config(
    page_title="Model Performance Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a premium look
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Headers */
    h1 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    h2, h3 {
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    
    /* Metric Cards */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 1.5rem !important;
        border-radius: 0.75rem !important;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1) !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    /* Metric Text */
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
    }
    [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-size: 1.875rem !important;
        font-weight: 700 !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    
    /* General text */
    .stMarkdown, p, span {
        color: #334155;
    }
    </style>
    """, unsafe_allow_html=True)

# App Title
st.title("📊 Image Classification Model Comparison")

# Dataset Information Section
with st.expander("ℹ️ About the Dataset & Accuracy Settings", expanded=True):
    st.markdown("""
    ### Why is the accuracy low?
    The project is currently using **Synthetic Data** because the `data/` directory was empty during the last run. 
    
    **Dataset Properties:**
    - **Type**: Synthetic Colored Noise
    - **Classes**: 3 (Red-tinted, Green-tinted, Blue-tinted)
    - **Sample Count**: 200 total images (split 80/20 for train/test)
    - **Resolution**: 64x64 pixels (RGB)
    
    **Note**: Deep Learning models like ResNet and CNN typically require thousands of real-world images to achieve high accuracy. On random synthetic noise, these models struggle to find meaningful patterns, leading to the observed accuracy (often near random chance for 3 classes).
    
    ### To Improve Accuracy:
    1. Add real image folders to the `data/` directory (e.g., `data/cats/`, `data/dogs/`).
    2. Re-run the model notebooks to update the performance metrics.
    """)

st.markdown("---")

# Data Loading
RESULTS_FILE = "results/model_comparison.csv"

if not os.path.exists(RESULTS_FILE):
    st.error(f"Results file not found at {RESULTS_FILE}. Please run the models first.")
    st.stop()

df = pd.read_csv(RESULTS_FILE)

# Sidebar
st.sidebar.header("Controls & Filters")
model_types = st.sidebar.multiselect(
    "Select Model Types",
    options=df["Type"].unique(),
    default=df["Type"].unique()
)

filtered_df = df[df["Type"].isin(model_types)]

# Key Metrics Overview
st.subheader("🚀 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

best_acc_row = df.loc[df['Accuracy'].idxmax()]
fastest_inf_row = df.loc[df['Inference Time (s/sample)'].idxmin()]

with col1:
    st.metric("Top Accuracy", f"{best_acc_row['Accuracy']:.2%}", delta=best_acc_row['Model'])
with col2:
    st.metric("Fastest Inference", f"{fastest_inf_row['Inference Time (s/sample)']*1000:.2f}ms", delta=fastest_inf_row['Model'], delta_color="inverse")
with col3:
    st.metric("Total Models", len(df))
with col4:
    st.metric("Average Accuracy", f"{df['Accuracy'].mean():.2%}")

st.markdown("---")

# Visualizations
c1, c2 = st.columns(2)

with c1:
    st.subheader("🎯 Accuracy Comparison")
    fig_acc = px.bar(
        filtered_df,
        x="Model",
        y="Accuracy",
        color="Type",
        text_auto='.2%',
        title="Model Accuracy (Higher is better)",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig_acc.update_layout(
        showlegend=True, 
        xaxis_title="Model", 
        yaxis_title="Accuracy",
        yaxis_range=[0, 1],
        legend=dict(
            title_font=dict(color="#000000"),
            font=dict(color="#000000")
        ),
        font=dict(color="#000000"), # Global chart font
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    fig_acc.update_xaxes(
        tickfont=dict(color="#000000"),
        title_font=dict(color="#000000")
    )
    fig_acc.update_yaxes(
        tickfont=dict(color="#000000"),
        title_font=dict(color="#000000")
    )
    fig_acc.update_traces(textfont_color="#000000", textposition="outside")
    st.plotly_chart(fig_acc, use_container_width=True)

with c2:
    st.subheader("⏱️ Training Time")
    fig_train = px.bar(
        filtered_df,
        x="Model",
        y="Training Time (s)",
        color="Type",
        title="Total Training Duration (Seconds)",
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig_train.update_layout(
        showlegend=True, 
        xaxis_title="Model",
        yaxis_title="Seconds",
        legend=dict(
            title_font=dict(color="#000000"),
            font=dict(color="#000000")
        ),
        font=dict(color="#000000"), # Global chart font
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    fig_train.update_xaxes(
        tickfont=dict(color="#000000"),
        title_font=dict(color="#000000")
    )
    fig_train.update_yaxes(
        tickfont=dict(color="#000000"),
        title_font=dict(color="#000000")
    )
    fig_train.update_traces(textfont_color="#000000")
    st.plotly_chart(fig_train, use_container_width=True)

c3, c4 = st.columns(2)

with c3:
    st.subheader("⚡ Inference Speed")
    fig_inf = px.scatter(
        filtered_df,
        x="Accuracy",
        y="Inference Time (s/sample)",
        size="Parameters" if filtered_df["Parameters"].sum() > 0 else None,
        color="Model",
        hover_name="Model",
        title="Accuracy vs. Latency (Bubble size = Parameters)",
        labels={"Inference Time (s/sample)": "Latency (s)"}
    )
    fig_inf.update_layout(
        font=dict(color="#000000"), # Global chart font
        legend=dict(
            title_font=dict(color="#000000"),
            font=dict(color="#000000")
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    fig_inf.update_xaxes(
        tickfont=dict(color="#000000"),
        title_font=dict(color="#000000")
    )
    fig_inf.update_yaxes(
        tickfont=dict(color="#000000"),
        title_font=dict(color="#000000")
    )
    st.plotly_chart(fig_inf, use_container_width=True)

with c4:
    st.subheader("📂 Raw Metrics Table")
    st.dataframe(filtered_df.sort_values(by="Accuracy", ascending=False), use_container_width=True)

# Footer
st.markdown("---")
st.caption("Developed by Antigravity AI • Data source: results/model_comparison.csv")
