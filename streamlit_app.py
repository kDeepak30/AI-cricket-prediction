import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="IPL Performance Predictor", layout="wide")

# Load models and data
runs_model = joblib.load("best_model_runs.pkl")
wkts_model = joblib.load("best_model_wkts.pkl")
data = pd.read_csv("final_dataset.csv")

st.title("🏏 IPL Player Performance Prediction Dashboard")

tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Model Metrics", "🔍 Feature Importance"])

# ------------------- TAB 1 -------------------
with tab1:
    st.header("Player Match Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        player = st.selectbox("Select Player", sorted(data["striker"].unique()))

    with col2:
        opponent = st.selectbox("Select Opponent", sorted(data["bowling_team"].unique()))

    with col3:
        venue = st.selectbox("Select Venue", sorted(data["venue"].unique()))

    player_data = data[
        (data["striker"] == player) &
        (data["bowling_team"] == opponent) &
        (data["venue"] == venue)
    ]

    if st.button("Predict Performance"):
        if len(player_data) > 0:
            X_input = player_data.drop(
                columns=["target_runs_next_match", "target_wickets_next_match"],
                errors="ignore"
            ).iloc[0:1]

            pred_runs = runs_model.predict(X_input)[0]
            pred_wkts = wkts_model.predict(X_input)[0]

            colA, colB = st.columns(2)
            colA.metric("Predicted Runs", round(pred_runs, 2))
            colB.metric("Predicted Wickets", round(pred_wkts, 2))
        else:
            st.warning("No historical data found for this selection.")

# ------------------- TAB 2 -------------------
with tab2:
    st.header("Model Performance Comparison")

    results = pd.DataFrame({
        "Model": ["Baseline", "Random Forest", "XGBoost", "LightGBM"],
        "RMSE": [19.61, 16.64, 16.34, 16.28],
        "MAE": [14.33, 11.46, 11.09, 10.92],
        "R2": [0.26, 0.47, 0.48, 0.49]
    })

    st.dataframe(results)
    st.bar_chart(results.set_index("Model")[["R2"]])


# ------------------- TAB 3 -------------------
with tab3:
    st.header("Feature Importance")

    try:
        features = runs_model.feature_names_in_
        importances = runs_model.feature_importances_

        imp_df = pd.DataFrame({
            "Feature": features,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        st.subheader("Top 10 Important Features")
        st.dataframe(imp_df.head(10))

        fig, ax = plt.subplots()
        ax.barh(imp_df["Feature"][:10], imp_df["Importance"][:10])
        ax.invert_yaxis()
        st.pyplot(fig)

    except Exception as e:
        st.write("Feature importance not available:", e)
