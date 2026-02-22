import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="IPL Performance Predictor", layout="wide")

# ================= LOAD MODELS =================
try:
    runs_model = joblib.load("best_model_runs.pkl")
    wkts_model = joblib.load("best_model_wkts.pkl")
    scaler_wkts = joblib.load("scaler_wkts.pkl")
    st.sidebar.success("Models loaded successfully")
except Exception as e:
    st.sidebar.error(f"Model loading error: {e}")
    runs_model, wkts_model, scaler_wkts = None, None, None

# ================= LOAD DATA =================
try:
    data = pd.read_csv("final_dataset.csv")
    st.sidebar.success("Dataset loaded successfully")
except Exception as e:
    st.sidebar.error(f"Dataset loading error: {e}")
    data = pd.DataFrame()

st.title("🏏 IPL Player Performance Prediction Dashboard")

tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Model Metrics", "🔍 Feature Importance"])
if st.button("Predict Performance"):

    # Check if player appears as batsman
    batsman_data = data[
        (data["striker"] == player) &
        (data["bowling_team"] == opponent) &
        (data["venue"] == venue)
    ]

    # Check if player appears as bowler
    bowler_data = data[
        (data["bowler"] == player) &
        (data["batting_team"] == opponent) &
        (data["venue"] == venue)
    ]

    prediction_made = False

    # ========================
    # BOWLER → WICKETS ONLY
    # ========================
    if not bowler_data.empty:

        required_wkts_cols = wkts_model.feature_names_in_

        X_wkts = bowler_data.reindex(
            columns=required_wkts_cols,
            fill_value=0
        ).iloc[0:1]

        X_wkts = X_wkts.apply(pd.to_numeric, errors="coerce").fillna(0)

        raw_pred = wkts_model.predict(X_wkts)[0]

        # Normalize + realistic limit
        predicted_wickets = raw_pred / 4
        predicted_wickets = max(0, min(predicted_wickets, 5))

        st.success(f"🎯 Predicted Wickets: {predicted_wickets:.2f}")
        prediction_made = True

    # ========================
    # BATSMAN → RUNS ONLY
    # ========================
    elif not batsman_data.empty:

        required_runs_cols = runs_model.feature_names_in_

        X_runs = batsman_data.reindex(
            columns=required_runs_cols,
            fill_value=0
        ).iloc[0:1]

        X_runs = X_runs.apply(pd.to_numeric, errors="coerce").fillna(0)

        predicted_runs = runs_model.predict(X_runs)[0]
        predicted_runs = max(0, predicted_runs)

        st.success(f"🏏 Predicted Runs: {predicted_runs:.2f}")
        prediction_made = True

    if not prediction_made:
        st.warning("No historical data found for this player under selected conditions.")
# ============================================================
# ===================== TAB 2: Metrics =======================
# ============================================================
with tab2:
    st.header("Model Performance Comparison")

    results = pd.DataFrame({
        "Model": ["LightGBM (Runs)", "Poisson (Wickets)"],
        "RMSE": [23.68, 1.006],
        "MAE": [17.25, 0.814],
        "R2": [-0.13, -0.017]
    })

    st.dataframe(results)

    fig, ax = plt.subplots()
    ax.bar(results["Model"], results["R2"])
    ax.set_ylabel("R2 Score")
    ax.set_title("Model R2 Comparison")
    st.pyplot(fig)

# ============================================================
# ===================== TAB 3: Feature Importance ============
# ============================================================
with tab3:
    st.header("Feature Importance")

    if runs_model is not None:
        features_runs = runs_model.feature_names_in_
        importances_runs = runs_model.feature_importances_

        imp_df_runs = pd.DataFrame({
            "Feature": features_runs,
            "Importance": importances_runs
        }).sort_values(by="Importance", ascending=False)

        st.subheader("Runs Model Top 10 Features")
        st.dataframe(imp_df_runs.head(10))

        fig, ax = plt.subplots()
        ax.barh(imp_df_runs["Feature"][:10],
                imp_df_runs["Importance"][:10])
        ax.invert_yaxis()
        st.pyplot(fig)

    if wkts_model is not None:
        features_wkts = wkts_model.feature_names_in_

        if hasattr(wkts_model, "coef_"):
            importances_wkts = np.abs(wkts_model.coef_)
        else:
            importances_wkts = np.zeros(len(features_wkts))

        imp_df_wkts = pd.DataFrame({
            "Feature": features_wkts,
            "Importance": importances_wkts
        }).sort_values(by="Importance", ascending=False)

        st.subheader("Wickets Model Top 10 Features")
        st.dataframe(imp_df_wkts.head(10))

        fig, ax = plt.subplots()
        ax.barh(imp_df_wkts["Feature"][:10],
                imp_df_wkts["Importance"][:10])
        ax.invert_yaxis()
        st.pyplot(fig)

st.markdown("---")
st.markdown("Developed for IPL Analytics Submission")
