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
# ============================================================
# ===================== TAB 1: Prediction ====================
# ============================================================
with tab1:
    st.header("Player Match Prediction")

    if data.empty or runs_model is None or wkts_model is None:
        st.error("Models or dataset not loaded properly.")
    else:

        all_players = sorted(
            set(data["striker"].dropna().unique())
            .union(set(data["bowler"].dropna().unique()))
        )

        all_teams = sorted(data["batting_team"].dropna().unique())
        all_venues = sorted(data["venue"].dropna().unique())

        player = st.selectbox("Select Player", all_players)
        opponent = st.selectbox("Select Opponent Team", all_teams)
        venue = st.selectbox("Select Venue", all_venues)

        if st.button("Predict Performance"):

            # ================= CHECK IF BATSMAN =================
            batsman_data = data[
                (data["striker"] == player) &
                (data["bowling_team"] == opponent) &
                (data["venue"] == venue)
            ]

            # ================= CHECK IF BOWLER =================
            bowler_data = data[
                (data["bowler"] == player) &
                (data["batting_team"] == opponent) &
                (data["venue"] == venue)
            ]

            # ----------- RUNS PREDICTION -----------
            if not batsman_data.empty:

                required_runs_cols = runs_model.feature_names_in_

                X_runs = batsman_data.reindex(
                    columns=required_runs_cols,
                    fill_value=0
                ).iloc[0:1]

                X_runs = X_runs.apply(pd.to_numeric, errors="coerce").fillna(0)

                predicted_runs = runs_model.predict(X_runs)[0]

                st.success(f"🏏 Predicted Runs: {predicted_runs:.2f}")

            # ----------- WICKETS PREDICTION -----------
            if not bowler_data.empty:

                required_wkts_cols = wkts_model.feature_names_in_

                X_wkts = bowler_data.reindex(
                    columns=required_wkts_cols,
                    fill_value=0
                ).iloc[0:1]

                X_wkts = X_wkts.apply(pd.to_numeric, errors="coerce").fillna(0)

                if scaler_wkts is not None:
                    X_wkts = pd.DataFrame(
                        scaler_wkts.transform(X_wkts),
                        columns=required_wkts_cols
                    )

                predicted_wickets = wkts_model.predict(X_wkts)[0]

                st.success(f"🎯 Predicted Wickets: {predicted_wickets:.2f}")

            if batsman_data.empty and bowler_data.empty:
                st.warning("No historical data found for this player.")
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
