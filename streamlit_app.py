import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler # Import StandardScaler

st.set_page_config(page_title="IPL Performance Predictor", layout="wide")

# Paths for models and encoders
MODELS_DIR = "models"
ENCODERS_DIR = "label_encoders"

# Load models
try:
    runs_model = joblib.load(os.path.join(MODELS_DIR, "best_model_runs.pkl"))
    wkts_model = joblib.load(os.path.join(MODELS_DIR, "best_model_wkts.pkl"))
    scaler_wkts = joblib.load(os.path.join(MODELS_DIR, "scaler_wkts.pkl")) # Load the scaler
    st.sidebar.success("Models and Scaler loaded successfully!")
except FileNotFoundError as e:
    st.sidebar.error(f"Error loading models or scaler: {e}. Make sure all required files are in the '{MODELS_DIR}' directory.")
    runs_model, wkts_model, scaler_wkts = None, None, None

# Load LabelEncoders
loaded_encoders = {}
try:
    for filename in os.listdir(ENCODERS_DIR):
        if filename.startswith('label_encoder_') and filename.endswith('.pkl'):
            col_name = filename.replace('label_encoder_', '').replace('.pkl', '')
            loaded_encoders[col_name] = joblib.load(os.path.join(ENCODERS_DIR, filename))
    st.sidebar.success("Label encoders loaded successfully!")
except FileNotFoundError as e:
    st.sidebar.error(f"Error loading encoders: {e}. Make sure encoders are in the '{ENCODERS_DIR}' directory.")

# Load data for player/team/venue options
try:
    data = pd.read_csv("final_dataset.csv")
    st.sidebar.success("Dataset loaded successfully!")
except FileNotFoundError as e:
    st.sidebar.error(f"Error loading data: {e}. Make sure 'final_dataset.csv' is in the current directory.")
    data = pd.DataFrame() # Create empty DataFrame to prevent errors


st.title("🏏 IPL Player Performance Prediction Dashboard")
st.markdown("Predict the performance of batsmen and bowlers in upcoming matches based on historical data.")

tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Model Metrics", "🔍 Feature Importance"])

# ------------------- TAB 1: Prediction -------------------
with tab1:
    st.header("Player Match Prediction")
    st.markdown("Select a batsman, bowler, opponent team, and venue to get predictions.")

    if data.empty or runs_model is None or wkts_model is None or not loaded_encoders or scaler_wkts is None:
        st.error("Cannot proceed with predictions. Please ensure models, encoders, scaler, and data are loaded correctly.")
    else:
        # Get unique values for dropdowns
        all_strikers = sorted(data["striker"].unique())
        all_bowlers = sorted(data["bowler"].unique())
        all_teams = sorted(data["striker_team"].unique()) # Assuming striker_team contains all team names
        all_venues = sorted(data["venue"].unique())

        col1, col2 = st.columns(2)

        with col1:
            selected_batsman = st.selectbox("Select Batsman", all_strikers)
            selected_batsman_team = st.selectbox("Select Batsman's Team", all_teams)
            selected_venue = st.selectbox("Select Venue", all_venues)

        with col2:
            selected_bowler = st.selectbox("Select Bowler", all_bowlers)
            selected_bowler_team = st.selectbox("Select Bowler's Team", all_teams) # Assuming bowler's team can be any of the teams
            selected_opponent_team = st.selectbox("Select Opponent Team (for both)", all_teams)


        # Create a generalized input feature vector
        # This requires aggregating historical data to create representative features
        # For simplicity, we'll average relevant features from historical matches for the selected player/bowler
        # In a real-world app, this would involve more sophisticated logic to get 'most recent' or 'average' form.

        if st.button("Predict Performance"):
            # Prepare input for Batsman Runs Prediction
            batsman_historical_data = data[
                (data['striker'] == selected_batsman) &
                (data['striker_team'] == selected_batsman_team) &
                (data['striker_opponent_team'] == selected_opponent_team) &
                (data['venue'] == selected_venue)
            ]

            if not batsman_historical_data.empty:
                # Use the mean of numerical features and mode of categorical features for prediction
                X_input_batsman = batsman_historical_data.drop(
                    columns=[col for col in data.columns if 'target' in col or 'match_id' == col],
                    errors='ignore'
                ).mean(numeric_only=True).to_frame().T # Get mean of numeric features

                # Ensure all columns expected by the model are present and in order
                model_features_runs = runs_model.feature_names_in_ if hasattr(runs_model, 'feature_names_in_') else list(data.drop(columns=[col for col in data.columns if 'target' in col or 'match_id' == col], errors='ignore').columns)
                X_input_batsman_final = pd.DataFrame(columns=model_features_runs)

                # Populate the input row with selected and averaged values
                X_input_batsman_final['striker'] = selected_batsman
                X_input_batsman_final['striker_team'] = selected_batsman_team
                X_input_batsman_final['striker_opponent_team'] = selected_opponent_team
                X_input_batsman_final['venue'] = selected_venue
                X_input_batsman_final['season'] = data['season'].max() # Assume latest season for prediction

                # Fill other numerical features with their mean from historical data
                for col in model_features_runs:
                    if col not in ['striker', 'striker_team', 'striker_opponent_team', 'venue', 'season']:
                        if col in X_input_batsman.columns:
                            X_input_batsman_final[col] = X_input_batsman[col].values[0]
                        else:
                            # Fallback: use mean from entire dataset if not in historical data
                            X_input_batsman_final[col] = data[col].mean()

                # Apply Label Encoding for batsman prediction
                categorical_cols_runs_app = [col for col in X_input_batsman_final.columns if X_input_batsman_final[col].dtype == 'object'] # Identify object columns in X_input_batsman_final
                for col in categorical_cols_runs_app:
                    if col in loaded_encoders:
                        try:
                            # Use .loc for safe assignment with single row DataFrame
                            X_input_batsman_final.loc[0, col] = loaded_encoders[col].transform([X_input_batsman_final.loc[0, col]])[0]
                        except ValueError:
                            # Handle unseen labels by assigning a default value (e.g., 0)
                            st.warning(f"Unseen label '{X_input_batsman_final.loc[0, col]}' for '{col}' in batsman prediction. Assigning 0.")
                            X_input_batsman_final.loc[0, col] = 0 # Or a better default
                    else:
                        X_input_batsman_final.loc[0, col] = 0 # Default if encoder missing for some reason

                # Ensure all numeric columns are numeric (after potential defaults for categorical)
                for col in model_features_runs:
                    if col not in categorical_cols_runs_app:
                        X_input_batsman_final[col] = pd.to_numeric(X_input_batsman_final[col], errors='coerce').fillna(0)

                # Predict runs
                predicted_runs = runs_model.predict(X_input_batsman_final)[0]
                st.metric(f"Predicted Runs for {selected_batsman}", f"{predicted_runs:.2f}")

            else:
                st.warning(f"No historical data found for {selected_batsman} under these conditions for runs prediction.")


            # Prepare input for Bowler Wickets Prediction
            bowler_historical_data = data[
                (data['bowler'] == selected_bowler) &
                (data['bowler_team'] == selected_bowler_team) &
                (data['bowler_opponent_team'] == selected_opponent_team) &
                (data['venue'] == selected_venue)
            ]

            if not bowler_historical_data.empty:
                X_input_bowler = bowler_historical_data.drop(
                    columns=[col for col in data.columns if 'target' in col or 'match_id' == col],
                    errors='ignore'
                ).mean(numeric_only=True).to_frame().T

                model_features_wkts = list(X_train_wkts.columns) # Use the exact columns the model was trained on

                X_input_bowler_final = pd.DataFrame(columns=model_features_wkts)

                X_input_bowler_final['bowler'] = selected_bowler
                X_input_bowler_final['bowler_team'] = selected_bowler_team
                X_input_bowler_final['bowler_opponent_team'] = selected_opponent_team
                X_input_bowler_final['venue'] = selected_venue
                X_input_bowler_final['season'] = data['season'].max()

                for col in model_features_wkts:
                    if col not in ['bowler', 'bowler_team', 'bowler_opponent_team', 'venue', 'season']:
                        if col in X_input_bowler.columns:
                            X_input_bowler_final[col] = X_input_bowler[col].values[0]
                        else:
                            X_input_bowler_final[col] = data[col].mean()

                # Apply Label Encoding for bowler prediction
                categorical_cols_wkts_app = [col for col in X_input_bowler_final.columns if X_input_bowler_final[col].dtype == 'object']
                for col in categorical_cols_wkts_app:
                    if col in loaded_encoders:
                        try:
                            X_input_bowler_final.loc[0, col] = loaded_encoders[col].transform([X_input_bowler_final.loc[0, col]])[0]
                        except ValueError:
                            st.warning(f"Unseen label '{X_input_bowler_final.loc[0, col]}' for '{col}' in bowler prediction. Assigning 0.")
                            X_input_bowler_final.loc[0, col] = 0
                    else:
                        X_input_bowler_final.loc[0, col] = 0

                for col in model_features_wkts:
                    if col not in categorical_cols_wkts_app:
                        X_input_bowler_final[col] = pd.to_numeric(X_input_bowler_final[col], errors='coerce').fillna(0)

                # Scale numerical features for PoissonRegressor
                numerical_cols_wkts_app = X_input_bowler_final.select_dtypes(include=np.number).columns
                X_input_bowler_final_scaled = X_input_bowler_final.copy()
                X_input_bowler_final_scaled[numerical_cols_wkts_app] = scaler_wkts.transform(X_input_bowler_final[numerical_cols_wkts_app])

                # Predict wickets
                predicted_wickets = wkts_model.predict(X_input_bowler_final_scaled)[0]
                st.metric(f"Predicted Wickets for {selected_bowler}", f"{predicted_wickets:.2f}")
            else:
                st.warning(f"No historical data found for {selected_bowler} under these conditions for wickets prediction.")


# ------------------- TAB 2: Model Metrics -------------------
with tab2:
    st.header("Model Performance Comparison")

    # Display actual metrics for runs
    st.subheader("Runs Prediction Model Metrics")
    results_runs_data = {
        "Model": ["Baseline", "Random Forest", "XGBoost", "LightGBM"],
        "RMSE": [21.65, 23.94, 22.86, 23.68],
        "MAE": [16.22, 17.61, 16.93, 17.25],
        "R2": [0.05, -0.16, -0.05, -0.13]
    }
    results_runs_df = pd.DataFrame(results_runs_data)
    st.dataframe(results_runs_df)
    fig_runs_r2, ax_runs_r2 = plt.subplots()
    results_runs_df.set_index("Model")["R2"][:].plot(kind="bar", ax=ax_runs_r2)
    ax_runs_r2.set_ylabel("R2 Score")
    ax_runs_r2.set_title("Runs Prediction R2 Scores")
    st.pyplot(fig_runs_r2)

    # Display actual metrics for wickets
    st.subheader("Wickets Prediction Model Metrics")
    results_wkts_data = {
        "Model": ["Random Forest", "XGBoost", "LightGBM", "Poisson Regressor"],
        "RMSE": [1.078, 1.067, 1.077, 1.006],
        "MAE": [0.860, 0.853, 0.856, 0.814],
        "R2": [-0.169, -0.145, -0.166, -0.017]
    }
    results_wkts_df = pd.DataFrame(results_wkts_data)
    st.dataframe(results_wkts_df)
    fig_wkts_r2, ax_wkts_r2 = plt.subplots()
    results_wkts_df.set_index("Model")["R2"][:].plot(kind="bar", ax=ax_wkts_r2)
    ax_wkts_r2.set_ylabel("R2 Score")
    ax_wkts_r2.set_title("Wickets Prediction R2 Scores")
    st.pyplot(fig_wkts_r2)

# ------------------- TAB 3: Feature Importance -------------------
with tab3:
    st.header("Feature Importance (Top 10)")
    st.markdown("Understanding which features contribute most to the predictions.")

    if runs_model is not None:
        st.subheader("Runs Model Feature Importance")
        # Ensure feature_names_in_ exists or use X_train_runs.columns
        features_runs_display = runs_model.feature_names_in_ if hasattr(runs_model, 'feature_names_in_') else X_train_runs.columns # Assuming X_train_runs.columns is available at app launch time
        importances_runs = runs_model.feature_importances_

        imp_df_runs = pd.DataFrame({
            "Feature": features_runs_display,
            "Importance": importances_runs
        }).sort_values(by="Importance", ascending=False)

        st.dataframe(imp_df_runs.head(10))

        fig_runs_imp, ax_runs_imp = plt.subplots(figsize=(10, 6))
        ax_runs_imp.barh(imp_df_runs["Feature"][:10], imp_df_runs["Importance"][:10])
        ax_runs_imp.invert_yaxis()
        ax_runs_imp.set_title("Runs Model Feature Importance")
        st.pyplot(fig_runs_imp)
    else:
        st.warning("Runs model not loaded, cannot display feature importance.")

    if wkts_model is not None:
        st.subheader("Wickets Model Feature Importance")
        # For PoissonRegressor, use coefficients as importance
        features_wkts_display = list(X_train_wkts.columns) # Use the exact columns the model was trained on
        importances_wkts = np.abs(wkts_model.coef_) # Absolute coefficients for importance

        imp_df_wkts = pd.DataFrame({
            "Feature": features_wkts_display,
            "Importance": importances_wkts
        }).sort_values(by="Importance", ascending=False)

        st.dataframe(imp_df_wkts.head(10))

        fig_wkts_imp, ax_wkts_imp = plt.subplots(figsize=(10, 6))
        ax_wkts_imp.barh(imp_df_wkts["Feature"][:10], imp_df_wkts["Importance"][:10])
        ax_wkts_imp.invert_yaxis()
        ax_wkts_imp.set_title("Wickets Model Feature Importance")
        st.pyplot(fig_wkts_imp)
    else:
        st.warning("Wickets model not loaded, cannot display feature importance.")

st.markdown("---")
st.markdown("Developed with ❤️ for IPL Analytics")