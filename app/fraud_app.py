import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from io import StringIO
from ui_layout import render_sidebar, render_header, render_footer

# -----------------------------
# Streamlit Configuration
# -----------------------------
st.set_page_config(page_title="Stripe Fraud Detector", page_icon="💳", layout="centered")
render_sidebar()
render_header()

# -----------------------------
# Load the trained ML model
# -----------------------------
model = joblib.load(r"C:\Users\shash\Downloads\Stripe_Fraud_Detection_Project\models\fraud_detection_model_smote.pkl")

# The required feature order (must match model training)
FEATURE_ORDER = ['Time'] + [f"V{i}" for i in range(1, 29)] + ['Amount']

# -----------------------------
# Main Interface
# -----------------------------
tab1, tab2 = st.tabs(["📤 Upload CSV", "🧪 Generate Mock Data"])

# ==============================
# MODE 1: CSV FILE UPLOAD (tab1)
# ==============================
with tab1:
    uploaded_file = st.file_uploader("Upload a CSV file with columns: Time, V1–V28, Amount", type=["csv"])

    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            threshold = st.slider("Set fraud detection threshold", 0.01, 0.80, 0.15, step=0.01, key="csv_threshold")

            if all(col in user_df.columns for col in FEATURE_ORDER):
                user_df = user_df[FEATURE_ORDER]

                # Predict
                probas = model.predict_proba(user_df)[:, 1]
                predictions = [1 if p >= threshold else 0 for p in probas]
                user_df["Fraud Probability"] = probas
                user_df["Prediction"] = ["🚨 Fraud" if p == 1 else "✅ Legit" for p in predictions]

                st.success("✅ Prediction completed!")
                st.metric("Fraud Detected", f"{predictions.count(1)} / {len(predictions)}")

                # Show data and download option
                st.dataframe(user_df)
                csv = user_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Results (CSV)", csv, "fraud_predictions.csv", "text/csv")

                # Plot fraud probability histogram
                st.subheader("📊 Fraud Probability Distribution")
                fig, ax = plt.subplots()
                sns.histplot(user_df["Fraud Probability"], bins=30, kde=True, ax=ax)
                ax.set_xlabel("Fraud Probability")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)

                # Dashboard-style visuals
                st.subheader("📊 Prediction Breakdown")
                pred_counts = pd.Series(predictions).value_counts().sort_index()
                st.bar_chart(pred_counts.rename(index={0: "Legit", 1: "Fraud"}))

                st.subheader("📈 Fraud Probability Trend")
                st.line_chart(user_df["Fraud Probability"])

                st.subheader("📆 Fraud vs Legit by Hour")
                temp_df = user_df.copy()
                temp_df["Class"] = temp_df["Prediction"].apply(lambda x: 1 if "Fraud" in x else 0)
                temp_df["Hour"] = pd.to_datetime(temp_df["Time"], unit="s").dt.hour
                fig, ax = plt.subplots()
                sns.histplot(data=temp_df, x="Hour", hue="Class", multiple="stack", discrete=True, ax=ax)
                ax.set_title("Fraudulent vs Legit Transactions by Hour")
                ax.set_xlabel("Hour of Day")
                st.pyplot(fig)

                # Enhanced fraud trend: bucket by 1hr intervals
                st.subheader("📊 Enhanced Time-Based Fraud Rate")
                df_time = temp_df.copy()
                bucket_edges = np.arange(0, df_time["Time"].max() + 3600, 3600)
                df_time["Time Bucket"] = pd.cut(df_time["Time"], bins=bucket_edges)
                time_trend = df_time.groupby("Time Bucket")["Class"].mean().reset_index()
                time_trend["Time Bucket"] = time_trend["Time Bucket"].apply(lambda x: f"{pd.to_datetime(x.left, unit='s').strftime('%H:%M')}–{pd.to_datetime(x.right, unit='s').strftime('%H:%M')}")
                fig, ax = plt.subplots(figsize=(10,4))
                sns.lineplot(data=time_trend, x="Time Bucket", y="Class", ax=ax)
                ax.set_ylabel("Fraud Rate")
                ax.set_xlabel("Time Interval (HH:MM)")
                ax.set_title("Fraud Rate Over Time")
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)

                st.subheader("🔍 Top 5 Highest Risk Transactions")
                top_frauds = user_df.sort_values(by="Fraud Probability", ascending=False).head(5)
                st.dataframe(top_frauds)

                # Generate HTML report
                st.subheader("📄 Download Summary Report")
                fraud_rate = predictions.count(1) / len(predictions) * 100
                summary_html = f"""
                <html>
                    <head><title>Fraud Detection Report</title></head>
                    <body>
                        <h2>Stripe Fraud Detection Summary</h2>
                        <p><strong>Total Transactions:</strong> {len(predictions)}</p>
                        <p><strong>Flagged as Fraud:</strong> {predictions.count(1)}</p>
                        <p><strong>Legit:</strong> {predictions.count(0)}</p>
                        <p><strong>Fraud Rate:</strong> {fraud_rate:.2f}%</p>
                        <br>
                        <h3>Top Risky Transactions</h3>
                        {top_frauds.to_html(index=False)}
                    </body>
                </html>
                """
                st.download_button(
                    label="⬇️ Download Report (HTML)",
                    data=summary_html,
                    file_name="fraud_summary_report.html",
                    mime="text/html"
                )

            else:
                st.error("Uploaded file must contain all required columns.")

        except Exception as e:
            st.error(f"Error reading file: {e}")

# ==============================
# MODE 2: MOCK DATA GENERATION (tab2)
# ==============================
with tab2:
    mock_type = np.random.choice(["Random", "Likely Legitimate", "Likely Fraudulent"])

    # Function to generate mock transaction
    def generate_mock_transaction(type="Random"):
        np.random.seed()
        mock = {
            "Time": np.random.randint(10000, 172792),
            "Amount": np.round(np.random.uniform(0.5, 2500), 2)
        }
        for i in range(1, 29):
            if type == "Random":
                mock[f"V{i}"] = np.round(np.random.normal(0, 1), 4)
            elif type == "Likely Legitimate":
                mock[f"V{i}"] = np.round(np.random.normal(0, 0.5), 4)
            elif type == "Likely Fraudulent":
                mock[f"V{i}"] = np.round(np.random.normal(3, 2), 4) if np.random.rand() > 0.5 else np.round(np.random.normal(-3, 2), 4)
        return mock

    # Persist mock transaction using session state
    if "mock_data" not in st.session_state:
        st.session_state.mock_data = generate_mock_transaction(mock_type)

    if st.button("🔁 Refresh Mock Data"):
        st.session_state.mock_data = generate_mock_transaction(mock_type)

    # Show mock data
    input_data = st.session_state.mock_data
    st.subheader("Generated Transaction Features")
    st.json(input_data)

    # Prepare input for prediction
    input_df = pd.DataFrame([input_data])[FEATURE_ORDER]
    threshold = st.slider("Set fraud detection threshold", 0.01, 0.80, 0.15, step=0.01, key="mock_threshold")

    if st.button("Check Transaction", key="predict_button"):
        proba = model.predict_proba(input_df)[0][1]
        result = 1 if proba >= threshold else 0

        # Gauge chart for fraud probability
        st.subheader("📟 Fraud Risk Meter")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba,
            title={'text': "Fraud Probability"},
            gauge={'axis': {'range': [0, 1]},
                   'bar': {'color': "red" if proba >= threshold else "green"},
                   'steps': [
                       {'range': [0, 0.3], 'color': "lightgreen"},
                       {'range': [0.3, 0.6], 'color': "yellow"},
                       {'range': [0.6, 1], 'color': "orangered"}]
                   }
        ))
        st.plotly_chart(fig)

        # Show prediction results
        st.write(f"🧠 Fraud probability: {proba:.2f} (Threshold: {threshold})")
        if result == 1:
            st.error(f"🚨 FRAUDULENT (Confidence: {proba:.2f})")
        else:
            st.success(f"✅ LEGITIMATE (Confidence: {proba:.2f})")

# -----------------------------
# Footer
# -----------------------------
render_footer()
