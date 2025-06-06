import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# Page setup
st.set_page_config(page_title="Heart Failure Prediction", layout="wide")
st.title("💓 Heart Failure Prediction using Logistic Regression")

# File upload
uploaded_file = st.file_uploader("Upload your Heart Failure CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📋 Preview of Uploaded Data")
        st.dataframe(df.head())

        if "DEATH_EVENT" not in df.columns:
            st.error("❌ 'DEATH_EVENT' column not found in the dataset.")
        else:
            # Prepare data
            X = df.drop("DEATH_EVENT", axis=1)
            y = df["DEATH_EVENT"]

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, stratify=y, random_state=42
            )

            # Train model
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # Accuracy
            accuracy = accuracy_score(y_test, y_pred)
            st.metric("✅ Model Accuracy", f"{accuracy:.2%}")

            # Confusion Matrix using Plotly
            cm = confusion_matrix(y_test, y_pred)
            st.subheader("📊 Confusion Matrix")
            fig_cm = px.imshow(cm,
                               text_auto=True,
                               color_continuous_scale="Blues",
                               labels=dict(x="Predicted", y="Actual", color="Count"))
            fig_cm.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(fig_cm)

            # ROC Curve using Plotly
            st.subheader("📈 ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {roc_auc:.2f}'))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
            fig_roc.update_layout(xaxis_title='False Positive Rate',
                                  yaxis_title='True Positive Rate',
                                  title='ROC Curve')
            st.plotly_chart(fig_roc)

            # Feature Importance using Plotly
            st.subheader("📌 Feature Coefficients")
            coefficients = pd.Series(model.coef_[0], index=X.columns).sort_values()
            fig_coeff = px.bar(coefficients, orientation='h', title='Logistic Regression Coefficients')
            st.plotly_chart(fig_coeff)

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
else:
    st.info("📥 Please upload a CSV file to begin.")
