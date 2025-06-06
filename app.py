import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# Page settings
st.set_page_config(page_title="Heart Failure Prediction", layout="wide")
st.title("💓 Heart Failure Prediction using Logistic Regression")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file with 'DEATH_EVENT' column", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("🔍 Preview of Uploaded Data")
        st.dataframe(df.head())

        # Ensure target column exists
        if "DEATH_EVENT" not in df.columns:
            st.error("❌ The dataset must contain a 'DEATH_EVENT' column.")
        else:
            # Split features and labels
            X = df.drop("DEATH_EVENT", axis=1)
            y = df["DEATH_EVENT"]

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Split dataset
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, stratify=y, random_state=42
            )

            # Train Logistic Regression
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # Accuracy
            acc = accuracy_score(y_test, y_pred)
            st.metric(label="✅ Model Accuracy", value=f"{acc:.2%}")

            # Confusion Matrix
            st.subheader("📊 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
            fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="blues")
            fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(fig_cm)

            # ROC Curve
            st.subheader("📈 ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {roc_auc:.2f}'))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
            fig_roc.update_layout(title='ROC Curve',
                                  xaxis_title='False Positive Rate',
                                  yaxis_title='True Positive Rate',
                                  showlegend=True)
            st.plotly_chart(fig_roc)

            # Feature Importance
            st.subheader("📌 Feature Coefficients")
            coefficients = pd.Series(model.coef_[0], index=X.columns).sort_values()
            fig_feat = px.bar(coefficients, orientation='h', labels={'value': 'Coefficient', 'index': 'Feature'})
            fig_feat.update_layout(title="Logistic Regression Coefficients")
            st.plotly_chart(fig_feat)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
else:
    st.info("📥 Please upload a CSV file to begin.")
