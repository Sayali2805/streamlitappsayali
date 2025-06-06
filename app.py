import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

        # Basic check
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

            # Train logistic regression
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # Accuracy
            accuracy = accuracy_score(y_test, y_pred)
            st.metric("✅ Model Accuracy", f"{accuracy:.2%}")

            # Confusion Matrix
            st.subheader("📊 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig1, ax1 = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
            ax1.set_xlabel("Predicted")
            ax1.set_ylabel("Actual")
            st.pyplot(fig1)

            # ROC Curve
            st.subheader("📈 ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax2.plot([0, 1], [0, 1], 'k--')
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.legend()
            st.pyplot(fig2)

            # Feature Importance
            st.subheader("📌 Feature Coefficients")
            coefficients = pd.Series(model.coef_[0], index=X.columns).sort_values()
            fig3, ax3 = plt.subplots()
            coefficients.plot(kind="barh", ax=ax3)
            ax3.set_title("Logistic Regression Coefficients")
            st.pyplot(fig3)

    except Exception as e:
        st.error(f"❌ An error occurred while processing the file:\n\n{e}")
else:
    st.info("📥 Please upload a CSV file to begin.")
