import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

st.set_page_config(page_title="Heart Failure Prediction", layout="wide")
st.title("💓 Heart Failure Prediction using Logistic Regression")

uploaded_file = st.file_uploader("Upload your Heart Failure CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📋 Preview of Uploaded Data")
        st.dataframe(df.head())

        if "DEATH_EVENT" not in df.columns:
            st.error("❌ 'DEATH_EVENT' column missing.")
        else:
            X = df.drop("DEATH_EVENT", axis=1)
            y = df["DEATH_EVENT"]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, stratify=y, random_state=42
            )

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            st.metric("✅ Accuracy", f"{acc:.2%}")

            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm,
                                 index=["Actual 0", "Actual 1"],
                                 columns=["Predicted 0", "Predicted 1"])
            st.subheader("Confusion Matrix")
            st.dataframe(cm_df)

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            st.subheader("ROC Curve Data")
            roc_df = pd.DataFrame({"False Positive Rate": fpr, "True Positive Rate": tpr})
            st.line_chart(roc_df.set_index("False Positive Rate"))
            st.write(f"AUC Score: **{roc_auc:.2f}**")

            coef_df = pd.DataFrame({
                "Feature": X.columns,
                "Coefficient": model.coef_[0]
            }).sort_values("Coefficient")
            st.subheader("Feature Coefficients")
            st.bar_chart(coef_df.set_index("Feature"))

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
else:
    st.info("📥 Please upload a CSV file.")
