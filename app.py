# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# Set page title
st.set_page_config(page_title="Heart Failure Prediction", layout="wide")

# Title
st.title("💓 Heart Failure Prediction using Logistic Regression")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Raw Data")
    st.dataframe(df.head())

    # Preprocessing
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
    acc = accuracy_score(y_test, y_pred)
    st.metric("📊 Accuracy", f"{acc:.2f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    st.pyplot(plt.gcf())
    plt.clf()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    st.subheader("ROC Curve")
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()

    # Feature importance
    st.subheader("Feature Coefficients")
    importance = pd.Series(model.coef_[0], index=X.columns).sort_values()
    importance.plot(kind='barh', title='Logistic Regression Coefficients')
    st.pyplot(plt.gcf())
    plt.clf()
