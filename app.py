import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

st.title("Machine Learning Classification App")
st.write("Upload your test dataset to evaluate different models.")

uploaded_file = st.file_uploader("Upload CSV Test Data", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:", data.head())
    
    if 'target' not in data.columns:
        st.error("The uploaded CSV must contain a 'target' column.")
    else:
        X_test = data.drop(columns=['target'])
        y_test = data['target']
        
        model_options = {
            "Logistic Regression": "logistic_regression.pkl",
            "Decision Tree": "decision_tree.pkl",
            "KNN": "knn.pkl",
            "Naive Bayes": "naive_bayes.pkl",
            "Random Forest (Ensemble)": "random_forest.pkl",
            "XGBoost (Ensemble)": "xgboost.pkl"
        }
        
        selected_model_name = st.selectbox("Select a Model", list(model_options.keys()))
        model_path = os.path.join("model", model_options[selected_model_name])
        
        if st.button("Evaluate Model"):
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
                
                st.subheader(f"Metrics for {selected_model_name}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
                col2.metric("AUC Score", f"{roc_auc_score(y_test, y_prob):.4f}")
                col3.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
                
                col4, col5, col6 = st.columns(3)
                col4.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
                col5.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")
                col6.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.4f}")
                
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                st.pyplot(fig)
            else:
                st.error("Model file not found.")
