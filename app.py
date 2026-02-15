import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

st.title("Breast Cancer Classification App")
st.write("Upload your test data to evaluate the machine learning models.")

# a. Dataset upload option
uploaded_file = st.file_uploader("Upload CSV Test Data", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:", data.head())
    
    # Check for 'target' or 'diagnosis' column and process accordingly
    if 'target' in data.columns:
        y_test = data['target']
        X_test = data.drop(columns=['target'])
        
    elif 'diagnosis' in data.columns:
        # Map 'M' to 1 and 'B' to 0
        y_test = data['diagnosis'].map({'M': 1, 'B': 0})
        X_test = data.drop(columns=['diagnosis'])
        
        # If they uploaded the raw dataset, drop 'id' and 'Unnamed: 32' to match model training
        cols_to_drop = []
        if 'id' in X_test.columns:
            cols_to_drop.append('id')
        if 'Unnamed: 32' in X_test.columns:
            cols_to_drop.append('Unnamed: 32')
        if cols_to_drop:
            X_test = X_test.drop(columns=cols_to_drop)
            
    else:
        st.error("Error: The uploaded CSV must contain either a 'target' (1/0) or 'diagnosis' (M/B) column.")
        st.stop() # Stops execution if neither column is found

    # b. Model selection dropdown
    model_options = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "KNN": "knn.pkl",
        "Naive Bayes": "naive_bayes.pkl",
        "Random Forest (Ensemble)": "random_forest.pkl",
        "XGBoost (Ensemble)": "xgboost.pkl"
    }
    
    selected_model_name = st.selectbox("Select a Model to Evaluate", list(model_options.keys()))
    model_path = os.path.join("model", model_options[selected_model_name])
    
    if st.button("Evaluate Model"):
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
            
            # c. Display of evaluation metrics
            st.subheader(f"Evaluation Metrics for {selected_model_name}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
            col2.metric("AUC Score", f"{roc_auc_score(y_test, y_prob):.4f}")
            col3.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
            
            col4, col5, col6 = st.columns(3)
            col4.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
            col5.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")
            col6.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.4f}")
            
            # d. Confusion matrix display
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted (1=Malignant, 0=Benign)')
            ax.set_ylabel('Actual (1=Malignant, 0=Benign)')
            st.pyplot(fig)
        else:
            st.error("Model file not found. Please ensure the model files are uploaded to the 'model/' directory on GitHub.")
