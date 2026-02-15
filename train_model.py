!pip install streamlit xgboost scikit-learn pandas numpy matplotlib seaborn joblib

from math import e
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Create model directory
os.makedirs('model', exist_ok=True)

# 1. Load the dataset directly from the downloaded Kaggle CSV
data = pd.read_csv('data.csv')

# 2. Preprocess the dataset
try: 
  # Drop 'id' and the empty 'Unnamed: 32' column
  cols_to_drop = ['id']
  if 'Unnamed: 32' in data.columns:
      cols_to_drop.append('Unnamed: 32')
  data = data.drop(columns=cols_to_drop)
except:
  print("Columns already dropped")

# Encode 'diagnosis' (M = 1, B = 0) and rename it to 'target' for Streamlit consistency
data['target'] = data['diagnosis'].map({'M': 1, 'B': 0})
data = data.drop(columns=['diagnosis'])

# Separate features (X) and target (y)
X = data.drop(columns=['target'])
y = data['target']

# Split data - saving 20% to create the test_data.csv for Streamlit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the test data to a CSV so you can upload it in the Streamlit app
test_df = X_test.copy()
test_df['target'] = y_test
test_df.to_csv('test_data.csv', index=False)

# Define the 6 Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

print("Training models and calculating metrics...")
results = []

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    # Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    results.append({
        "ML Model Name": name, "Accuracy": acc, "AUC": auc, 
        "Precision": prec, "Recall": rec, "F1": f1, "MCC": mcc
    })
    
    # Save Model
    joblib.dump(model, f'model/{name.replace(" ", "_").lower()}.pkl')

# Print table for your README
results_df = pd.DataFrame(results)
print("\n--- Use this table for your README.md ---")
print(results_df.to_markdown(index=False))
