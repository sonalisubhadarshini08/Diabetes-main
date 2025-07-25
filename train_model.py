import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load up to 50,000 rows
DATA_PATH = '../healthcare_dataset.csv'
df = pd.read_csv(DATA_PATH, nrows=50000)

# Try to infer the target column
possible_targets = ['readmitted', 'readmission', 'target', 'output', 'label']
target_col = None
for col in df.columns:
    if col.lower() in possible_targets:
        target_col = col
        break
if not target_col:
    # Fallback: use the last column
    target_col = df.columns[-1]

print(f'Using target column: {target_col}')

# Drop obvious ID columns if present
id_cols = [c for c in df.columns if 'id' in c.lower()]
X = df.drop([target_col] + id_cols, axis=1, errors='ignore')
y = df[target_col]

# Encode categorical features
for col in X.select_dtypes(include=['object', 'category']).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Encode target if needed
if y.dtype == 'object' or y.dtype.name == 'category':
    y = LabelEncoder().fit_transform(y.astype(str))

# Fill missing values
X = X.fillna(-1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump({'model': clf, 'features': X.columns.tolist()}, 'model.pkl')
print('Model saved as model.pkl') 