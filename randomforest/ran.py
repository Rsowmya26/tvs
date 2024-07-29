import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Load a sample of the dataset for quicker testing
print("Loading dataset...")
df = pd.read_csv('D:\\intern\\training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None, nrows=100000)  # Load only 100000 rows
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

# Preprocess the Dataset
print("Preprocessing dataset...")
df = df[['target', 'text']]
df['target'] = df['target'].replace(4, 1)

# Separate features and target
X = df['text']
y = df['target']

# Split the data into training, validation, and test sets
print("Splitting dataset...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Vectorize the text data using TF-IDF
print("Vectorizing text data...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

# Model Training
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)

# Model Evaluation
print("Evaluating model...")
y_test_pred = rf_model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_test_pred)
print("Accuracy Score:", accuracy)

# Save the Model and Vectorizer
print("Saving model and vectorizer...")
model_path = 'rf_model.pkl'
vectorizer_path = 'vectorizer.pkl'

joblib.dump(rf_model, model_path)
joblib.dump(vectorizer, vectorizer_path)

# Confirm files are created
if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    print(f"Model and vectorizer saved successfully to {model_path} and {vectorizer_path}.")
else:
    print("Error: Model and vectorizer were not saved.")
