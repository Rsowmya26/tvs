import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load the Dataset
df = pd.read_csv('D:\\intern\\training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None)
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

# Preprocess the Dataset
df = df[['target', 'text']]
df['target'] = df['target'].replace(4, 1)

# Separate features and target
X = df['text']
y = df['target']

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

# Model Training
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train_tfidf, y_train)

# Model Evaluation
y_test_pred = logistic_model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_test_pred)
print("Accuracy Score:", accuracy)

# Save the Model and Vectorizer
joblib.dump(logistic_model, 'logistic_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
