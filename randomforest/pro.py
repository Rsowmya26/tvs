from flask import Flask, request, render_template
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Load the trained model and vectorizer
lr_model = joblib.load('rf_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Initialize Flask app
app = Flask(__name__)

def preprocess_text(text):
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    words = text.split()
    # Remove stop words and lemmatize
    lemmatizer = WordNetLemmatizer()
    meaningful_words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    return ' '.join(meaningful_words)

def is_meaningful_sentence(text):
    # Basic check: Ensure the sentence has more than one word
    words = text.split()
    return len(words) > 1 and any(word.lower() in set(stopwords.words('english')) for word in words)

def predict_sentiment(text):
    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    # Vectorize the input text
    vectorized_text = vectorizer.transform([preprocessed_text])
    # Predict sentiment
    prediction = lr_model.predict(vectorized_text)
    return int(prediction[0])

# Define the main route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the text input from the form
        text = request.form['text']
       
        # Check if the text input is empty
        if not text.strip():
            return render_template('index.html', prediction_text="Please enter text to analyze sentiment.")
       
        # Check if the text contains meaningful English words
        if not is_meaningful_sentence(text):
            return render_template('index.html', prediction_text="The text does not appear to be meaningful. Please enter a correct sentence.")
       
        # Get the prediction
        sentiment = predict_sentiment(text)
       
        # Return the result
        if sentiment == 1:
            return render_template('index.html', prediction_text='The sentiment is positive.')
        else:
            return render_template('index.html', prediction_text='The sentiment is negative.')

if __name__ == "__main__":
    app.run(debug=True)
