from flask import Flask, request, jsonify
import joblib
import re
import nltk
import os
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

app = Flask(__name__)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the final Random Forest model and the TF-IDF vectorizer
best_model = joblib.load(os.path.join(script_dir, "best_model.joblib"))
vectorizer = joblib.load(os.path.join(script_dir, "tfidf_vectorizer.joblib"))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

@app.route("/")
def home():
    return """
    <h1>Fake News Detection API</h1>
    <p>Welcome to the Fake News Detection API!</p>
    <p>To use this API, send a POST request to <code>/predict</code> with JSON data:</p>
    <pre>
    {
        "title": "Your news title here"
    }
    </pre>
    <p>The API will return whether the news is "Fake" or "Real".</p>
    """

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    raw_title = data.get("title", "")

    # Preprocess
    cleaned_title = preprocess_text(raw_title)

    # Vectorize
    X_vec = vectorizer.transform([cleaned_title])

    # Predict (0 = Fake, 1 = Real)
    prediction = best_model.predict(X_vec)[0]
    label_str = "Real" if prediction == 1 else "Fake"

    return jsonify({"prediction": label_str})

if __name__ == "__main__":
    app.run(debug=True)
