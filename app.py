from flask import Flask, request, jsonify
import pandas as pd
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

nltk.download('punkt')

app = Flask(__name__)

# Load the CSV data
df = pd.read_csv("ChatbotDataset.csv")
df['Question'] = df['Question'].str.lower()

model = SentenceTransformer('all-MiniLM-L6-v2')
question_embeddings = model.encode(df['Question'].tolist())

def match_question(user_input):
    user_embedding = model.encode([user_input.lower()])
    similarities = cosine_similarity(user_embedding, question_embeddings)
    most_similar_index = similarities.argmax()
    if similarities[0][most_similar_index] < 0.5:
        return "Sorry, I don't have an answer for that. Can you please ask differently?"
    else:
        return df['Answer'][most_similar_index]

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    fraction_match = re.search(r'-?\d+/\d+', user_input)

    if fraction_match:
        fraction = fraction_match.group()
        matched_answer = handle_fraction(fraction)
        if matched_answer:
            return jsonify({"response": matched_answer})
        else:
            return jsonify({"response": "It looks like there was an issue with the fraction format. Please try again."})
    else:
        matched_answer = match_question(user_input)
        return jsonify({"response": matched_answer})

def handle_fraction(fraction):
    try:
        numerator, denominator = map(int, fraction.split('/'))
        is_negative = numerator < 0
        numerator = abs(numerator)
        whole_number = numerator // denominator
        remainder = numerator % denominator
        fraction_type = "improper" if numerator > denominator else "proper"
        explanation = (
            f"{fraction} is a {fraction_type} fraction. {numerator}/{denominator} can be written as "
            f"{whole_number} {remainder}/{denominator}."
        )
        return explanation
    except ValueError:
        return None

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)  # Railway uses port 5000 by default
