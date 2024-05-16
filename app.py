# save this as app.py
from flask import Flask, render_template, request
import pandas as pd
from flask_cors import CORS
import re
from nltk.translate import AlignedSent, IBMModel1
import jsonify

app = Flask(__name__)
CORS(app) 

# Load dataset and train translation models
data = pd.read_csv("data/data.csv")  # Update with your file path
shona_sentences = data['shona'].values
ndebele_sentences = data['ndebele'].values

def clean_sentences(sentences):
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        sentence = sentence.lower()
        sentence = re.sub(r"[^a-zA-Z0-9]+", " ", sentence)
        cleaned_sentences.append(sentence.strip())
    return cleaned_sentences

cleaned_shona_sentences = clean_sentences(shona_sentences)
cleaned_ndebele_sentences = clean_sentences(ndebele_sentences)

def train_translation_model(source_sentences, target_sentences):
    aligned_sentences = [AlignedSent(source.split(), target.split()) for source, target in zip(source_sentences, target_sentences)]
    ibm_model = IBMModel1(aligned_sentences, 10)
    return ibm_model

shona_to_ndebele_model = train_translation_model(cleaned_shona_sentences, cleaned_ndebele_sentences)
ndebele_to_shona_model = train_translation_model(cleaned_ndebele_sentences, cleaned_shona_sentences)

def translate(ibm_model, source_text):
    cleaned_text = clean_sentences(source_text.split())
    source_words = cleaned_text
    translated_words = []
    for source_word in source_words:
        max_prob = 0.0
        translated_word = None
        for target_word in ibm_model.translation_table[source_word]:
            prob = ibm_model.translation_table[source_word][target_word]
            if prob > max_prob:
                max_prob = prob
                translated_word = target_word
        if translated_word is not None:
            translated_words.append(translated_word)
        print(source_word)
    translated_text = ' '.join(translated_words)
    return translated_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate_text():
    request_data = request.json  # Get the JSON data from the request body
    source_text = request_data['source_text']
    translation_model = request_data['translation_model']

    if translation_model == 'shona_to_ndebele':
        translated_text = translate(shona_to_ndebele_model, source_text)
    elif translation_model == 'ndebele_to_shona':
        translated_text = translate(ndebele_to_shona_model, source_text)
    
    print(source_text)
    # Create a dictionary to hold the translation result
    translation_result = {
        "source_text": source_text,
        "translated_text": translated_text
    }
    
    # Return the translation result as JSON
    return translation_result

if __name__ == '__main__':
    app.run(debug=True)