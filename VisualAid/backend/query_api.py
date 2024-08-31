import os
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
from gtts import gTTS
import tempfile
import pygame
from flask import Flask, request, jsonify

# Suppress warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Initialize Flask app
app = Flask(__name__)

# Initialize the retriever model and tokenizer
retriever_model = SentenceTransformer('thenlper/gte-large')
retriever_tokenizer = AutoTokenizer.from_pretrained('thenlper/gte-large')

# Initialize the generator model and tokenizer
generator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")
generator_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def retrieve_passages(query, stored_passages, num_passages=5):
    # Encode the stored passages using SentenceTransformer
    stored_embeddings = retriever_model.encode(stored_passages, convert_to_tensor=True)

    # Tokenize and encode the query
    query_inputs = retriever_tokenizer(query, return_tensors="pt", max_length=512, truncation=True, padding=True)

    # Get embeddings for the query
    with torch.no_grad():
        query_outputs = retriever_model.encode(query, convert_to_tensor=True)

    # Compute cosine similarity between query embedding and stored passage embeddings
    scores = util.pytorch_cos_sim(query_outputs, stored_embeddings)[0]

    # Get top passages based on scores
    top_passages = [(score.item(), passage) for score, passage in zip(scores, stored_passages)]

    # Sort passages by similarity score
    top_passages.sort(key=lambda x: x[0], reverse=True)

    # Return top num_passages passages
    return top_passages[:num_passages]

def generate_answer(context, question):
    input_text = context + " " + question  # Combine context and question into a single string

    inputs = generator_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        outputs = generator_model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=50)
    return generator_tokenizer.batch_decode(outputs, skip_special_tokens=True)

def speak_text(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        tts.save(fp.name + ".mp3")
        pygame.mixer.init()
        pygame.mixer.music.load(fp.name + ".mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue

@app.route('/api/query', methods=['POST'])
def query_api():
    print("ENTERED")
    data = request.get_json()
    query = data.get('query')
    stored_passages = data.get('stored_passages')
    # print("Query: "+query+"\n")
    # print("Stored Passages"+stored_passages+"\n")

    if not query or not stored_passages:
        return jsonify({"error": "Query and stored_passages are required"}), 400

    retrieved_passages = retrieve_passages(query, stored_passages)

    # Get the top passage
    if retrieved_passages:
        top_passage = retrieved_passages[0][1]

        # Generate answer
        answers = generate_answer(top_passage, query)

        # Respond with the generated answer
        response = {
            "query": query,
            "retrieved_passages": [{"score": score, "passage": passage} for score, passage in retrieved_passages],
            "answers": answers
        }
        return jsonify(response)

    return jsonify({"error": "No passages found"}), 404

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5003, debug=True)