from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import requests

device = "cpu"
local_model_dir = "./models"
model_name = "humarin/chatgpt_paraphraser_on_T5_base"

def is_internet_available(url="http://www.google.com/", timeout=5):
    try:
        _ = requests.get(url, timeout=timeout)
        return True
    except (requests.ConnectionError, requests.exceptions.RequestException):
        return False

def load_model_and_tokenizer():
    if is_internet_available():
        print("Internet is available, attempting to load model from Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        if not os.path.exists(local_model_dir):
            os.makedirs(local_model_dir, exist_ok=True)
            tokenizer.save_pretrained(local_model_dir)
            model.save_pretrained(local_model_dir)
            print("Save model successfully")
    else:
        print("Internet not available or failed to load from Hugging Face, loading model from local cache...")
        tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(local_model_dir).to(device)
    
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

def paraphrase(
    question,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=1,
    repetition_penalty=5.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    max_length=128
):
    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids.to(device)
    
    outputs = model.generate(
        input_ids, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty,
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res

app = Flask(__name__)

@app.route('/paraphrase', methods=['POST'])
def paraphrase_handler():
    data = request.json
    question = data.get('question', None)
    if not question:
        return jsonify({"error": "No question provided"}), 400
    try:
        paraphrased_texts = paraphrase(question)
        return jsonify({"paraphrased_texts": paraphrased_texts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7999, debug=False)
