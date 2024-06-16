import itertools

import nltk
nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
from flask import Flask, request, jsonify
from peft.peft_model import PeftModel
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)
from sentence_transformers import SentenceTransformer

smoothing_function = SmoothingFunction().method1
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


PROMPT_PLACEHOLDER = """
generate distractors for given context, question and answer:
context: {context};
question: {question};
answer: {correct};
</s>
"""

MODEL_NAME = 't5-small'
SOURCE_MAX_TOKEN_LEN = 512
TARGET_MAX_TOKEN_LEN = 64

device = "cpu"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
TOKENIZER_LEN = len(tokenizer)

t5model = T5ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    return_dict=True
)
peft_path = './best-checkpoint-modif.ckpt'
peft_model = PeftModel.from_pretrained(t5model, peft_path).to(device)
peft_model.to(device)

# app = Flask(__name__)

def cosine_sim(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def generate_distractors(answer: str, context: str, question: str) -> str:
    formatted_distractor = PROMPT_PLACEHOLDER.format(
        context=context,
        question=question,
        correct=answer,
    )
    source_encoding = tokenizer(
        formatted_distractor,
        max_length=SOURCE_MAX_TOKEN_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )

    generated_ids = peft_model.generate(
        input_ids=source_encoding['input_ids'].to(device),
        attention_mask=source_encoding['attention_mask'].to(device),
        num_beams=20,
        max_length=TARGET_MAX_TOKEN_LEN,
        early_stopping=True,
        use_cache=True,
        num_return_sequences=10,
    )

    preds = {
        tokenizer.decode(generated_id, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        for generated_id in generated_ids
    }
    
    formated_options = []
    for option in preds:
        option = option.replace('<pad>', '')
        option = option.replace('</s>', '')
        distractors = option.split(';')
        for distractor in distractors:
            if distractor:
                formated_options.append(distractor)
    
    for option in formated_options:
        option = option.strip()
        
    formated_options = list(set(formated_options))
    answer_embedding = model.encode([answer])
    
    if len(formated_options) == 0:
        formated_options.append("-")
        formated_options.append("-")
        formated_options.append("-")
    if len(formated_options) == 1:
        formated_options.append("-")
        formated_options.append("-")
    if len(formated_options) == 2:
        formated_options.append("-")

    best_combination = None
    best_score = float('inf')  # Initialize with a low value since we are maximizing the score
    for list_opts in itertools.combinations(formated_options, 3):
        total_score = 0.0
        list_opts_embeddings = model.encode(formated_options, convert_to_tensor=True)
        # Calculate the combined score based on the options
        total_similarity = 0.0
        num_pairs1 = 0
        for i in range(len(list_opts) - 1):
            for j in range(i + 1, len(list_opts)):
                semantic_similarity = cosine_sim(list_opts_embeddings[j], list_opts_embeddings[i]).item()
                bleu_similarity = sentence_bleu([list_opts[i].split()], list_opts[j].split(), smoothing_function=smoothing_function)
                num_pairs1 += 1
                score = 0.4*semantic_similarity + 0.6*bleu_similarity
                total_similarity += score
        total_score += total_similarity / num_pairs1
        # Calculate the combined score based on the answer
        total_similarity = 0.0
        num_pairs2 = 0
        for i in range(len(list_opts)):
            semantic_similarity = cosine_sim(answer_embedding, list_opts_embeddings[i]).item()
            bleu_similarity = sentence_bleu([answer.split()], list_opts[i].split(), smoothing_function=smoothing_function)
            score = 0.4*semantic_similarity + 0.6*bleu_similarity
            total_similarity += score
            num_pairs2 += 1
        total_score += total_similarity / num_pairs2
        if total_score < best_score:
            best_score = total_score
            best_combination = list_opts

    return list(best_combination)

if __name__ == '__main__':
    context = '''
    The Yanomami live along the rivers of the rainforest in the north of Brazil. 
    They have lived in the rainforest for about 10,000 years and they use more than 2,000 different plants for food and for medicine. 
    But in 1988, someone found gold in their forest, and suddenly 45,000 people came to the forest and began looking for gold. 
    They cut down the forest to make roads. 
    They made more than a hundred airports. 
    The Yanomami people lost land and food. 
    Many died because new diseases came to the forest with the strangers.
    In 1987, they closed fifteen roads for eight months. 
    No one cut down any trees during that time. In Panama, the Kuna people saved their forest. 
    They made a forest park which tourists pay to visit. 
    The Gavioes people of Brazil use the forest, but they protect it as well. 
    They find and sell the Brazil nuts which grow on the forest trees.
    '''
    question = "What was the purpose that those people built roads and airports?"
    answer = "carry away the gold conveniently"
    distractors = generate_distractors(answer, context, question)
    print(distractors)