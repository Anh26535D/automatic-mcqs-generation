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
        temperature=1.2,
        repetition_penalty=2.5,
        top_p=0.95,
        max_length=TARGET_MAX_TOKEN_LEN,
        early_stopping=True,
        use_cache=True,
        num_return_sequences=20,
        do_sample=True,
    )

    preds = {
        tokenizer.decode(generated_id, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        for generated_id in generated_ids
    }
    
    formatted_options = []
    for option in preds:
        option = option.replace('<pad>', '')
        option = option.replace('</s>', '')
        distractors = option.split(';')
        for distractor in distractors:
            if distractor:
                formatted_options.append(distractor)
        
    formatted_options = list(set(formatted_options))

    if len(formatted_options) == 0:
        formatted_options.append("-")
        formatted_options.append("-")
        formatted_options.append("-")
    if len(formatted_options) == 1:
        formatted_options.append("-")
        formatted_options.append("-")
    if len(formatted_options) == 2:
        formatted_options.append("-")
    
    formatted_options = [option.strip() for option in formatted_options]
    # remove mark in options
    marks = ['.', ',', '?', '!', ':', ';']
    formatted_options = [option if option[-1] not in marks else option[:-1] for option in formatted_options]
    
    # Remove options that have high similarity with the answer (above 0.25 percentile)
    lst_option_embeddings = model.encode(formatted_options, convert_to_tensor=True)
    answer_embedding = model.encode([answer])
    similarity = [cosine_sim(answer_embedding, opt_embedding).item() for opt_embedding in lst_option_embeddings]
    selected_option_indices = [i for i, sim in enumerate(similarity) if sim > np.percentile(similarity, 25)]
    
    # Remove options that have seem to be the same with each other
    removed_option_indices = []
    for i in selected_option_indices:
        for j in selected_option_indices:
            if i != j:
                semantic_similarity = cosine_sim(lst_option_embeddings[i], lst_option_embeddings[j]).item()
                token_similarity = sentence_bleu([formatted_options[i].split()], formatted_options[j].split(), smoothing_function=smoothing_function)
                if semantic_similarity > 0.95 or token_similarity > 0.95:
                    if j not in removed_option_indices and i not in removed_option_indices:
                        removed_option_indices.append(j)
    selected_option_indices = [i for i in selected_option_indices if i not in removed_option_indices]

    best_com = None
    best_score = -float('inf')
    eps = 1e-6
    for comb in itertools.combinations(selected_option_indices, 3):
        d2d_score = 0
        cnt = 0
        for i in comb:
            for j in comb:
                if i != j:
                    cnt += 1
                    token_similarity = sentence_bleu([formatted_options[i].split()], formatted_options[j].split(), smoothing_function=smoothing_function)
                    semantic_similarity = cosine_sim(lst_option_embeddings[i], lst_option_embeddings[j]).item()
                    d2d_score = d2d_score + 1 / ((semantic_similarity + eps) + (token_similarity + eps)) 
        d2d_score = d2d_score / cnt
        
        d2a_score = 0
        cnt = 0
        for i in comb:
            cnt += 1
            token_similarity = sentence_bleu([answer.split()], formatted_options[i].split(), smoothing_function=smoothing_function)
            semantic_similarity = cosine_sim(answer_embedding, lst_option_embeddings[i]).item()
            d2a_score = d2a_score + 1 / ((semantic_similarity + eps) + (token_similarity + eps)) 
        d2a_score = d2a_score / cnt
        
        alpha = 0.5
        total_score = 1 / (alpha / (d2d_score + eps) + (1 - alpha) / (d2a_score + eps))

        if total_score > best_score:
            best_score = total_score
            best_com = comb

    return [formatted_options[i] for i in best_com]

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