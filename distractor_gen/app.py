import itertools

import nltk
nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from flask import Flask, request, jsonify
from peft.peft_model import PeftModel
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)


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

smoothing_function = SmoothingFunction().method1

t5model = T5ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    return_dict=True
)
peft_path = './best-checkpoint-modif.ckpt'
peft_model = PeftModel.from_pretrained(t5model, peft_path).to(device)
peft_model.to(device)

app = Flask(__name__)

def generate_distractors(qgmodel, answer: str, context: str, question: str) -> str:
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

    generated_ids = qgmodel.generate(
        input_ids=source_encoding['input_ids'].to(device),
        attention_mask=source_encoding['attention_mask'].to(device),
        num_beams=10,
        temperature=1.5,
        max_length=TARGET_MAX_TOKEN_LEN,
        repetition_penalty=2.5,
        early_stopping=True,
        use_cache=True,
        num_return_sequences=10,
        do_sample=True,
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
    best_similarity = float('inf')  # Initialize with a high value
    for list_opts in itertools.combinations(formated_options, 3):
        total_similarity = 0.0
        for i in range(len(list_opts)):
            for j in range(i+1, len(list_opts)):
                similarity = sentence_bleu([list_opts[i].split()], list_opts[j].split(), smoothing_function=smoothing_function)
                total_similarity += similarity

        for options in list_opts:
            total_similarity += sentence_bleu([answer.split()], options.split(), smoothing_function=smoothing_function)

        if total_similarity < best_similarity:
            best_similarity = total_similarity
            best_combination = list_opts
    for element in best_combination:
        element = element.strip()
    return list(best_combination)


@app.route('/generate_distractors', methods=['POST'])
def generate_distractors_api():
    data = request.json
    context = data['context']
    question = data['question']
    answer = data['answer']

    distractors = generate_distractors(peft_model, answer, context, question)
    return jsonify({'distractors': distractors})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002, debug=True)