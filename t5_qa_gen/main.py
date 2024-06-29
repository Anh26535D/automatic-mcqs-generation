from flask import Flask, request, jsonify
from peft.peft_model import PeftModel
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)
import numpy as np
from tqdm import tqdm

MODEL_NAME = 't5-small'
SEP_TOKEN = '</s>'
SOURCE_MAX_TOKEN_LEN = 512
TARGET_MAX_TOKEN_LEN = 256

device = "cpu"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
TOKENIZER_LEN = len(tokenizer)

t5model = T5ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    return_dict=True
)
peft_path = './t5_qa_gen/best-checkpoint-qg'
peft_model = PeftModel.from_pretrained(t5model, peft_path).to(device)
peft_model.to(device)

app = Flask(__name__)

def generate_qa(paragraph, lqc, lac, coqc, coac, mpos) -> str:
    task_prefix = "Question generation: "
    control_qg = f"LQC {lqc} LAC {lac} COQC {coqc} COAC {coac} POS {mpos}"
    input_text = f"{task_prefix} {control_qg} {SEP_TOKEN} {paragraph}"
    
    source_encoding = tokenizer(
        input_text,
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
        num_beams=10,
        temperature=1.5,
        repetition_penalty=2.5,
        top_p=0.95,
        do_sample=True,
        max_length=TARGET_MAX_TOKEN_LEN,
        early_stopping=True,
        use_cache=True
    )

    preds = {
        tokenizer.decode(generated_id, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        for generated_id in generated_ids
    }

    result = ' '.join(preds)
    result = result.replace('<pad>', '')
    result = result.replace('</s>', '')
    result = result.strip()
    list_result = result.split(';')
    question = 'question'
    answer = 'answer'
    if len(list_result) < 2:
        return question, answer
    question = result.split(';')[0]
    answer = result.split(';')[1]
    return question.strip(), answer.strip()


def generate(context):
    qa_pairs = []
    mean_values = {
        'lqc': 0.097,
        'lac': 0.030,
        'mpos': 0.435
    }

    # Create ranges near the mean values
    stretched_range = lambda mean: np.linspace(mean * 0.65, mean * 1.35, 3).round(3).tolist()

    lqc_range = stretched_range(mean_values['lqc'])
    lac_range = stretched_range(mean_values['lac'])
    mpos_range = stretched_range(mean_values['mpos'])
    coqc = 0.047
    coac = 0.026
    
    total_iterations = len(lqc_range) * len(lac_range) * len(mpos_range)
    
    with tqdm(total=total_iterations, desc="Generating QA pairs") as pbar:
        for lqc in lqc_range:
            for lac in lac_range:
                for mpos in mpos_range:
                    question, answer = generate_qa(context, lqc, lac, coqc, coac, mpos)
                    qa_pairs.append({'question': question, 'answer': answer})
                    pbar.update(1)
                    
    return qa_pairs


if __name__ == "__main__": 
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
        They find the Brazil nuts which grow on the forest trees.
        '''
    qa_pairs = generate(context)
    for qa_pair in qa_pairs:
        print(qa_pair)