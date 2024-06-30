import requests
from tqdm import tqdm
import logging
import os

# Set up logging
logging.basicConfig(filename='mcq_generation.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

PARAPHRASE_SERVICE_URL = 'http://localhost:7999/paraphrase'
T5QG_SERVICE_URL = 'http://localhost:8000/generate_t5_qa'
QG_SERVICE_URL = 'http://localhost:8001/generate_qa'
DG_SERVICE_URL = 'http://localhost:8002/generate_distractors'

def clean_text(text: str):
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('“', '"')
    text = text.replace('”', '"')
    text = text.replace("’", "'")
    return text

def check_service(url, payload):
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            logging.error(f"Service at {url} returned error: {response.status_code}")
            return None
    except Exception as e:
        logging.error(f"Failed to connect to service at {url}: {str(e)}")
        return None

def generate_mcq(context, output_path):
    context = clean_text(context)
    # clear file content
    with open('generated_mcqs.txt', 'w', encoding='utf-8') as file:
        file.write('')
    
    qg_payload = {
        'context': context,
        'enhance_level': 2,
        'limit': 50,
        }
    rule_qg_response = check_service(QG_SERVICE_URL, qg_payload)
    t5qg_response = check_service(T5QG_SERVICE_URL, qg_payload)
    
    if rule_qg_response is None or t5qg_response is None:
        raise Exception('Failed to generate questions')

    questions = []
    for res in rule_qg_response:
        questions.append(res)
    print("Found questions from rule based generator: ", len(rule_qg_response))
    for res in t5qg_response:
        questions.append({
            'question': res['question'],
            'answer': res['answer'],
            'type': 't5'
        })
    print("Found questions from T5 based generator: ", len(t5qg_response))
    
    # remove duplicate questions
    unique_questions = []
    unique_qas = []
    for question in questions:
        if question['question'] not in unique_questions:
            unique_questions.append(question['question'])
            unique_qas.append(question)

    mcqs = []
    print("Found unique questions: ", len(unique_qas))
    for question_data in tqdm(unique_qas, total=len(unique_qas)):
        question = question_data['question']
        answer = question_data['answer']
        q_type = question_data['type']
        
        paraphrase_payload = {
            'question': question
        }
        paraphrase_response = check_service(PARAPHRASE_SERVICE_URL, paraphrase_payload)
        if paraphrase_response is None:
            raise Exception('Failed to paraphrase question')
        
        paraphrased_question = paraphrase_response['paraphrased_texts'][0]
        
        # only paraphrase the answer if it is long enough
        if len(answer) < 10:
            paraphrased_answer = answer
        else:
            paraphrase_payload = {
                'question': answer
            }
            paraphrase_response = check_service(PARAPHRASE_SERVICE_URL, paraphrase_payload)
            if paraphrase_response is None:
                raise Exception('Failed to paraphrase answer')        
            paraphrased_answer = paraphrase_response['paraphrased_texts'][0]
            
        dg_payload = {
            'context': context,
            'question': paraphrased_question,
            'answer':  paraphrased_answer       
        }
        dg_response = check_service(DG_SERVICE_URL, dg_payload)
        if dg_response is None:
            raise Exception('Failed to generate distractors')
        
        distractors = dg_response.get('distractors')
        mcq = {
            'original question': question,
            'original answer': answer,
            'question': paraphrased_question,
            'answer': paraphrased_answer,
            'distractors': distractors,
            'type': q_type
        }
        mcqs.append(mcq)
        
        with open(output_path, 'a', encoding='utf-8') as file:
            file.write(f"Original Question: {mcq['original question']}\n")
            file.write(f"Original Answer: {mcq['original answer']}\n")
            file.write(f"Question: {mcq['question']}\n")
            file.write(f"A: {mcq['answer']}\n")
            file.write(f"B: {mcq['distractors'][0]}\n")
            file.write(f"C: {mcq['distractors'][1]}\n")
            file.write(f"D: {mcq['distractors'][2]}\n")
            file.write(f"Type: {mcq['type']}\n\n")
    
    return mcqs

if __name__ == '__main__':
    context1 = """The Lobund Institute grew out of pioneering research in germ-free-life which began in 1928. This area of research originated in a question posed by Pasteur as to whether animal life was possible without bacteria. Though others had taken up this idea, their research was short lived and inconclusive. Lobund was the first research organization to answer definitively, that such life is possible and that it can be prolonged through generations. But the objective was not merely to answer Pasteur's question but also to produce the germ free animal as a new tool for biological and medical research. This objective was reached and for years Lobund was a unique center for the study and production of germ free animals and for their use in biological and medical investigations. Today the work has spread to other universities. In the beginning it was under the Department of Biology and a program leading to the master's degree accompanied the research program. In the 1940s Lobund achieved independent status as a purely research organization and in 1950 was raised to the status of an Institute. In 1958 it was brought back into the Department of Biology as integral part of that department, but with its own program leading to the degree of PhD in Gnotobiotics."""
    # context2 = """The Yanomami live along the rivers of the rainforest in the north of Brazil. They have lived in the rainforest for about 10,000 years and they use more than 2,000 different plants for food and for medicine. But in 1988, someone found gold in their forest, and suddenly 45,000 people came to the forest and began looking for gold. They cut down the forest to make roads. They made more than a hundred airports. The Yanomami people lost land and food. Many died because new diseases came to the forest with the strangers.In 1987, they closed fifteen roads for eight months. No one cut down any trees during that time. In Panama, the Kuna people saved their forest. They made a forest park which tourists pay to visit. The Gavioes people of Brazil use the forest, but they protect it as well. They find the Brazil nuts which grow on the forest trees."""
    contexts = [context1]
    for i, context in enumerate(contexts):     
        save_path = os.path.join(os.getcwd(), f'context{i}.txt')
        with open(save_path, 'w', encoding='utf-8') as file:
            file.write(f"{context}\n")
            
        try:
            mcqs = generate_mcq(context, save_path)
            print("MCQ generation completed successfully.")
        except Exception as e:
            logging.error(f"Error during MCQ generation: {str(e)}")
