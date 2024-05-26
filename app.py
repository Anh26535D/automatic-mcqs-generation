from flask import Flask, request, jsonify, render_template
import requests

app = Flask(__name__)

QG_SERVICE_URL = 'http://localhost:8001/generate_qa'
DG_SERVICE_URL = 'http://localhost:8002/generate_distractors'


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_mcq', methods=['POST'])
def generate_mcq():
    data = request.json
    context = data.get('context')
    
    qg_payload = {'context': context}
    qg_response = requests.post(QG_SERVICE_URL, json=qg_payload)
    if qg_response.status_code != 200:
        return jsonify({'error': 'Failed to generate questions'}), 500
    
    questions = qg_response.json()
    
    mcqs = []
    for question_data in questions:
        question = question_data['question']
        answer = question_data['answer']
        type = question_data['type']

        dg_payload = {
            'context': context,
            'question': question,
            'answer': answer
        }
        dg_response = requests.post(DG_SERVICE_URL, json=dg_payload)
        if dg_response.status_code != 200:
            return jsonify({'error': 'Failed to generate distractors'}), 500
        
        distractors = dg_response.json().get('distractors')
        mcq = {
            'question': question,
            'answer': answer,
            'distractors': distractors,
            'type': type
        }
        mcqs.append(mcq)
    
    return jsonify({'mcqs': mcqs})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)