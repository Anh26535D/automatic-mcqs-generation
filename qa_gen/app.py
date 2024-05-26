import os
import json
import re
import warnings

from flask import Flask, request, jsonify
import spacy
from allennlp.predictors import Predictor
from fastcoref import spacy_component
from QConstructor import QConstructor
from QDeconstructor import QDeconstructor

warnings.filterwarnings("ignore")

app = Flask(__name__)

cur_dir = os.getcwd()
CONTRACTIONS_PATH = os.path.join(cur_dir, 'utility_files', 'contractions.json')
SRL_MODEL_PATH = 'https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz'

contractions_dict = json.loads(open(CONTRACTIONS_PATH).read())
contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe("fastcoref")
predictor = Predictor.from_path(SRL_MODEL_PATH)
    
def expandContractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)

def generate(text: str, verbose=False):
    text = expandContractions(text)
    textList = []
    textList.append(text)

    doc = nlp(u'' + text, component_cfg={"fastcoref": {'resolve_text': True}})
    
    srls = []
    for sent in doc.sents:
        sent_start = sent.start_char
        sent_text = sent.text
        srlResult = predictor.predict_json({"sentence": sent_text})
        if verbose:
            print('--> SRL Result:', srlResult)
        words = srlResult['words']
        word_offsets = []
        current_offset = 0
        
        for word in words:
            start = sent_text.find(word, current_offset)
            if start == -1:
                continue
            end = start + len(word)
            word_offsets.append((start + sent_start, end + sent_start))
            current_offset = end

        verbs = srlResult['verbs']
        for verb in verbs:
            tags = verb['tags']
            tag_offsets = {}
            for tag, word_offset in zip(tags, word_offsets):
                if tag == 'O':
                    continue
                if tag[2:] not in tag_offsets.keys():
                    tag_offsets[tag[2:]] = [word_offset]
                else:
                    tag_offsets[tag[2:]].append(word_offset)
            
            mod_tag_offsets = {}
            for tag, offsets in tag_offsets.items():
                min_offset = min(offset[0] for offset in offsets)
                max_offset = max(offset[1] for offset in offsets)
                mod_tag_offsets[tag] = (min_offset, max_offset)
            srls.append(mod_tag_offsets)

    if verbose:
        print('[SRL_GQ] SRLs:')
        for srl in srls:
            print('-'*50)
            for key, value in srl.items():
                print(f'{key}: {text[value[0]:value[1]]}')

    qdeconstructor = QDeconstructor(doc, srls)
    qdeconstruct_result = qdeconstructor.deconstruct()
    
    question_constructor = QConstructor(doc)
    found_questions = question_constructor.constructQuestion(qdeconstruct_result, True)
    
    return found_questions

@app.route('/generate_qa', methods=['POST'])
def generate_qa():
    data = request.json
    text = data.get('context', None)
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        qa_pairs = generate(text, verbose=False)
        return jsonify(qa_pairs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
