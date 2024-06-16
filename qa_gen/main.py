import os
import json
import re

import warnings
warnings.filterwarnings("ignore")

import spacy
from allennlp.predictors import Predictor
from fastcoref import spacy_component

from QConstructor import QConstructor
from QDeconstructor import QDeconstructor


cur_dir = os.getcwd()

CONTRACTIONS_PATH = os.path.join(cur_dir, 'utility_files', 'contractions.json')

DISTRACTOR_MODEL_PATH = 'voidful/bart-distractor-generation-both'
SRL_MODEL_PATH = 'https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz'

contractions_dict = json.loads(open(CONTRACTIONS_PATH).read())
contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe("fastcoref")
predictor = Predictor.from_path(SRL_MODEL_PATH)

def explandContractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)


def clean_text(text: str):
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('“', '"')
    text = text.replace('”', '"')
    text = text.replace("’", "'")
    return text


def generate(text: str, verbose=False):
    text = explandContractions(text)
    textList = []
    textList.append(text)
    
    doc = nlp(u''+text, component_cfg={"fastcoref": {'resolve_text': True}})  # See https://github.com/shon-otmazgin/fastcoref
    
    
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
    
    question_constructor = QConstructor(doc, srls, 2)
    found_questions = question_constructor.constructQuestion(qdeconstruct_result, True)
    
    return found_questions


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
    They find and sell the Brazil nuts which grow on the forest trees.
    '''
    context = clean_text(context)
    from helper import Helper
    Helper.visualize_dependencies(context)
    raise 1
    qa_pairs = generate(context, True)
    debug_file = 'debug.txt'
    with open(debug_file, 'w') as f:
        f.writelines(f'Context: {context}\n')
        f.writelines(f'Number of QA pairs: {len(qa_pairs)}\n')
        for idx, qa in enumerate(qa_pairs):
            f.writelines(f'========> QA-{idx} =================\n')
            f.writelines(f'Original: {qa["original"]}\n')
            f.writelines(f'Question: {qa["question"]}\n')
            f.writelines(f'Answer: {qa["answer"]}\n')
            f.writelines(f'Type: {qa["type"]}\n')
