import os
import json
import re

import spacy
from allennlp.predictors import Predictor

from practnlptools.tools import Annotator
import utils
from QConstructor import QConstructor
from QDeconstructor import QDeconstructor


cur_dir = os.getcwd()

IDIOMS_PATH = os.path.join(cur_dir, 'utility_files', 'idioms.json')
CONTRACTIONS_PATH = os.path.join(cur_dir, 'utility_files', 'contractions.json')

SRL_MODEL_PATH = 'https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz'
SENNA_PATH = os.path.join(cur_dir, 'practnlptools')
PNTL_PATH = os.path.join(cur_dir, 'practnlptools')

contractions_dict = json.loads(open(CONTRACTIONS_PATH).read())
contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

def explandContractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)

def preprocess(text):
    text = re.sub(r'\([^)]*\)', '', text)
    text = ' '.join(text.split())
    text = text.replace(' ,', ',')
    p = re.compile(r'(?:(?<!\w)\'((?:.|\n)+?\'?)(?:(?<!s)\'(?!\w)|(?<=s)\'(?!([^\']|\w\'\w)+\'(?!\w))))')
    subst = "\"\g<1>\""
    text = re.sub(p, subst, text)
    text = explandContractions(text)
    return text

def generate(text, verbose=False):
    text = preprocess(text)
    textList = []
    textList.append(text)
    annotator = Annotator(
        SENNA_PATH, 
        PNTL_PATH, 
        "edu.stanford.nlp.trees."
    )

    try:
        posTags = annotator.get_annotations(textList, dep_parse=False)['pos']
        chunks = annotator.get_annotations(textList, dep_parse=False)['chunk']
    except IndexError:
        emptyList = []
        return emptyList
    
    predictor = Predictor.from_path(SRL_MODEL_PATH)
    srlResult = predictor.predict_json({"sentence": text})
    srls = []
    try:
        for i in range(len(srlResult['verbs'])):
            current_srl = {}
            description = srlResult['verbs'][i]['description']
            while (True):
                value = utils.getValueBetweenTexts(description, "[", "]")
                if (value == ""): 
                    break
                else:
                    parts = value.split(": ")
                    current_srl[parts[0]] = parts[1]
                    description = description.replace("["+value+"]", "")
            if (current_srl):
                srls.append(current_srl)
    except IndexError:
        emptyList = []
        return emptyList

    nlp = spacy.load('en_core_web_sm')
    doc = nlp(u''+text)

    if verbose:
        print('[POS_GQ] POS Tags:', posTags)
        print('[CHUNKS_GQ] Chunks:', chunks)
        print('[SRL_GQ] SRLs:')
        for srl in srls:
            print(srl)

    print('[SRL_GQ] Start processing...')
    print('*'*50)
    ##################################################################################
    ###############################  Annotation Stage  ###############################
    ################################################################################## 

    dativeWord = []
    dativeVerb = []
    dativeSubType = []
    dobjWord = []
    dobjVerb = []
    dobjSubType = []
    acompWord = []
    acompVerb = []
    acompSubType = []
    attrWord = []
    attrVerb = []
    attrSubType = []
    pcompPreposition = []
    pcompWord = []
    pcompSubType = []
    dateWord = []
    dateSubType = []
    numWord = []
    numSubType = []
    personWord = []
    personSubType = []
    whereWord = []
    whereSubType = []
    idiom_dict = json.loads(open(IDIOMS_PATH).read())
    for word in doc:
        if (word.dep_ in ['dobj', 'ccomp', 'xcomp', 'dative', 'acomp', 'attr', 'oprd']):
            baseFormVerb = utils.lemmatizeVerb(word.head.text)
            if (baseFormVerb in idiom_dict.keys()) and (idiom_dict[baseFormVerb] in word.text): 
                continue
        #what question / who question
        if (word.dep_ in ['dobj', 'ccomp', 'xcomp']):
            dobjVerb.append(word.head.text)
            dobjWord.append(word.text)
            dobjSubType.append('dobj')
        if (word.dep_ == 'dative'):
            dativeVerb.append(word.head.text)
            dativeWord.append(word.text)
            dativeSubType.append('dative')
        if (word.dep_ == 'acomp'):
            acompVerb.append(word.head.text)
            acompWord.append(word.text)
            acompSubType.append('acomp')
        if (word.dep_ == 'attr') or (word.dep_ == 'oprd'):
            attrVerb.append(word.head.text)
            attrWord.append(word.text)
            attrSubType.append('attr')
        #what question
        if (word.dep_ == 'pcomp'):
            pcompPreposition.append(word.head.text)
            pcompWord.append(word.text)
            pcompSubType.append('pcomp')

    for ent in doc.ents:
        #when question
        if (ent.label_ == 'DATE') and ('year old' not in ent.text) and ('years old' not in ent.text):
            dateWord.append(ent.text)
            dateSubType.append(ent.label_)
        #how many question
        if (ent.label_ == 'CARDINAL'):
            numWord.append(ent.text)
            numSubType.append(ent.label_)
        #who question
        if (ent.label_ == 'PERSON'):
            personWord.append(ent.text)
            personSubType.append(ent.label_)
        # Where question, including Location (LOC), Facility (FACILITY), Organization (ORG), and Geopolitical Entity (GPE)
        if (ent.label_ in ['FACILITY', 'ORG', 'GPE', 'LOC']):
            whereWord.append(ent.text)
            whereSubType.append('LOC')

    ##################################################################################
    #############################  Deconstruction Stage  #############################
    ##################################################################################
    qdeconstructor = QDeconstructor(doc, srls, chunks)
    qdeconstruct_result = qdeconstructor.deconstruct(
        dativeWord, dativeVerb, dativeSubType, 
        dobjWord, dobjVerb, dobjSubType, 
        acompWord, acompVerb, acompSubType, 
        attrWord, attrVerb, attrSubType, 
        dateWord, dateSubType, 
        whereWord, whereSubType, 
        pcompWord, pcompPreposition, pcompSubType, 
        numWord, numSubType, 
        personWord, personSubType
    )

    ##################################################################################
    ##############################  CONSTRUCTION STAGE  ##############################
    ##################################################################################  
    question_constructor = QConstructor(text, doc, posTags)
    found_questions = question_constructor.constructQuestion(qdeconstruct_result, True)
    return found_questions
    
if __name__ == "__main__":
    text = 'He studied hard during his college years and in the months.'
    qa_pairs = generate(text, True)
    print('[SRL_GQ] End processing...')
    print('*'*50)
    for idx, qa in enumerate(qa_pairs):
        print('========> QA Pair: ', idx)
        print('question: ', qa['question'])
        print('answer: ', qa['answer'])