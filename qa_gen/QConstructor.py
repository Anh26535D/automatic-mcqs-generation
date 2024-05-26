from typing import List

from QDeconstructor import QDeconstructionResult
from helper import Helper
import utils

class QConstructor:
    
    def __init__(self, doc):
        self.doc = doc

    
    def constructQuestion(
            self, 
            deconstruction_results: List[QDeconstructionResult], 
            verbose: bool = False):
        qa_pairs = []
        for deconstruction_result in deconstruction_results:
            if verbose:
                print("===> Deconstruction result: ")
                print(deconstruction_result)
            predicate_text = Helper.merge_tokens(deconstruction_result.predicate)
            
            predicate = deconstruction_result.predicate.copy()
            
            negativeIndex = -1
            numOfVerbs = 0
            firstFoundVerbIndex = -1
            having_word_to = False
            having_word_and = False
            # VB: Verb, base form
            # VBD: Verb, past tense
            # VBG: Verb, gerund or present participle
            # VBN: Verb, past participle
            # VBP: Verb, non-3rd person singular present
            # VBZ: Verb, 3rd person singular present
            # MD: Modal
            # RB: Adverb
            for idx, tok in enumerate(predicate):
                if (tok.text == 'and'): 
                    having_word_and = True
                if (tok.text == 'to'):
                    having_word_to = True
                if (tok.tag_ in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD']):
                    if (numOfVerbs == 0):
                        firstFoundVerbIndex = idx
                    numOfVerbs = numOfVerbs + 1
                if (tok.tag_ == 'RB') and (tok.text.lower() == 'not'):
                    if (numOfVerbs == 0):
                        firstFoundVerbIndex = idx
                    numOfVerbs = numOfVerbs + 1
                    negativeIndex = idx
            aux_text = ''
            root_verb = ''
            predArr = predicate
            predicate_strs = Helper.merge_tokens(predArr).split()
            negativePart = ''

            if (not having_word_and):
                if (negativeIndex > -1):
                    negativePart = predArr.pop(negativeIndex).text
                if (numOfVerbs == 1) or (having_word_to):
                    if (predArr[0].text not in ['am', 'is', 'are', 'was', 'were']):
                        word = predArr[firstFoundVerbIndex].text
                        tag = predArr[firstFoundVerbIndex].tag_
                        if (tag == 'MD'): # modal verb
                            pass
                        elif (tag == 'VBG'): # gerund
                            deconstruction_result.type = ''
                        elif (tag == 'VBZ'): # 3rd person singular present
                            aux_text = 'does'
                            if word == 'has':
                                root_verb = word
                            else:
                                root_verb = utils.lemmatizeVerb(word)
                        elif tag == 'VBP': # non-3rd person singular present
                            aux_text = 'do'
                            root_verb = word
                        elif tag == 'VBD' or tag == 'VBN': # past tense or past participle
                            aux_text = 'did'
                            root_verb = utils.lemmatizeVerb(word)
                        else:
                            isFound = False
                            for l, tok in enumerate(self.doc):
                                if isFound: 
                                    break
                                for m in range(len(deconstruction_result.subject)):
                                    if tok == deconstruction_result.subject[m]:
                                        if tok.tag_ == 'NN': # singular noun
                                            aux_text = 'does'
                                            root_verb = word
                                            isFound = True
                                            break
                                        elif tok.tag_ == 'NNS': # plural noun
                                            aux_text = 'do'
                                            root_verb = word
                                            isFound = True
                                            break
                                    if (l == len(self.doc)-1) and (m == len(deconstruction_result.subject)-1):
                                        aux_text = 'do'
                                        root_verb = word
                                        isFound = True
                                        break
                        predArr.pop(firstFoundVerbIndex)
                        predicate_strs = [root_verb] + Helper.merge_tokens(predArr).split()
                if numOfVerbs == 0 and len(predArr) == 1 and deconstruction_result.type != 'attr':
                    mainVerb = predArr[0].text
                    if utils.lemmatizeVerb(predArr[0].text) == predArr[0].text:
                        aux_text = 'do'
                    else:
                        aux_text = 'does'
                    predicate_strs = [mainVerb]
                if numOfVerbs > 1: # 
                    word = predArr[firstFoundVerbIndex].text
                    if word in ['am', 'is', 'are', 'was', 'were', 'has', 'have', 'had']:
                        aux_text = word
                        predArr.pop(firstFoundVerbIndex)
                        if not isinstance(predArr, list):
                            predArr = list(predArr)
                        predicate_strs = Helper.merge_tokens(predArr).split()
                    
            if having_word_and: 
                predicate_strs.insert(0, '')
            isQuestionMarkExist = False
            verbRemainingPart = Helper.merge_strs(predicate_strs)
            question = ''
            
            answer = Helper.merge_tokens(deconstruction_result.key_answer)
            subject_text = Helper.merge_tokens(deconstruction_result.subject)
            object_text = Helper.merge_tokens(deconstruction_result.object)
            extra_text = Helper.merge_tokens(deconstruction_result.extra_field)
            type_text = deconstruction_result.type

            if type_text == 'dative_question':
                whQuestion = 'What '
                for ent in self.doc.ents:
                    if (ent.text == answer and ent.label_ == 'PERSON'):
                        whQuestion = 'Whom '
                        break
                question = whQuestion + aux_text + ' ' + subject_text + ' ' + negativePart + ' ' + verbRemainingPart + ' ' + object_text + ' ' + extra_text
            elif type_text == 'dobj_question' or type_text == 'pcomp_question':
                whQuestion = 'What '
                for ent in self.doc.ents:
                    if (ent.text == object_text and ent.label_ == 'PERSON'):
                        whQuestion = 'Who '
                        break
                question = whQuestion + aux_text + ' ' + negativePart + ' ' + subject_text + ' ' + verbRemainingPart  + ' ' + extra_text
            elif type_text == 'attr_question':
                question = 'How would you describe ' + object_text
            elif type_text == 'acomp_question':
                isQuestionMarkExist = True  
                question = 'Indicate characteristics of ' + utils.getObjectPronun(subject_text)
            elif type_text == 'nsubj_question':
                question_word = 'What '
                for ent in self.doc.ents:
                    if (ent.text == answer and ent.label_ == 'PERSON'):
                        question_word = 'Who '
                question = question_word + predicate_text + ' ' + object_text + ' ' + extra_text
            elif type_text == 'direct':
                aux_text= aux_text[:1].upper() + aux_text[1:]
                if object_text.endswith('.'):
                    object_text = object_text[:-1]
                question = aux_text + ' ' + subject_text + ' ' + verbRemainingPart + ' ' + object_text  + ' ' + extra_text
                answer = 'Yes' if not negativePart else 'No'
            elif type_text == 'srl_causal':
                question = 'Why '+ aux_text + ' ' + negativePart + ' ' + subject_text + ' ' + verbRemainingPart + ' ' + object_text  + ' ' + extra_text
            elif type_text == 'srl_purpose':
                question = 'What '+ aux_text + negativePart + ' the purpose of ' + subject_text + ' ' + verbRemainingPart + ' ' + object_text  + ' ' + extra_text
            elif type_text == 'srl_manner':
                question = 'How '+ aux_text + ' ' + negativePart + ' ' + subject_text + ' ' + verbRemainingPart + ' ' + object_text  + ' ' + extra_text
            elif type_text == 'srl_temporal':
                question = 'When '+ aux_text + ' ' + negativePart + ' ' + subject_text + ' ' + verbRemainingPart + ' ' + object_text  + ' ' + extra_text
            elif type_text == 'srl_locative':
                question = 'Where '+aux_text + ' ' + negativePart + ' ' + subject_text + ' ' + verbRemainingPart + ' ' + object_text  + ' ' + extra_text
            elif type_text == 'ner_date_question':
                question = 'When '+ aux_text + ' ' + negativePart + ' ' + subject_text + ' ' + verbRemainingPart + ' ' + object_text  + ' ' + extra_text
            elif type_text == 'ner_loc_question':
                question = 'Where '+aux_text + ' ' + negativePart + ' ' + subject_text + ' ' + verbRemainingPart + ' ' + object_text  + ' ' + extra_text
            elif type_text == 'ner_cardinal_question':
                question = 'How many ' + subject_text + ' ' + aux_text + ' ' + negativePart + ' ' + extra_text  + ' ' + verbRemainingPart + ' ' + object_text
            elif type_text == 'ner_person_question':
                if object_text.endswith('.'):
                    object_text = object_text[:-1]
                question = 'Who ' + aux_text + ' ' + negativePart + ' ' + verbRemainingPart + ' ' + object_text + ' ' + extra_text
            
            words = question.split()
            formattedQuestion = ''
            for word_idx, word in enumerate(words):
                if word in ['He', 'She', 'It', 'They', 'We', 'In'] and word_idx != 0:
                    word = word.lower()
                if word in ['.', ',', '?', '!', ':', ';'] or word == "'s":
                    formattedQuestion = formattedQuestion + word
                else:
                    formattedQuestion = formattedQuestion + ' ' + word

            if formattedQuestion != '':
                if isQuestionMarkExist == False:
                    formattedQuestion = formattedQuestion + '?'
                else: 
                    formattedQuestion = formattedQuestion + '.'
                if verbose:
                    print('\tGenerate final question: ', formattedQuestion)
                qa_pairs.append({
                    'question': formattedQuestion,
                    'answer': answer,
                    'type': type_text,
                })

        # Remove duplicates
        unique_questions = []
        unique_answers = []
        unique_qa_pairs = []
        for i in range(len(qa_pairs)):
            if (qa_pairs[i]['question'] not in unique_questions) or (qa_pairs[i]['answer'] not in unique_answers):
                unique_questions.append(qa_pairs[i]['question'])
                unique_answers.append(qa_pairs[i]['answer'])
                unique_qa_pairs.append(qa_pairs[i])

        unique_qa_pairs.sort(key = lambda s: len(s['question']))
        return unique_qa_pairs