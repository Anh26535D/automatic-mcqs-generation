import re
from typing import List

from QDeconstructor import QDeconstructionResult
import utils

class QConstructor:
    def __init__(self, original_text, doc, pos_tags):
        self.original_text = original_text
        self.doc = doc
        self.pos_tags = pos_tags

    def postProcess(self, text: str):
        postProcessTextArr = self.original_text.split(' ')
        lowerCasedWord = postProcessTextArr[0][0].lower() + postProcessTextArr[0][1:]

        for ent in self.doc.ents:
            if (ent.text.find(postProcessTextArr[0]) != -1) and (ent.label_ in ['PERSON', 'FACILITY', 'GPE', 'ORG']):
                lowerCasedWord = postProcessTextArr[0]

        if lowerCasedWord == 'i': 
            lowerCasedWord = 'I'
        #Postprocess stage for lower casing common nouns, omitting extra spaces and dots
        formatted_text = text.replace(postProcessTextArr[0], lowerCasedWord)
        formatted_text = ' '.join(formatted_text.split())
        formatted_text = formatted_text.replace(' ,', ',')
        formatted_text = formatted_text.replace(" 's " , "'s ")
        formatted_text = formatted_text.replace("s ' " , "s' ")
        quotatedString = re.findall('"([^"]*)"', formatted_text)
        quotatedOrgString = re.findall('"([^"]*)"', formatted_text)
        for l in range(len(quotatedString)):
            if quotatedString[l][0] == " ": 
                quotatedString[l] = quotatedString[l][1:]
            if quotatedString[l][-1] == " ":
                quotatedString[l] = quotatedString[l][:-1]
            formatted_text = formatted_text.replace(quotatedOrgString[l], quotatedString[l])

        formatted_text = formatted_text.strip()
        if formatted_text.endswith('.') or formatted_text.endswith(','):
            formatted_text = formatted_text[:-1]
        return formatted_text     
    
    def constructQuestion(
            self, 
            deconstruction_results: List[QDeconstructionResult], 
            verbose: bool = False):
        qa_pairs = []
        for deconstruction_result in deconstruction_results:
            if verbose:
                print('---- With deconstruction results : ----')
                print('Subject : ' + deconstruction_result.subject)
                print('Predicate : ' + deconstruction_result.predicate)
                print('Object : ' + deconstruction_result.object)
                print('Extra Field : ' + deconstruction_result.extraField)
                print('Type : ' + deconstruction_result.type)
                print ('===> Generate raw question: ', end='')
            predArr = deconstruction_result.predicate.split(' ')
            negativePart = ''
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
            for idx, predicate in enumerate(predArr):
                if (predicate == 'and'): 
                    having_word_and = True
                for k in range(len(self.pos_tags)):
                    if (self.pos_tags[k][0] == predicate):
                        if (self.pos_tags[k][0] == 'to'):
                            having_word_to = True
                        if (self.pos_tags[k][1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD']):
                            if (numOfVerbs == 0):
                                firstFoundVerbIndex = idx
                            numOfVerbs = numOfVerbs + 1
                            break
                        if (self.pos_tags[k][1] == 'RB') and (self.pos_tags[k][0].lower() == 'not'):
                            if (numOfVerbs == 0):
                                firstFoundVerbIndex = idx
                            numOfVerbs = numOfVerbs + 1
                            negativeIndex = idx
                            break
            if (not having_word_and):
                if (negativeIndex > -1):
                    negativePart = predArr.pop(negativeIndex)
                if (numOfVerbs == 1) or (having_word_to):
                    if (predArr[0] not in ['am', 'is', 'are', 'was', 'were']):
                        for k in range (len(self.pos_tags)):
                            predArrNew = []
                            if self.pos_tags[k][0] == predArr[firstFoundVerbIndex]:
                                predArrNew = []
                                if self.pos_tags[k][1] == 'MD':
                                    break
                                elif self.pos_tags[k][1] == 'VBG': 
                                    deconstruction_result.type = '' 
                                    break
                                elif self.pos_tags[k][1] == 'VBZ':
                                    predArrNew.append('does')
                                    if self.pos_tags[k][0] == 'has':
                                        predArrNew.append(self.pos_tags[k][0])
                                    else:
                                        predArrNew.append(utils.lemmatizeVerb(self.pos_tags[k][0]))
                                elif self.pos_tags[k][1] == 'VBP':
                                    predArrNew = []
                                    predArrNew.append('do')
                                    predArrNew.append(self.pos_tags[k][0])
                                elif self.pos_tags[k][1] == 'VBD' or self.pos_tags[k][1] == 'VBN':
                                    predArrNew = []
                                    predArrNew.append('did')
                                    predArrNew.append(utils.lemmatizeVerb(self.pos_tags[k][0]))
                                else:
                                    predArrNew = []
                                    subjectParts = deconstruction_result.subject.split(' ')
                                    isFound = False
                                    for l in range (len(self.pos_tags)):
                                        if isFound: 
                                            break
                                        for m in range(len(subjectParts)):
                                            if self.pos_tags[l][0] == subjectParts[m]:
                                                if self.pos_tags[l][1] == 'NN':
                                                    predArrNew.append('does')
                                                    predArrNew.append(self.pos_tags[k][0])
                                                    isFound = True
                                                    break
                                                elif self.pos_tags[l][1] == 'NNS':
                                                    predArrNew.append('do')
                                                    predArrNew.append(self.pos_tags[k][0])
                                                    isFound = True
                                                    break
                                            if (l == len(self.pos_tags)-1) and (m == len(subjectParts)-1):
                                                predArrNew.append('do')
                                                predArrNew.append(self.pos_tags[k][0])
                                                isFound = True
                                                break
                                predArr.pop(firstFoundVerbIndex)
                                predArrTemp = predArr
                                predArr = predArrNew + predArrTemp
                                break
                if numOfVerbs == 0 and len(predArr) == 1 and deconstruction_result.type != 'attr':
                    mainVerb = predArr[0]
                    qVerb = ''
                    if utils.lemmatizeVerb(predArr[0]) == predArr[0]:
                        qVerb = 'do'
                    else:
                        qVerb = 'does'
                    predArr = []
                    predArr.append(qVerb)
                    predArr.append(mainVerb)

            if having_word_and: 
                predArr.insert(0, '')
            isQuestionMarkExist = False
            verbRemainingPart = ''
            question = ''
            answer = deconstruction_result.key_answer
            for k in range (1, len(predArr)):
                verbRemainingPart = verbRemainingPart + ' ' + predArr[k]

            if deconstruction_result.type == 'dative':
                question = 'What ' + predArr[0] + ' ' + negativePart + ' ' + verbRemainingPart + ' ' + deconstruction_result.object + ' ' + deconstruction_result.extraField
                answer = deconstruction_result.subject
            elif deconstruction_result.type == 'dobj' or deconstruction_result.type == 'pcomp':
                whQuestion = 'What '
                for ent in self.doc.ents:
                    if (ent.text == deconstruction_result.object and ent.label_ == 'PERSON'):
                        whQuestion = 'Who '
                        break
                question = whQuestion + predArr[0] + ' ' + negativePart + ' ' + deconstruction_result.subject + verbRemainingPart  + ' ' + deconstruction_result.extraField
                answer = deconstruction_result.object
            elif deconstruction_result.type == 'DATE':
                question = 'When '+ predArr[0] + ' ' + negativePart + ' ' + deconstruction_result.subject + verbRemainingPart + ' ' + deconstruction_result.object  + ' ' + deconstruction_result.extraField
            elif deconstruction_result.type == 'ARG_TEMPORAL_TYPE':
                question = 'When '+ predArr[0] + ' ' + negativePart + ' ' + deconstruction_result.subject + verbRemainingPart + ' ' + deconstruction_result.object  + ' ' + deconstruction_result.extraField
            elif deconstruction_result.type == 'LOC':
                question = 'Where '+predArr[0] + ' ' + negativePart + ' ' + deconstruction_result.subject + verbRemainingPart + ' ' + deconstruction_result.object  + ' ' + deconstruction_result.extraField
            elif deconstruction_result.type == 'CARDINAL':
                question = 'How many ' + deconstruction_result.subject + ' ' + predArr[0] + ' ' + negativePart + ' ' + deconstruction_result.extraField  + ' ' + verbRemainingPart + ' ' + deconstruction_result.object
            elif deconstruction_result.type == 'attr':
                question = 'How would  '+ deconstruction_result.subject + ' ' + predArr[0] + ' ' + negativePart + ' ' + verbRemainingPart + ' ' + deconstruction_result.object
            elif deconstruction_result.type == 'PERSON':
                if deconstruction_result.object.endswith('.'):
                    deconstruction_result.object = deconstruction_result.object[:-1]
                question = 'Who  ' + predArr[0] + ' ' + negativePart + ' ' + verbRemainingPart + ' ' + deconstruction_result.object + ' ' + deconstruction_result.extraField
            elif deconstruction_result.type == 'WHAT':
                if deconstruction_result.object.endswith('.'):
                    deconstruction_result.object = deconstruction_result.object[:-1]
                question = 'What  ' + predArr[0] + ' ' + negativePart + ' ' + verbRemainingPart + ' ' + deconstruction_result.object + ' ' + deconstruction_result.extraField
            elif deconstruction_result.type == 'acomp':
                isQuestionMarkExist = True  
                question = 'Indicate characteristics of ' + utils.getObjectPronun(deconstruction_result.subject)
            elif deconstruction_result.type == 'direct':
                predArr[0]= predArr[0][:1].upper() + predArr[0][1:]
                if deconstruction_result.object.endswith('.'):
                    deconstruction_result.object = deconstruction_result.object[:-1]
                question = predArr[0] + ' ' + deconstruction_result.subject + ' ' + verbRemainingPart + ' ' + deconstruction_result.object  + ' ' + deconstruction_result.extraField
                answer = 'Yes' if not negativePart else 'No'
            elif deconstruction_result.type == 'why':
                question = 'Why '+predArr[0] + ' ' + negativePart + ' ' + deconstruction_result.subject + verbRemainingPart + ' ' + deconstruction_result.object  + ' ' + deconstruction_result.extraField
            elif deconstruction_result.type == 'purpose':
                question = 'For what purpose '+predArr[0] + ' ' + negativePart + ' ' + deconstruction_result.subject + verbRemainingPart + ' ' + deconstruction_result.object  + ' ' + deconstruction_result.extraField
            elif deconstruction_result.type == 'ARG_MANNER_TYPE':
                question = 'How '+ predArr[0] + ' ' + negativePart + ' ' + deconstruction_result.subject + verbRemainingPart + ' ' + deconstruction_result.object  + ' ' + deconstruction_result.extraField
   
            if verbose:
                print(question)

            formattedQuestion = self.postProcess(question)
            formattedAnswer = self.postProcess(answer)
            if formattedQuestion != '':
                if isQuestionMarkExist == False:
                    formattedQuestion = formattedQuestion + '?'
                else: 
                    formattedQuestion = formattedQuestion + '.'
                if verbose:
                    print('===> Generate final question: ', formattedQuestion)
                if formattedAnswer != '':
                    formattedAnswer = formattedAnswer + '.'
                    formattedAnswer = formattedAnswer[0].upper() + formattedAnswer[1:]
                qa_pairs.append({
                    'question': formattedQuestion,
                    'answer': formattedAnswer
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