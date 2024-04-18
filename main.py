from __future__ import generators, print_function, unicode_literals
import json
import re
import os

import spacy
from allennlp.predictors import Predictor

from practnlptools.tools import Annotator

cur_dir = os.getcwd()

IDIOMS_PATH = os.path.join(cur_dir, 'utility_files', 'idioms.json')
CONTRACTIONS_PATH = os.path.join(cur_dir, 'utility_files', 'contractions.json')
OBJECT_PRONOUNS_PATH = os.path.join(cur_dir, 'utility_files', 'object_pronouns.txt')
VERBFORMS_PATH = os.path.join(cur_dir, 'utility_files', 'verb_forms.txt')

SRL_MODEL_PATH = 'https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz'
SENNA_PATH = os.path.join(cur_dir, 'practnlptools')
PNTL_PATH = os.path.join(cur_dir, 'practnlptools')

def findDependencyWord( strParam, orderNo ):
    if orderNo == 0:
        prm  = re.compile('\((.*?)-', re.DOTALL |  re.IGNORECASE).findall(strParam)
    elif orderNo == 1:
        prm  = re.compile(', (.*?)-', re.DOTALL |  re.IGNORECASE).findall(strParam)
    if prm :
        return prm[0]

def checkForAppropriateObjOrSub(srls,j,sType):
    if (sType == 0):
        for i in range(0,5):
            if 'ARG' + str(i) in srls[j].keys():
                return srls[j]['ARG' + str(i)]
    elif (sType == 1):
        foundIndex = 0
        for i in range(0,5):
            if 'ARG' + str(i) in srls[j].keys():
                foundIndex = foundIndex + 1
                if (foundIndex == 2):
                    return srls[j]['ARG' + str(i)]
    elif (sType == 2):
        foundIndex = 0
        for i in range(0,6):
            if 'ARG' + str(i) in srls[j].keys():
                foundIndex = foundIndex + 1
                if (foundIndex == 3):
                    return srls[j]['ARG' + str(i)]

    return ''

def getBaseFormOfVerb (verb):
    #return lemma(verb)
    #todo: pattern no longer working use another library!
    with open(VERBFORMS_PATH, 'r') as myfile:
        verb = verb.lower()
        f = myfile.read()
        oldString = find_between(f, "", "| "+ verb +" ")
        oldString = oldString + '|'
        k = oldString.rfind("<")
        newString = oldString[:k] + "_" + oldString[k+1:]
        if find_between(newString, "_ ", " |") == '':
            if find_between(f, "< "+verb+" "," >") == '':
                with open('logs.txt', "a") as logFile:
                    logFile.write('Warning! Base form of verb '+verb+ ' not found\n')
                    print ('Warning! Base form of verb '+verb+ ' not found')
            return verb

        return find_between(newString, "_ ", " |")

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def getObjectPronun(text):
    with open(OBJECT_PRONOUNS_PATH) as myfile:
        f=myfile.read()
        string = find_between(f, '< '+text.lower()+' | ', ' >')
        if(string != ''): return string
        return text

contractions_dict = json.loads(open(CONTRACTIONS_PATH).read())
contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)

def generate(text):
    text = re.sub(r'\([^)]*\)', '', text)
    text = ' '.join(text.split())
    text = text.replace(' ,', ',')

    p = re.compile(r'(?:(?<!\w)\'((?:.|\n)+?\'?)(?:(?<!s)\'(?!\w)|(?<=s)\'(?!([^\']|\w\'\w)+\'(?!\w))))')
    subst = "\"\g<1>\""
    text = re.sub(p, subst, text)

    predicates = []
    subjects = []
    objects = []
    extraFields = []
    types = []
   
    text = expand_contractions(text)
    print ("\n")
    print ("Preprocessed text: "+ text)
    textList = []
    textList.append(text)
    annotator = Annotator(
        SENNA_PATH, 
        PNTL_PATH, 
        "edu.stanford.nlp.trees."
    )

    try:
        posTags = annotator.get_annotaions(textList, dep_parse=False)['pos']
        chunks = annotator.get_annotaions(textList, dep_parse=False)['chunk']
    except IndexError:
        emptyList = []
        return emptyList
    
    predictor = Predictor.from_path(SRL_MODEL_PATH)
    srlResult = predictor.predict_json({"sentence": text})
    print(srlResult)
    srls = []
    try:
        for i in range(0,len(srlResult['verbs'])):
            myDict = {}
            description = srlResult['verbs'][i]['description']
            while (find_between(description,"[","]") != ""):
                parts = find_between(description,"[","]").split(": ")
                myDict[parts[0]] = parts[1]
                description = description.replace("["+find_between(description,"[","]")+"]", "")
                if (find_between(description,"[","]") == ""):
                    srls.append(myDict)
    except IndexError:
        emptyList = []
        return emptyList

    nlp = spacy.load('en_core_web_sm')
    doc = nlp(u''+text)

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
    foundQuestions = []
    idiomJson = json.loads(open(IDIOMS_PATH).read())
    for word in doc:
        if word.dep_ == 'dobj' or word.dep_ == 'ccomp' or word.dep_ == 'xcomp' or word.dep_ == 'dative' or word.dep_ == 'acomp' or word.dep_ == 'attr' or word.dep_ == "oprd":
            try:
                baseFormVerb = getBaseFormOfVerb(word.head.text)
                if word.text.find(idiomJson[baseFormVerb]) != -1: continue
            except KeyError:
                pass
        #what question / who question
        if word.dep_ == 'dobj' or word.dep_ == 'ccomp' or word.dep_ == 'xcomp':
            dobjVerb.append(word.head.text)
            dobjWord.append(word.text)
            dobjSubType.append('dobj')
        if word.dep_ == 'dative':
            dativeVerb.append(word.head.text)
            dativeWord.append(word.text)
            dativeSubType.append('dative')
        if word.dep_ == 'acomp':
            acompVerb.append(word.head.text)
            acompWord.append(word.text)
            acompSubType.append(word.dep_)
        if word.dep_ == 'attr' or word.dep_ == "oprd":
            attrVerb.append(word.head.text)
            attrWord.append(word.text)
            attrSubType.append('attr')
        #what question
        if word.dep_ == 'pcomp':
            pcompPreposition.append(word.head.text)
            pcompWord.append(word.text)
            pcompSubType.append(word.dep_)
    for ent in doc.ents:
        #when question
        if (ent.label_ == 'DATE' and ent.text.find('year old') == -1 and ent.text.find('years old') == -1 ):
            dateWord.append(ent.text)
            dateSubType.append(ent.label_)
        #how many question
        if ent.label_ == 'CARDINAL':
            numWord.append(ent.text)
            numSubType.append(ent.label_)
        #who question
        if ent.label_ == 'PERSON':
            personWord.append(ent.text)
            personSubType.append(ent.label_)
        #where question
        if ent.label_ == 'FACILITY' or ent.label_ == 'ORG' or ent.label_ == 'GPE' or ent.label_ == 'LOC':
            whereWord.append(ent.text)
            whereSubType.append('LOC')

    #Beginning of deconstruction stage      
    for i in range(0,len(dativeWord)):
        for k in range(0,len(dobjWord)):
            if dobjVerb[k] != dativeVerb[i]: continue
            for j in range(0,len(srls)):
                foundSubject = checkForAppropriateObjOrSub(srls,j,0)
                foundObject = checkForAppropriateObjOrSub(srls,j,1)
                foundIndirectObject = checkForAppropriateObjOrSub(srls,j,2)
                if (foundSubject == '') or (foundObject == '')  or (foundSubject == foundObject) or ('V' not in srls[j].keys()):
                    continue
                if (foundIndirectObject == '')  or (foundIndirectObject == foundObject)  or (foundIndirectObject == foundSubject): 
                    continue
                elif (srls[j]['V'] == dobjVerb[k]) and (foundObject.find(dobjWord[k]) != -1 ) and (foundIndirectObject.find(dativeWord[i]) != -1 ):
                    index =1 -1
                    totalPredicates = srls[j]['V']
                    for k in range(0,len(chunks)):
                        found = re.compile(srls[j]['V'], re.DOTALL |  re.IGNORECASE).findall(str(chunks[k]))
                        if found:
                            index = k
                    for k in range(0,index):
                        reversedIndex = index -1 -k
                        if reversedIndex == -1: break
                        resultType = re.search("', '(.*)'\)", str(chunks[reversedIndex]))
                        try:
                            if resultType.group(1) == 'B-VP' or resultType.group(1) == 'E-VP' or resultType.group(1) == 'I-VP' or resultType.group(1) == 'S-VP':
                                result = re.search("\('(.*)',", str(chunks[reversedIndex]))
                                if foundSubject.find(result.group(1)) != -1: break
                                totalPredicates = result.group(1) + ' ' + totalPredicates 
                            else: break
                        except AttributeError:
                            break
                    nextIndex = index + 1
                    if (srls[j]['V'] != 'am' and srls[j]['V'] != 'is' and srls[j]['V'] != 'are' and srls[j]['V'] != 'was' and srls[j]['V'] != 'were'  and srls[j]['V'] != 'be'):
                        if (nextIndex < len(chunks)):
                            resultType = re.search("', '(.*)'\)", str(chunks[nextIndex]))
                            try:
                                if resultType.group(1) == 'S-PRT':
                                    result = re.search("\('(.*)',", str(chunks[nextIndex]))
                                    if foundSubject.find(result.group(1)) != -1: break
                                    totalPredicates = result.group(1) + ' ' + totalPredicates
                            except AttributeError:
                                pass
                    if totalPredicates[:3] == 'to ':
                        totalPredicates= totalPredicates[3:]
                    predicates.append(totalPredicates)
                    objects.append(foundIndirectObject + " " + foundObject)
                    subjects.append(foundSubject)
                    extraFieldsString = ''
                    if 'ARGM-LOC' in srls[j].keys():
                        if (totalPredicates.find(srls[j]['ARGM-LOC']) == -1):
                            extraFieldsString = srls[j]['ARGM-LOC']
                    if 'ARGM-TMP' in srls[j].keys():
                        if (totalPredicates.find(srls[j]['ARGM-TMP']) == -1):
                            extraFieldsString = extraFieldsString + ' ' + srls[j]['ARGM-TMP'] 
                    extraFields.append(extraFieldsString)
                    types.append(dativeSubType[i])
      
    for i in range(0,len(dobjWord)):
        for j in range(0,len(srls)):
            foundSubject = checkForAppropriateObjOrSub(srls,j,0)
            foundObject = checkForAppropriateObjOrSub(srls,j,1)
            if (foundSubject == '') or (foundObject == '') or (foundSubject == foundObject) or ('V' not in srls[j].keys()): continue
            elif (srls[j]['V'] == dobjVerb[i]) and (foundObject.find(dobjWord[i]) != -1 ) :
                index =1 -1
                totalPredicates = srls[j]['V']
                for k in range(0,len(chunks)):
                    found = re.compile(srls[j]['V'], re.DOTALL |  re.IGNORECASE).findall(str(chunks[k]))
                    if found:
                        index = k
                for k in range(0,index):
                    reversedIndex = index -1 -k
                    if reversedIndex == -1: break
                    resultType = re.search("', '(.*)'\)", str(chunks[reversedIndex]))
                    try:
                        if resultType.group(1) == 'B-VP' or resultType.group(1) == 'E-VP' or resultType.group(1) == 'I-VP' or resultType.group(1) == 'S-VP':
                            result = re.search("\('(.*)',", str(chunks[reversedIndex]))
                            if foundSubject.find(result.group(1)) != -1: break
                            totalPredicates = result.group(1) + ' ' + totalPredicates 
                        else: break
                    except AttributeError:
                        break
                nextIndex = index + 1
                if (srls[j]['V'] != 'am' and srls[j]['V'] != 'is' and srls[j]['V'] != 'are' and srls[j]['V'] != 'was' and srls[j]['V'] != 'were'  and srls[j]['V'] != 'be'):
                    if (nextIndex < len(chunks)):
                        resultType = re.search("', '(.*)'\)", str(chunks[nextIndex]))
                        try:
                            if resultType.group(1) == 'S-PRT':
                                result = re.search("\('(.*)',", str(chunks[nextIndex]))
                                if foundSubject.find(result.group(1)) != -1: break
                                totalPredicates = result.group(1) + ' ' + totalPredicates
                        except AttributeError:
                            pass
                if totalPredicates[:3] == 'to ':
                    totalPredicates= totalPredicates[3:]
                predicates.append(totalPredicates)
                objects.append(foundObject)
                subjects.append(foundSubject)
                extraFieldsString = ''
                if 'ARGM-LOC' in srls[j].keys():
                    if (totalPredicates.find(srls[j]['ARGM-LOC']) == -1):
                        extraFieldsString = srls[j]['ARGM-LOC']
                if 'ARGM-TMP' in srls[j].keys():
                    if (totalPredicates.find(srls[j]['ARGM-TMP']) == -1):
                        extraFieldsString = extraFieldsString + ' ' + srls[j]['ARGM-TMP'] 
                extraFields.append(extraFieldsString)
                types.append(dobjSubType[i])

    for i in range(0,len(acompWord)):
        for j in range(0,len(srls)):
            foundSubject = checkForAppropriateObjOrSub(srls,j,0)
            foundObject = checkForAppropriateObjOrSub(srls,j,1)
            if (foundSubject == '') or (foundObject == '') or (foundSubject == foundObject) or ('V' not in srls[j].keys()): continue
            elif (srls[j]['V'] == acompVerb[i]) and (foundObject.find(acompWord[i]) != -1 ) :
                predicates.append('indicate')
                objects.append(foundObject)
                subjects.append(foundSubject)
                extraFields.append(srls[j].get('ARGM-LOC', '') + ' ' + srls[j].get('ARGM-TMP', ''))
                types.append(acompSubType[i])

    for i in range(0,len(attrWord)):
        for j in range(0,len(srls)):
            foundSubject = checkForAppropriateObjOrSub(srls,j,0)
            if (foundSubject == '') or ('V' not in srls[j].keys()): continue
            for key, value in srls[j].items():
                if (srls[j]['V'] == attrVerb[i] and (value.find(attrWord[i]) != -1 ) and key != "V" and value != foundSubject):
                    predicates.append('describe')
                    objects.append(foundSubject)
                    subjects.append('you')
                    extraFields.append(srls[j].get('ARGM-LOC', '') + ' ' + srls[j].get('ARGM-TMP', ''))
                    types.append(attrSubType[i])

    for i in range(0,len(dateWord)):
        for j in range(0,len(srls)):
            foundSubject = checkForAppropriateObjOrSub(srls,j,0)
            foundObject = checkForAppropriateObjOrSub(srls,j,1)
            if (foundSubject == '') or (foundSubject == foundObject) or ('V' not in srls[j].keys()): continue
            for key, value in srls[j].items():
                if (value.find(dateWord[i]) != -1 ) and key != "V" and key!= "ARGM-TMP" and value != foundSubject and value != foundObject:
                    index =1 -1
                    totalPredicates = srls[j]['V']
                    for k in range(0,len(chunks)):
                        found = re.compile(srls[j]['V'], re.DOTALL |  re.IGNORECASE).findall(str(chunks[k]))
                        if found:
                            index = k
                    for k in range(0,index):
                        reversedIndex = index -1 -k
                        if reversedIndex == -1: break
                        resultType = re.search("', '(.*)'\)", str(chunks[reversedIndex]))
                        try:
                            if resultType.group(1) == 'B-VP' or resultType.group(1) == 'E-VP' or resultType.group(1) == 'I-VP' or resultType.group(1) == 'S-VP':
                                result = re.search("\('(.*)',", str(chunks[reversedIndex]))
                                if foundSubject.find(result.group(1)) != -1: break
                                totalPredicates = result.group(1) + ' ' + totalPredicates 
                            else: break
                        except AttributeError:
                            break
                    if totalPredicates[:3] == 'to ':
                        totalPredicates= totalPredicates[3:]
                    predicates.append(totalPredicates)
                    objects.append(foundObject)
                    subjects.append(foundSubject)
                    extraFieldsString = ''
                    if 'ARGM-LOC' in srls[j].keys():
                        if totalPredicates.find(srls[j]['ARGM-LOC']) == -1:
                            extraFieldsString = srls[j]['ARGM-LOC']
                    extraFieldsString = extraFieldsString.replace(dateWord[i], "")
                    extraFields.append(extraFieldsString)
                    types.append(dateSubType[i])

    for i in range(0,len(whereWord)):
        for j in range(0,len(srls)):
            foundSubject = checkForAppropriateObjOrSub(srls,j,0)
            foundObject = checkForAppropriateObjOrSub(srls,j,1)
            if (foundSubject == '') or (foundSubject == foundObject) or ('V' not in srls[j].keys()): continue
            for key, value in srls[j].items():
                if (value.find(whereWord[i]) != -1 ) and key != "V" and key!= "ARGM-LOC" and value != foundSubject:
                    index =1 -1
                    totalPredicates = srls[j]['V']
                    for k in range(0,len(chunks)):
                        found = re.compile(srls[j]['V'], re.DOTALL |  re.IGNORECASE).findall(str(chunks[k]))
                        if found:
                            index = k
                    for k in range(0,index):
                        reversedIndex = index -1 -k
                        if reversedIndex == -1: break
                        resultType = re.search("', '(.*)'\)", str(chunks[reversedIndex]))
                        try:
                            if resultType.group(1) == 'B-VP' or resultType.group(1) == 'E-VP' or resultType.group(1) == 'I-VP' or resultType.group(1) == 'S-VP':
                                result = re.search("\('(.*)',", str(chunks[reversedIndex]))
                                if foundSubject.find(result.group(1)) != -1: break
                                totalPredicates = result.group(1) + ' ' + totalPredicates 
                            else: break
                        except AttributeError:
                            break
                    if totalPredicates[:3] == 'to ':
                        totalPredicates= totalPredicates[3:]

                    realObj = ''
                    if(foundObject != '' and value == foundObject):
                        valueArray = value.split(' ')
                        for l in range(0,len((doc))):
                            if(l + 1 >= len(doc)): break
                            if(value.find(doc[l].text) == -1) or (value.find(doc[l+1].text) == -1): continue

                            if whereWord[i].find(doc[l+1].text) != -1 and doc[l].pos_ == 'ADP':
                                break
                            
                            if(realObj == ''): realObj = doc[l].text
                            else: realObj += ' ' + doc[l].text 
                    else:
                        realObj = foundObject

                    predicates.append(totalPredicates)
                    if realObj[-4:] == ' the':
                        realObj = realObj[:-4]
                    objects.append(realObj)
                    subjects.append(foundSubject)
                    extraFieldsString = ''
                    if 'ARGM-TMP' in srls[j].keys():
                        if totalPredicates.find(srls[j]['ARGM-TMP']) == -1:
                            extraFieldsString = srls[j]['ARGM-TMP']
                    extraFields.append(extraFieldsString)
                    types.append(whereSubType[i])

    for i in range(0,len(pcompWord)):
        for j in range(0,len(srls)):
            if 'V' not in srls[j].keys(): 
                continue
            foundSubject = checkForAppropriateObjOrSub(srls,j,0)
            index =1 -1
            totalPredicates = pcompPreposition[i]
            
            for k in range(0,len(chunks)):
                found = re.compile(pcompPreposition[i], re.DOTALL |  re.IGNORECASE).findall(str(chunks[k]))
                if found:
                    index = k
            isMainVerbFound = False
            for k in range(0,index):
                reversedIndex = index -1 -k
                if reversedIndex == -1: break
                if (isMainVerbFound == False):
                    totalPredicates= str(chunks[reversedIndex][0]) + ' '+totalPredicates 
                    if (chunks[reversedIndex][0] == srls[j]['V']): 
                        isMainVerbFound = True
                elif (isMainVerbFound == True):
                    resultType = re.search("', '(.*)'\)", str(chunks[reversedIndex]))
                    try:
                        if resultType.group(1) == 'B-VP' or resultType.group(1) == 'E-VP' or resultType.group(1) == 'I-VP' or resultType.group(1) == 'S-VP':
                            result = re.search("\('(.*)',", str(chunks[reversedIndex]))
                            if foundSubject.find(result.group(1)) != -1: break
                            totalPredicates = result.group(1) + ' ' + totalPredicates 
                        else: break
                    except AttributeError:
                        break
            if (totalPredicates.find(srls[j]['V']) != -1 ):
                if (checkForAppropriateObjOrSub(srls,j,0) != ''):
                    foundObject = checkForAppropriateObjOrSub(srls,j,1)
                else:
                    continue
                for key, value in srls[j].items():
                    if(key != 'V' and value.find(pcompWord[i]) != -1 and foundSubject != ''):
                        if (foundSubject == value and foundObject != '' and foundObject != foundSubject ):
                            subjects.append(foundObject)
                        else:
                            subjects.append(foundSubject)

                        objects.append('')
                        predicates.append(totalPredicates)
                        extraFields.append('')
                        types.append(pcompSubType[i])
            else: 
                continue

    for i in range(0,len(numWord)):
        for j in range(0,len(srls)):
            foundSubject = checkForAppropriateObjOrSub(srls,j,0)
            foundObject = checkForAppropriateObjOrSub(srls,j,1)
            if (foundSubject == '') or (foundSubject == foundObject) or ('V' not in srls[j].keys()): continue
            for key, value in srls[j].items():
                if (value.find(numWord[i]) != -1 ) and key != "V":
                    index =1 -1
                    totalPredicates = srls[j]['V']
                    for k in range(0,len(chunks)):
                        found = re.compile(srls[j]['V'], re.DOTALL |  re.IGNORECASE).findall(str(chunks[k]))
                        if found:
                            index = k
                    for k in range(0,index):
                        reversedIndex = index -1 -k
                        if reversedIndex == -1: break
                        resultType = re.search("', '(.*)'\)", str(chunks[reversedIndex]))
                        try:
                            if resultType.group(1) == 'B-VP' or resultType.group(1) == 'E-VP' or resultType.group(1) == 'I-VP' or resultType.group(1) == 'S-VP':
                                result = re.search("\('(.*)',", str(chunks[reversedIndex]))
                                if foundSubject.find(result.group(1)) != -1: break
                                totalPredicates = result.group(1) + ' ' + totalPredicates 
                            else: break
                        except AttributeError:
                            break
                    nextIndex = index + 1
                    if (srls[j]['V'] != 'am' and srls[j]['V'] != 'is' and srls[j]['V'] != 'are' and srls[j]['V'] != 'was' and srls[j]['V'] != 'were'  and srls[j]['V'] != 'be'):
                        if (nextIndex < len(chunks)):
                            resultType = re.search("', '(.*)'\)", str(chunks[nextIndex]))
                            try:
                                if resultType.group(1) == 'S-PRT':
                                    result = re.search("\('(.*)',", str(chunks[nextIndex]))
                                    if foundSubject.find(result.group(1)) != -1: break
                                    totalPredicates = result.group(1) + ' ' + totalPredicates
                            except AttributeError:
                                pass
                    if totalPredicates[:3] == 'to ':
                        totalPredicates= totalPredicates[3:]
                    midFoundIndex = -1
                    valueArray = value.split(" ")
                    for l in range(0,len(valueArray)):
                        if valueArray[l].find(numWord[i]) != -1:
                            midFoundIndex = l

                    valueArrayFirstPart = valueArray[(midFoundIndex + 1):]
                    valueArrayLastPart = valueArray[:midFoundIndex]

                    valueFirstPart = ""
                    for l in range(0,len(valueArrayFirstPart)):
                        if valueFirstPart == "": valueFirstPart = valueArrayFirstPart[l]
                        else: valueFirstPart = valueFirstPart + " " + valueArrayFirstPart[l]

                    valueLastPart = ""
                    for l in range(0,len(valueArrayLastPart)):
                        if valueLastPart == "": valueLastPart = valueArrayLastPart[l]
                        elif l == (len(valueArrayLastPart) -1) and valueArrayLastPart[l] == "the":  break
                        else: valueLastPart = valueLastPart + " " + valueArrayLastPart[l]
                    
                    #if (nouns == ""): continue
                    predicates.append(totalPredicates)
                    subjects.append(valueFirstPart)
                    #extraFields.append(valueLastPart)
                    #objects.append(foundObject + " " + keyCheck('ARGM-LOC',srls[j],'') + " " + keyCheck('ARGM-TMP',srls[j],''))
                    types.append(numSubType[i])

                    extraFieldsString = ''
                    if 'ARGM-LOC' in srls[j].keys():
                        if (totalPredicates.find(srls[j]['ARGM-LOC']) == -1):
                            extraFieldsString = srls[j]['ARGM-LOC']
                    if 'ARGM-TMP' in srls[j].keys():
                        if (totalPredicates.find(srls[j]['ARGM-TMP']) == -1):
                            extraFieldsString = extraFieldsString + ' ' + srls[j]['ARGM-TMP'] 

                    if (value == foundObject and foundSubject != ''):
                        objects.append(extraFieldsString)
                        extraFields.append(foundSubject)
                    else:
                        objects.append(valueLastPart + " " + foundObject + " " + extraFieldsString)
                        extraFields.append('')
  

    for i in range(0,len(personWord)):
        for j in range(0,len(srls)):
            foundSubject = checkForAppropriateObjOrSub(srls,j,0)
            foundObject = checkForAppropriateObjOrSub(srls,j,1)
            if (foundSubject == '') or (foundSubject == foundObject) or ('V' not in srls[j].keys()): continue
            for key, value in srls[j].items():
                if (value.find(personWord[i]) != -1 ) and key != "V" and value == foundSubject and value != foundObject:
                    index =1 -1
                    totalPredicates = srls[j]['V']
                    for k in range(0,len(chunks)):
                        found = re.compile(srls[j]['V'], re.DOTALL |  re.IGNORECASE).findall(str(chunks[k]))
                        if found:
                            index = k
                    for k in range(0,index):
                        reversedIndex = index -1 -k
                        if reversedIndex == -1: break
                        resultType = re.search("', '(.*)'\)", str(chunks[reversedIndex]))
                        try:
                            if resultType.group(1) == 'B-VP' or resultType.group(1) == 'E-VP' or resultType.group(1) == 'I-VP' or resultType.group(1) == 'S-VP':
                                result = re.search("\('(.*)',", str(chunks[reversedIndex]))
                                if foundSubject.find(result.group(1)) != -1: break
                                totalPredicates = result.group(1) + ' ' + totalPredicates 
                            else: break
                        except AttributeError:
                            break
                    nextIndex = index + 1
                    if (srls[j]['V'] != 'am' and srls[j]['V'] != 'is' and srls[j]['V'] != 'are' and srls[j]['V'] != 'was' and srls[j]['V'] != 'were'  and srls[j]['V'] != 'be'):
                        if (nextIndex < len(chunks)):
                            resultType = re.search("', '(.*)'\)", str(chunks[nextIndex]))
                            try:
                                if resultType.group(1) == 'S-PRT':
                                    result = re.search("\('(.*)',", str(chunks[nextIndex]))
                                    if foundSubject.find(result.group(1)) != -1: break
                                    totalPredicates = result.group(1) + ' ' + totalPredicates
                            except AttributeError:
                                pass
                    if totalPredicates[:3] == 'to ':
                        totalPredicates= totalPredicates[3:]

                    relativeClauseDet = False
                    otherNounsDet = False
                    if (find_between(value,personWord[i],",") == " "): relativeClauseDet = True
                    if (find_between(value,personWord[i],"who") == " "): relativeClauseDet = True
                    if (find_between(value,personWord[i],"that") == " "): relativeClauseDet = True
                    if (find_between(value,personWord[i],"whose") == " "): relativeClauseDet = True
                    if (relativeClauseDet == False):
                        modifSrl = value.replace("' "+personWord[i]+" '", '')
                        modifSrl = value.replace('" '+personWord[i]+' "', '')
                        modifSrl = value.replace(personWord[i], '')
                        modifSrl = modifSrl.split(' ')
                        for m in range(0,len(modifSrl)):
                            for k in range(0,len(chunks)):
                                resultType = re.search("', '(.*)'\)", str(chunks[k]))
                                resultType1 = re.search("'(.*)',", str(chunks[k]))
                                try:
                                    if (modifSrl[m] == resultType1.group(1) and len(resultType1.group(1)) > 1):
                                        if resultType.group(1) == 'B-NP' or resultType.group(1) == 'E-NP' or resultType.group(1) == 'I-NP' or resultType.group(1) == 'S-NP':
                                            otherNounsDet = True
                                            break
                                except AttributeError:
                                    pass
                    extraFieldsString = ''
                    if 'ARGM-LOC' in srls[j].keys():
                        if (totalPredicates.find(srls[j]['ARGM-LOC']) == -1):
                            extraFieldsString = srls[j]['ARGM-LOC']
                    if 'ARGM-TMP' in srls[j].keys():
                        if (totalPredicates.find(srls[j]['ARGM-TMP']) == -1):
                            extraFieldsString = extraFieldsString + ' ' + srls[j]['ARGM-TMP'] 
                    if (otherNounsDet == False):
                        extraFields.append(extraFieldsString)
                        types.append(personSubType[i])
                        predicates.append(totalPredicates)
                        objects.append(foundObject)
                        subjects.append('')
                        

    for j in range(0,len(srls)):
        foundSubject = checkForAppropriateObjOrSub(srls,j,0)
        foundObject = checkForAppropriateObjOrSub(srls,j,1)
        if (foundSubject == '') or ('V' not in srls[j].keys()) or (foundSubject == foundObject): continue
        index =1 -1
        totalPredicates = srls[j]['V']
        for k in range(0,len(chunks)):
            found = re.compile(srls[j]['V'], re.DOTALL |  re.IGNORECASE).findall(str(chunks[k]))
            if found:
                index = k
        for k in range(0,index):
            reversedIndex = index -1 -k
            if reversedIndex == -1: break
            resultType = re.search("', '(.*)'\)", str(chunks[reversedIndex]))
            try:
                if resultType.group(1) == 'B-VP' or resultType.group(1) == 'E-VP' or resultType.group(1) == 'I-VP' or resultType.group(1) == 'S-VP':
                    result = re.search("\('(.*)',", str(chunks[reversedIndex]))
                    if foundSubject.find(result.group(1)) != -1: break
                    totalPredicates = result.group(1) + ' ' + totalPredicates 
                else: break
            except AttributeError:
                break
        nextIndex = index + 1
        if (srls[j]['V'] != 'am' and srls[j]['V'] != 'is' and srls[j]['V'] != 'are' and srls[j]['V'] != 'was' and srls[j]['V'] != 'were'  and srls[j]['V'] != 'be'):
            if (nextIndex < len(chunks)):
                resultType = re.search("', '(.*)'\)", str(chunks[nextIndex]))
                try:
                    if resultType.group(1) == 'S-PRT':
                        result = re.search("\('(.*)',", str(chunks[nextIndex]))
                        if foundSubject.find(result.group(1)) != -1: break
                        totalPredicates = result.group(1) + ' ' + totalPredicates
                except AttributeError:
                    pass
        if totalPredicates[:3] == 'to ':
            totalPredicates= totalPredicates[3:]

        if 'ARGM-CAU' in srls[j].keys():
            predicates.append(totalPredicates)
            objects.append(foundObject)
            subjects.append(foundSubject)
            extraFieldsString = ''
            if 'ARGM-LOC' in srls[j].keys():
                if (totalPredicates.find(srls[j]['ARGM-LOC']) == -1):
                    extraFieldsString = srls[j]['ARGM-LOC']
            if 'ARGM-TMP' in srls[j].keys():
                if (totalPredicates.find(srls[j]['ARGM-TMP']) == -1):
                    extraFieldsString = extraFieldsString + ' ' + srls[j]['ARGM-TMP'] 
            extraFields.append(extraFieldsString)
            types.append('why')
        if 'ARGM-PNC' in srls[j].keys():
            predicates.append(totalPredicates)
            objects.append(foundObject)
            subjects.append(foundSubject)
            extraFieldsString = ''
            if 'ARGM-LOC' in srls[j].keys():
                if (totalPredicates.find(srls[j]['ARGM-LOC']) == -1):
                    extraFieldsString = srls[j]['ARGM-LOC']
            if 'ARGM-TMP' in srls[j].keys():
                if (totalPredicates.find(srls[j]['ARGM-TMP']) == -1):
                    extraFieldsString = extraFieldsString + ' ' + srls[j]['ARGM-TMP'] 
            extraFields.append(extraFieldsString)
            types.append('purpose')
        if 'ARGM-MNR' in srls[j]:
            predicates.append(totalPredicates)
            objects.append(foundObject)
            subjects.append(foundSubject)
            extraFieldsString = ''
            if 'ARGM-LOC' in srls[j].keys():
                if (totalPredicates.find(srls[j]['ARGM-LOC']) == -1):
                    extraFieldsString = srls[j]['ARGM-LOC']
            if 'ARGM-TMP' in srls[j].keys():
                if (totalPredicates.find(srls[j]['ARGM-TMP']) == -1):
                    extraFieldsString = extraFieldsString + ' ' + srls[j]['ARGM-TMP'] 
            extraFields.append(extraFieldsString)
            types.append('how')
        if 'ARGM-TMP' in srls[j]:
            predicates.append(totalPredicates)
            objects.append(foundObject)
            subjects.append(foundSubject)
            extraFieldsString = ''
            if 'ARGM-LOC' in srls[j].keys():
                if (totalPredicates.find(srls[j]['ARGM-LOC']) == -1):
                    extraFieldsString = srls[j]['ARGM-LOC']
            extraFields.append(extraFieldsString)
            types.append('DATE')
        if 'ARGM-LOC' in srls[j]:
            predicates.append(totalPredicates)
            objects.append(foundObject)
            subjects.append(foundSubject)
            extraFieldsString = ''
            if 'ARGM-TMP' in srls[j].keys():
                if (totalPredicates.find(srls[j]['ARGM-TMP']) == -1):
                    extraFieldsString = srls[j]['ARGM-TMP']
            extraFields.append(extraFieldsString)
            types.append('LOC')

    for j in range(0,len(srls)):
        foundSubject = checkForAppropriateObjOrSub(srls,j,0)
        foundObject = checkForAppropriateObjOrSub(srls,j,1)
        if (foundSubject == '') or ('V' not in srls[j].keys()) or (foundSubject == foundObject) or (foundObject == ""): continue
        index =1 -1
        totalPredicates = srls[j]['V']
        for k in range(0,len(chunks)):
            found = re.compile(srls[j]['V'], re.DOTALL |  re.IGNORECASE).findall(str(chunks[k]))
            if found:
                index = k
        for k in range(0,index):
            reversedIndex = index -1 -k
            if reversedIndex == -1: break
            resultType = re.search("', '(.*)'\)", str(chunks[reversedIndex]))
            try:
                if resultType.group(1) == 'B-VP' or resultType.group(1) == 'E-VP' or resultType.group(1) == 'I-VP' or resultType.group(1) == 'S-VP':
                    result = re.search("\('(.*)',", str(chunks[reversedIndex]))
                    if foundSubject.find(result.group(1)) != -1: break
                    totalPredicates = result.group(1) + ' ' + totalPredicates 
                else: break
            except AttributeError:
                break
        nextIndex = index + 1
        if (srls[j]['V'] != 'am' and srls[j]['V'] != 'is' and srls[j]['V'] != 'are' and srls[j]['V'] != 'was' and srls[j]['V'] != 'were'  and srls[j]['V'] != 'be'):
            if (nextIndex < len(chunks)):
                resultType = re.search("', '(.*)'\)", str(chunks[nextIndex]))
                try:
                    if resultType.group(1) == 'S-PRT':
                        result = re.search("\('(.*)',", str(chunks[nextIndex]))
                        if foundSubject.find(result.group(1)) != -1: break
                        totalPredicates = result.group(1) + ' ' + totalPredicates
                except AttributeError:
                    pass
        if totalPredicates[:3] == 'to ':
            totalPredicates= totalPredicates[3:]

        predicates.append(totalPredicates)
        objects.append(foundObject)
        subjects.append(foundSubject)
        extraFieldsString = ''
        if 'ARGM-LOC' in srls[j].keys():
            if (totalPredicates.find(srls[j]['ARGM-LOC']) == -1):
                extraFieldsString = srls[j]['ARGM-LOC']
        if 'ARGM-TMP' in srls[j].keys():
            if (totalPredicates.find(srls[j]['ARGM-TMP']) == -1):
                extraFieldsString = extraFieldsString + ' ' + srls[j]['ARGM-TMP'] 
        extraFields.append(extraFieldsString)
        types.append('direct')

    print ('---- Found Deconstruction results : ----')
    for i in range(0,len(subjects)):
        print(subjects[i])
        print(predicates[i])
        print(objects[i])
        print(extraFields[i])
        print(types[i])
        print ('----------------------------------------')

    #Beginning of construction stage    
    for i in range(0,len(subjects)):
        negativePart = ''
        negativeIndex = -1
        predArr = predicates[i].split(' ')
        numOfVerbs = 0
        firstFoundVerbIndex = -1
        isToDetected = False
        isAndDetected = False
        for j in range(0,len(predArr)):
            if predArr[j] == 'and': isAndDetected = True
            for k in range (0,len(posTags)):
                if posTags[k][0]== predArr[j] and posTags[k][0] == 'to':
                    isToDetected = True
                if posTags[k][0]== predArr[j] and (posTags[k][1] == 'VB' or posTags[k][1] == 'VBD' or posTags[k][1] == 'VBG' or posTags[k][1] == 'VBN' or posTags[k][1] == 'VBP' or posTags[k][1] == 'VBZ' or posTags[k][1] == 'MD'):
                    if numOfVerbs == 0:
                        firstFoundVerbIndex = j
                    numOfVerbs = numOfVerbs + 1
                    break
                if posTags[k][0]== predArr[j] and posTags[k][1] == 'RB' and posTags[k][0].lower() == 'not':
                    if numOfVerbs == 0:
                        firstFoundVerbIndex = j
                    numOfVerbs = numOfVerbs + 1
                    negativeIndex = j
                    break
                #elif firstFoundVerbIndex != -1: break
        if isAndDetected == False:
            if negativeIndex > -1:
                negativePart = predArr.pop(negativeIndex)
            if numOfVerbs == 1 or isToDetected == True:
                if predArr[0] != 'am' and predArr[0] != 'is' and predArr[0] != 'are' and predArr[0] != 'was' and predArr[0] != 'were':
                    for k in range (0,len(posTags)):
                        predArrNew = []
                        if posTags[k][0] == predArr[firstFoundVerbIndex]:
                            predArrNew = []
                            if posTags[k][1] == 'MD':
                                break
                            if posTags[k][1] == 'VBG': 
                                types[i] = '' 
                                break
                            if posTags[k][1] == 'VBZ':
                                predArrNew.append('does')
                                if posTags[k][0] == 'has':
                                    predArrNew.append(posTags[k][0])
                                else:
                                    predArrNew.append(getBaseFormOfVerb(posTags[k][0]))
                            elif posTags[k][1] == 'VBP':
                                predArrNew = []
                                predArrNew.append('do')
                                predArrNew.append(posTags[k][0])
                            elif posTags[k][1] == 'VBD' or posTags[k][1] == 'VBN':
                                predArrNew = []
                                predArrNew.append('did')
                                predArrNew.append(getBaseFormOfVerb(posTags[k][0]))
                            else:
                                predArrNew = []
                                subjectParts = subjects[i].split(' ')
                                isFound = False
                                for l in range (0,len(posTags)):
                                    if isFound == True: break
                                    for m in range (0,len(subjectParts)):
                                        if posTags[l][0] == subjectParts[m]:
                                            if posTags[l][1] == 'NN':
                                                predArrNew.append('does')
                                                predArrNew.append(posTags[k][0])
                                                isFound = True
                                                break
                                            elif posTags[l][1] == 'NNS':
                                                predArrNew.append('do')
                                                predArrNew.append(posTags[k][0])
                                                isFound = True
                                                break
                                        if l == (len(posTags)-1) and m == (len(subjectParts)-1):
                                            predArrNew.append('do')
                                            predArrNew.append(posTags[k][0])
                                            isFound = True
                                            break
                            predArr.pop(firstFoundVerbIndex)
                            predArrTemp = predArr
                            predArr = predArrNew + predArrTemp
                            break
            if numOfVerbs == 0 and len(predArr) == 1 and types[i] != 'attr':
                mainVerb = predArr[0]
                qVerb = ''
                if getBaseFormOfVerb(predArr[0]) == predArr[0]:
                    qVerb = 'do'
                else:
                    qVerb = 'does'

                predArr = []
                predArr.append(qVerb)
                predArr.append(mainVerb)

        if isAndDetected == True: 
            predArr.insert(0, '')
        isQuestionMarkExist = False
        verbRemainingPart = ''
        question = ''
        for k in range (1,len(predArr)):
            verbRemainingPart = verbRemainingPart + ' ' + predArr[k]

        if types[i] == 'dative':
            question = 'What ' + predArr[0] + ' ' + negativePart + ' ' + verbRemainingPart + ' ' + objects[i] + ' ' + extraFields[i]
        if types[i] == 'dobj' or types[i] == 'pcomp':
            whQuestion = 'What '
            for ent in doc.ents:
                if(ent.text == objects[i] and ent.label_ == 'PERSON'):
                    whQuestion = 'Who '
            question = whQuestion + predArr[0] + ' ' + negativePart + ' ' + subjects[i] + verbRemainingPart  + ' ' + extraFields[i]
        elif types[i] == 'DATE':
            question = 'When '+predArr[0] + ' ' + negativePart + ' ' + subjects[i] + verbRemainingPart + ' ' + objects[i]  + ' ' + extraFields[i]
        elif types[i] == 'LOC':
            question = 'Where '+predArr[0] + ' ' + negativePart + ' ' + subjects[i] + verbRemainingPart + ' ' + objects[i]  + ' ' + extraFields[i]
        elif types[i] == 'CARDINAL':
            question = 'How many ' + subjects[i] + ' ' + predArr[0] + ' ' + negativePart + ' ' + extraFields[i]  + ' ' + verbRemainingPart + ' ' + objects[i]
        elif types[i] == 'attr':
            question = 'How would  '+ subjects[i] + ' ' + predArr[0] + ' ' + negativePart + ' ' + verbRemainingPart + ' ' + objects[i]
        elif types[i] == 'PERSON':
            if objects[i].endswith('.'):
                objects[i] = objects[i][:-1]
            question = 'Who  ' + predArr[0] + ' ' + negativePart + ' ' + verbRemainingPart + ' ' + objects[i] + ' ' + extraFields[i]
        elif types[i] == 'WHAT':
            if objects[i].endswith('.'):
                objects[i] = objects[i][:-1]
            question = 'What  ' + predArr[0] + ' ' + negativePart + ' ' + verbRemainingPart + ' ' + objects[i] + ' ' + extraFields[i]
        elif types[i] == 'acomp':
            isQuestionMarkExist = True  
            question = 'Indicate characteristics of ' + getObjectPronun(subjects[i])
        elif types[i] == 'direct':
            predArr[0]= predArr[0][:1].upper() + predArr[0][1:]
            if objects[i].endswith('.'):
                objects[i] = objects[i][:-1]
            question = predArr[0] + ' '  + negativePart + ' ' + subjects[i] + ' ' + verbRemainingPart + ' ' + objects[i]  + ' ' + extraFields[i]
        elif types[i] == 'why':
            question = 'Why '+predArr[0] + ' ' + negativePart + ' ' + subjects[i] + verbRemainingPart + ' ' + objects[i]  + ' ' + extraFields[i]
        elif types[i] == 'purpose':
            question = 'For what purpose '+predArr[0] + ' ' + negativePart + ' ' + subjects[i] + verbRemainingPart + ' ' + objects[i]  + ' ' + extraFields[i]
        elif types[i] == 'how':
            question = 'How '+predArr[0] + ' ' + negativePart + ' ' + subjects[i] + verbRemainingPart + ' ' + objects[i]  + ' ' + extraFields[i]


        
        isUpperWord = False
        postProcessTextArr = text.split(' ')
        lowerCasedWord = postProcessTextArr[0][0].lower() + postProcessTextArr[0][1:]

        for ent in doc.ents:
            if ent.text.find(postProcessTextArr[0]) != -1 and (ent.label_ == 'PERSON' or ent.label_ == 'FACILITY' or ent.label_ == 'GPE' or ent.label_ == 'ORG'):
                lowerCasedWord = postProcessTextArr[0]

        if lowerCasedWord == 'i': lowerCasedWord = 'I'
        #Postprocess stage for lower casing common nouns, omitting extra spaces and dots
        formattedQuestion = question.replace(postProcessTextArr[0],lowerCasedWord)
        formattedQuestion = ' '.join(formattedQuestion.split())
        formattedQuestion = formattedQuestion.replace(' ,', ',')
        formattedQuestion = formattedQuestion.replace(" 's " , "'s ")
        formattedQuestion = formattedQuestion.replace("s ' " , "s' ")
        quotatedString = re.findall('"([^"]*)"', formattedQuestion)
        quotatedOrgString = re.findall('"([^"]*)"', formattedQuestion)
        for l in range(0,len(quotatedString)):
            if quotatedString[l][0] == " ": quotatedString[l] = quotatedString[l][1:]
            if quotatedString[l][-1] == " ": quotatedString[l] = quotatedString[l][:-1]
            formattedQuestion = formattedQuestion.replace(quotatedOrgString[l], quotatedString[l])

        while (formattedQuestion.endswith(' ')):
            formattedQuestion = formattedQuestion[:-1]
        if formattedQuestion.endswith('.') or formattedQuestion.endswith(','):
            formattedQuestion = formattedQuestion[:-1]
        if formattedQuestion != '':
            if isQuestionMarkExist == False:
                formattedQuestion = formattedQuestion + '?'
            else: 
                formattedQuestion = formattedQuestion + '.'

            foundQuestions.append(formattedQuestion)

    foundQuestions.sort(key = lambda s: len(s))
    indexer = len(foundQuestions) - 1

    while (indexer > -1):
        if (indexer -1 < 0): break
        if foundQuestions[indexer] == foundQuestions[indexer-1]:
            foundQuestions.remove(foundQuestions[indexer])
        indexer = indexer - 1

    return foundQuestions
    
if __name__ == "__main__":
    text = "In 1980, the son of Vincent J. McMahon, Vincent Kennedy McMahon, founded Titan Sports, Inc. and in 1982 purchased Capitol Wrestling Corporation from his father."
    questions = generate(text)
    for question in questions:
        print(question)