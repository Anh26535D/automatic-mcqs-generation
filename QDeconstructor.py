import re
from typing import List

import utils

class QDeconstructionResult:
    def __init__(self, predicate=None, subject=None, object=None, extraField=None, type=None, key_answer=None):
        self.predicate = predicate
        self.subject = subject
        self.object = object
        self.extraField = extraField
        self.type = type
        self.key_answer = key_answer

class QDeconstructor:

    def __init__(self, doc, srls, chunks):
        self.doc = doc
        self.srls = srls
        self.chunks = chunks

    def checkForAppropriateObjOrSub(self, srl, occur_no):
        """
        Function to find the appropriate object or subject in the SRL dictionary.
        In common SRLs, the following roles are used: 
        ARG0 is an agent,
        ARG1 is a patient,
        ARG2 is a instrument, beneficiary, or attribute,
        ARG3 is a starting point, benefactive, or attribute,
        ARG4 is an ending point, benefactive, or attribute,
        others ARG are rarely occurred, and usually used for attributes.
        and ARGM- is a modifier.

        Args:
        -----
        srl: dict
            SRL dictionary
        occur_no: int
            The order of the object or subject in the SRL dictionary
        
        Returns:
        --------
        str
        """
        if (occur_no > 2) or (occur_no < 0):
            return ''
        if (occur_no < 2):
            max_idx = 5
        else:
            max_idx = 6
        for i in range(max_idx):
            if ('ARG' + str(i) in srl.keys()):
                occur_no = occur_no - 1
                if (occur_no == -1):
                    return srl['ARG' + str(i)]
        return ''


    def deconstruct(self, 
                    dativeWord, dativeVerb, dativeSubType, 
                    dobjWord, dobjVerb, dobjSubType, 
                    acompWord, acompVerb, acompSubType, 
                    attrWord, attrVerb, attrSubType, 
                    dateWord, dateSubType, 
                    whereWord, whereSubType, 
                    pcompWord, pcompPreposition, pcompSubType, 
                    numWord, numSubType, 
                    personWord, personSubType):
        deconstruction_result : List[QDeconstructionResult] = []
        
        # Dative (DATIVE)
        for i in range(len(dativeWord)):
            for k in range(len(dobjWord)):
                if (dobjVerb[k] != dativeVerb[i]): # direct object and dative is not from the same verb
                    continue
                for srl in self.srls:
                    if ('V' not in srl.keys()): 
                        continue
                    foundSubject = self.checkForAppropriateObjOrSub(srl, 0)
                    foundObject = self.checkForAppropriateObjOrSub(srl, 1)
                    foundIndirectObject = self.checkForAppropriateObjOrSub(srl, 2)
                    if (foundSubject == '') or (foundObject == '') or (foundIndirectObject == ''):
                        continue
                    if  (foundSubject == foundObject) or (foundIndirectObject == foundObject) or (foundIndirectObject == foundSubject): 
                        continue
                    if (srl['V'] == dobjVerb[k]) and (dobjWord[k] in foundObject) and (dativeWord[i] in foundIndirectObject):
                        last_index = 0
                        totalPredicates = srl['V']
                        for k in range(len(self.chunks)):
                            if (re.search(srl['V'], str(self.chunks[k]), re.DOTALL | re.IGNORECASE)):
                                last_index = k
                        for idx in range(last_index-1, -1, -1):
                            chunk_words = self.chunks[idx][0]
                            chunk_tag = self.chunks[idx][1]
                            if (chunk_tag in ['B-VP', 'E-VP', 'I-VP', 'S-VP']):
                                if (chunk_words in foundSubject): 
                                    break
                                totalPredicates = chunk_words + ' ' + totalPredicates 
                            else: 
                                break
                        if (last_index + 1 < len(self.chunks)):
                            if (srl['V'] not in ['am', 'is', 'are', 'was', 'were', 'be']):
                                chunk_words = self.chunks[last_index + 1][0]
                                chunk_tag = self.chunks[last_index + 1][1]
                                if (chunk_tag == 'S-PRT') and (chunk_words not in foundSubject):
                                    totalPredicates = chunk_words + ' ' + totalPredicates
                        if (totalPredicates.startswith('to ')):
                            totalPredicates = totalPredicates[3:]
                        current_result = QDeconstructionResult()
                        current_result.predicate = totalPredicates
                        current_result.object = foundIndirectObject + " " + foundObject
                        current_result.subject = foundSubject
                        extraFieldsString = ''
                        if ('ARGM-LOC' in srl.keys()) and (srl['ARGM-LOC'] not in totalPredicates):
                            extraFieldsString = srl['ARGM-LOC']
                        if ('ARGM-TMP' in srl.keys()) and (srl['ARGM-TMP'] not in totalPredicates):
                            extraFieldsString = extraFieldsString + ' ' + srl['ARGM-TMP'] 
                        current_result.extraField = extraFieldsString
                        current_result.type = dativeSubType[i]
                        deconstruction_result.append(current_result)

        # Direct Object (DOBJ)
        for i in range(len(dobjWord)):
            for srl in self.srls:
                if ('V' not in srl.keys()):
                    continue
                foundSubject = self.checkForAppropriateObjOrSub(srl, 0)
                foundObject = self.checkForAppropriateObjOrSub(srl, 1)
                if (foundSubject == '') or (foundObject == '') or (foundSubject == foundObject): 
                    continue
                if (srl['V'] == dobjVerb[i]) and (dobjWord[i] in foundObject) :
                    last_index = 0
                    totalPredicates = srl['V']
                    for idx in range(len(self.chunks)):
                        if (re.search(srl['V'], str(self.chunks[idx]), re.DOTALL | re.IGNORECASE)):
                            last_index = idx
                    for idx in range(last_index-1, -1, -1):
                        chunk_words = self.chunks[idx][0]
                        chunk_tag = self.chunks[idx][1]
                        if (chunk_tag in ['B-VP', 'E-VP', 'I-VP', 'S-VP']):
                            if (chunk_words in foundSubject): 
                                break
                            totalPredicates = chunk_words + ' ' + totalPredicates
                        else:
                            break
                    if (last_index + 1 < len(self.chunks)):
                        if (srl['V'] not in ['am', 'is', 'are', 'was', 'were', 'be']):
                            chunk_words = self.chunks[last_index + 1][0]
                            chunk_tag = self.chunks[last_index + 1][1]
                            if (chunk_tag == 'S-PRT') and (chunk_words not in foundSubject): 
                                totalPredicates = chunk_words + ' ' + totalPredicates
                    if (totalPredicates.startswith('to ')):
                        totalPredicates = totalPredicates[3:]
                    current_result = QDeconstructionResult()
                    current_result.predicate = totalPredicates
                    current_result.object = foundObject
                    current_result.subject = foundSubject
                    extraFieldsString = ''
                    if ('ARGM-LOC' in srl.keys()) and (srl['ARGM-LOC'] not in totalPredicates):
                        extraFieldsString = srl['ARGM-LOC']
                    if ('ARGM-TMP' in srl.keys()) and (srl['ARGM-TMP'] not in totalPredicates):
                        extraFieldsString = extraFieldsString + ' ' + srl['ARGM-TMP'] 
                    current_result.extraField = extraFieldsString
                    current_result.type = dobjSubType[i]
                    deconstruction_result.append(current_result)

        # Adjective Complement (ACOMP)
        for i in range(len(acompWord)):
            for srl in self.srls:
                if ('V' not in srl.keys()):
                    continue
                foundSubject = self.checkForAppropriateObjOrSub(srl, 0)
                foundObject = self.checkForAppropriateObjOrSub(srl, 1)
                if (foundSubject == '') or (foundObject == '') or (foundSubject == foundObject): 
                    continue
                if (srl['V'] == acompVerb[i]) and (acompWord[i] in foundObject) :
                    deconstruction_result.append(
                        QDeconstructionResult(
                            predicate='indicate',
                            object=foundObject,
                            subject=foundSubject,
                            extraField=srl.get('ARGM-LOC', '') + ' ' + srl.get('ARGM-TMP', ''),
                            type=acompSubType[i]
                        )
                    )

        # Attribute (ATTR)
        for i in range(len(attrWord)):
            for srl in self.srls:
                if ('V' not in srl.keys()):
                    continue
                foundSubject = self.checkForAppropriateObjOrSub(srl, 0)
                if (foundSubject == ''): 
                    continue
                if (srl['V'] == attrVerb[i]):
                    for k, v in srl.items():
                        if (k != 'V') and (v != foundSubject) and (attrWord[i] in v):
                            deconstruction_result.append(
                                QDeconstructionResult(
                                    predicate='describe',
                                    object=foundSubject,
                                    subject='you',
                                    extraField=srl.get('ARGM-LOC', '') + ' ' + srl.get('ARGM-TMP', ''),
                                    type=attrSubType[i]
                                )
                            )

        # Date (DATE)
        for i in range(len(dateWord)):
            for srl in self.srls:
                if ('V' not in srl.keys()):
                    continue
                foundSubject = self.checkForAppropriateObjOrSub(srl, 0)
                foundObject = self.checkForAppropriateObjOrSub(srl, 1)
                if (foundSubject == '') or (foundSubject == foundObject): 
                    continue
                for k, v in srl.items():
                    if (dateWord[i] in v) and (k != "V") and (k != "ARGM-TMP") and (v != foundSubject) and (v != foundObject):
                        last_index = 0
                        totalPredicates = srl['V']
                        for idx in range(len(self.chunks)):
                            if (re.search(srl['V'], str(self.chunks[idx]), re.DOTALL | re.IGNORECASE)):
                                last_index = idx
                        for idx in range(last_index-1, -1, -1):
                            chunk_words = self.chunks[idx][0]
                            chunk_tag = self.chunks[idx][1]
                            if (chunk_tag in ['B-VP', 'E-VP', 'I-VP', 'S-VP']):
                                if (chunk_words in foundSubject):
                                    break
                                totalPredicates = chunk_words + ' ' + totalPredicates 
                            else: 
                                break
                        if (totalPredicates.startswith('to ')):
                            totalPredicates = totalPredicates[3:]
                        current_result = QDeconstructionResult()
                        current_result.predicate = totalPredicates
                        current_result.object = foundObject
                        current_result.subject = foundSubject
                        extraFieldsString = ''
                        if ('ARGM-LOC' in srl.keys()) and (srl['ARGM-LOC'] not in totalPredicates):
                            extraFieldsString = srl['ARGM-LOC']
                        extraFieldsString = extraFieldsString.replace(dateWord[i], "")
                        current_result.extraField = extraFieldsString
                        current_result.type = dateSubType[i]
                        deconstruction_result.append(current_result)

        # Where (LOC)
        for i in range(len(whereWord)):
            for srl in self.srls:
                if ('V' not in srl.keys()):
                    continue
                foundSubject = self.checkForAppropriateObjOrSub(srl, 0)
                foundObject = self.checkForAppropriateObjOrSub(srl, 1)
                if (foundSubject == '') or (foundSubject == foundObject): 
                    continue
                last_index = 0
                totalPredicates = srl['V']
                for idx in range(len(self.chunks)):
                    if (re.search(srl['V'], str(self.chunks[idx]), re.DOTALL | re.IGNORECASE)):
                        last_index = idx
                for idx in range(last_index-1, -1, -1):
                    chunk_words = self.chunks[idx][0]
                    chunk_tag = self.chunks[idx][1]
                    if (chunk_tag in ['B-VP', 'E-VP', 'I-VP', 'S-VP']):
                        if (chunk_words in foundSubject): 
                            break
                        totalPredicates = chunk_words + ' ' + totalPredicates
                    else:
                        break
                if totalPredicates.startswith('to '):
                    totalPredicates= totalPredicates[3:]
                for k, v in srl.items():
                    if (whereWord[i] in v) and (k != 'V') and (k != 'ARGM-LOC') and (v != foundSubject):
                        realObj = ''
                        if (foundObject != '') and (v == foundObject):
                            for l in range(len(self.doc)-1):
                                if (self.doc[l].text not in v) or (self.doc[l+1].text not in v):
                                    continue
                                if (self.doc[l+1].text in whereWord[i]) and (self.doc[l].pos_ == 'ADP'):
                                    break
                                if(realObj == ''): 
                                    realObj = self.doc[l].text
                                else: 
                                    realObj += ' ' + self.doc[l].text 
                        else:
                            realObj = foundObject
                        current_result = QDeconstructionResult()
                        current_result.predicate = totalPredicates
                        if realObj.endswith(' the'):
                            realObj = realObj[:-4]
                        current_result.object = realObj
                        current_result.subject = foundSubject
                        extraFieldsString = ''
                        if ('ARGM-TMP' in srl.keys()) and (srl['ARGM-TMP'] not in totalPredicates):
                            extraFieldsString = srl['ARGM-TMP']
                        current_result.extraField = extraFieldsString
                        current_result.type = whereSubType[i]
                        deconstruction_result.append(current_result)

        # Prepositional Complement (PCOMP)
        for i in range(len(pcompWord)):
            for srl in self.srls:
                if ('V' not in srl.keys()): 
                    continue
                foundSubject = self.checkForAppropriateObjOrSub(srl, 0)
                last_index = 0
                totalPredicates = pcompPreposition[i]
                for idx in range(len(self.chunks)):
                    if (re.search(pcompPreposition[i], str(self.chunks[idx]), re.DOTALL | re.IGNORECASE)):
                        last_index = idx
                isMainVerbFound = False
                for idx in range(last_index-1, -1, -1):
                    if (not isMainVerbFound):
                        totalPredicates= str(self.chunks[idx][0]) + ' ' + totalPredicates 
                        if (self.chunks[idx][0] == srl['V']): 
                            isMainVerbFound = True
                    else:
                        chunk_words = self.chunks[idx][0]
                        chunk_tag = self.chunks[idx][1]
                        if (chunk_tag in ['B-VP', 'E-VP', 'I-VP', 'S-VP']):
                            if (chunk_words in foundSubject):
                                break
                            totalPredicates = chunk_words + ' ' + totalPredicates
                        else: break
                if (srl['V'] in totalPredicates):
                    if (self.checkForAppropriateObjOrSub(srl, 0) != ''):
                        foundObject = self.checkForAppropriateObjOrSub(srl, 1)
                        for k, v in srl.items():
                            if (k != 'V') and (pcompWord[i] in v) and (foundSubject != ''):
                                current_result = QDeconstructionResult()
                                current_result.predicate = totalPredicates
                                current_result.object = ''
                                current_result.subject = foundSubject
                                if (foundSubject == v) and (foundObject != '') and (foundObject != foundSubject):
                                    current_result.subject = foundObject
                                current_result.extraField = ''
                                current_result.type = pcompSubType[i]
                                deconstruction_result.append(current_result)

        # Number (CARDINAL)
        for i in range(len(numWord)):
            for srl in self.srls:
                if ('V' not in srl.keys()):
                    continue
                foundSubject = self.checkForAppropriateObjOrSub(srl, 0)
                foundObject = self.checkForAppropriateObjOrSub(srl, 1)
                if (foundSubject == '') or (foundSubject == foundObject):
                    continue
                last_index = 0
                totalPredicates = srl['V']
                for idx in range(len(self.chunks)):
                    if (re.search(srl['V'], str(self.chunks[idx]), re.DOTALL | re.IGNORECASE)):
                        last_index = idx
                for idx in range(last_index-1, -1, -1):
                    chunk_words = self.chunks[idx][0]
                    chunk_tag = self.chunks[idx][1]
                    if (chunk_tag in ['B-VP', 'E-VP', 'I-VP', 'S-VP']):
                        if (chunk_words in foundSubject): 
                            break
                        totalPredicates = chunk_words + ' ' + totalPredicates
                    else:
                        break
                if (last_index + 1 < len(self.chunks)):
                    if (srl['V'] not in ['am', 'is', 'are', 'was', 'were', 'be']):
                        chunk_words = self.chunks[last_index + 1][0]
                        chunk_tag = self.chunks[last_index + 1][1]
                        if (chunk_tag == 'S-PRT') and (chunk_words not in foundSubject): 
                            totalPredicates = chunk_words + ' ' + totalPredicates
                if (totalPredicates.startswith('to ')):
                    totalPredicates = totalPredicates[3:]
                for k, v in srl.items():
                    if (numWord[i] in v) and (k != 'V'):
                        found_idx = -1
                        valueArray = v.split(" ")
                        for l in range(len(valueArray)):
                            if (numWord[i] in valueArray[l]):
                                found_idx = l
                        valueArrayFirstPart = valueArray[found_idx+1:]
                        valueArrayLastPart = valueArray[:found_idx]
                        valueFirstPart = ''
                        for l in range(len(valueArrayFirstPart)):
                            if (valueFirstPart == ''): 
                                valueFirstPart = valueArrayFirstPart[l]
                            else: 
                                valueFirstPart = valueFirstPart + ' ' + valueArrayFirstPart[l]
                        valueLastPart = ''
                        for l in range(len(valueArrayLastPart)):
                            if (valueLastPart == ''): 
                                valueLastPart = valueArrayLastPart[l]
                            elif (l == (len(valueArrayLastPart)-1)) and (valueArrayLastPart[l] == 'the'):  
                                break
                            else: 
                                valueLastPart = valueLastPart + ' ' + valueArrayLastPart[l]
                        current_result = QDeconstructionResult()
                        current_result.predicate = totalPredicates
                        current_result.object = valueLastPart + ' '+ foundObject + ' ' + extraFieldsString
                        current_result.subject = valueFirstPart
                        current_result.extraField = ''
                        current_result.type = numSubType[i]
                        extraFieldsString = ''
                        if ('ARGM-LOC' in srl.keys()) and (srl['ARGM-LOC'] not in totalPredicates):
                            extraFieldsString = srl['ARGM-LOC']
                        if ('ARGM-TMP' in srl.keys()) and (srl['ARGM-TMP'] not in totalPredicates):
                            extraFieldsString = extraFieldsString + ' ' + srl['ARGM-TMP'] 
                        if (v == foundObject) and (foundSubject != ''):
                            current_result.object = extraFieldsString
                            current_result.extraFields = foundSubject
                        deconstruction_result.append(current_result)
    
        # Person (PERSON)
        for i in range(len(personWord)):
            for srl in self.srls:
                if ('V' not in srl.keys()):
                    continue
                foundSubject = self.checkForAppropriateObjOrSub(srl, 0)
                foundObject = self.checkForAppropriateObjOrSub(srl, 1)
                if (foundSubject == '') or (foundSubject == foundObject): 
                    continue
                for k, v in srl.items():
                    if (personWord[i] in v) and (k != 'V') and (v == foundSubject) and (v != foundObject):
                        last_index = 0
                        totalPredicates = srl['V']
                        for idx in range(len(self.chunks)):
                            if (re.search(srl['V'], str(self.chunks[idx]), re.DOTALL | re.IGNORECASE)):
                                last_index = idx
                        for idx in range(last_index-1, -1, -1):
                            chunk_words = self.chunks[idx][0]
                            chunk_tag = self.chunks[idx][1]
                            if (chunk_tag in ['B-VP', 'E-VP', 'I-VP', 'S-VP']):
                                if (chunk_words in foundSubject): 
                                    break
                                totalPredicates = chunk_words + ' ' + totalPredicates
                            else:
                                break
                        if (last_index + 1 < len(self.chunks)):
                            if (srl['V'] not in ['am', 'is', 'are', 'was', 'were', 'be']):
                                chunk_words = self.chunks[last_index + 1][0]
                                chunk_tag = self.chunks[last_index + 1][1]
                                if (chunk_tag == 'S-PRT') and (chunk_words not in foundSubject): 
                                    totalPredicates = chunk_words + ' ' + totalPredicates
                        if totalPredicates[:3].startswith('to '):
                            totalPredicates= totalPredicates[3:]

                        relativeClauseDet = False
                        otherNounsDet = False
                        if (utils.getValueBetweenTexts(v, personWord[i], ',') == ' '): 
                            relativeClauseDet = True
                        if (utils.getValueBetweenTexts(v, personWord[i], 'who') == ' '): 
                            relativeClauseDet = True
                        if (utils.getValueBetweenTexts(v, personWord[i], 'that') == ' '): 
                            relativeClauseDet = True
                        if (utils.getValueBetweenTexts(v, personWord[i], 'whose') == ' '): 
                            relativeClauseDet = True
                        if (not relativeClauseDet):
                            modifSrl = v.replace("' " + personWord[i] + " '", '')
                            modifSrl = v.replace('" ' + personWord[i]+ ' "' , '')
                            modifSrl = v.replace(personWord[i], '')
                            modifSrl = modifSrl.split(' ')
                            for m in range(len(modifSrl)):
                                for chunk in self.chunks:
                                    chunk_words = chunk[0]
                                    chunk_tag = chunk[1]
                                    if (chunk_words == modifSrl[m]) and (len(chunk_words) > 1):
                                        if (chunk_tag in ['B-NP', 'E-NP', 'I-NP', 'S-NP']):
                                            otherNounsDet = True
                                            break
                        extraFieldsString = ''
                        if ('ARGM-LOC' in srl.keys()) and (srl['ARGM-LOC'] not in totalPredicates):
                            extraFieldsString = srl['ARGM-LOC']
                        if ('ARGM-TMP' in srl.keys()) and (srl['ARGM-TMP'] not in totalPredicates):
                            extraFieldsString = extraFieldsString + ' ' + srl['ARGM-TMP'] 
                        if (not otherNounsDet):
                            deconstruction_result.append(
                                QDeconstructionResult(
                                    predicate=totalPredicates,
                                    object=foundObject,
                                    subject='',
                                    extraField=extraFieldsString,
                                    type=personSubType[i]
                                )
                            )

        for srl in self.srls:
            if ('V' not in srl.keys()):
                continue
            foundSubject = self.checkForAppropriateObjOrSub(srl, 0)
            foundObject = self.checkForAppropriateObjOrSub(srl, 1)
            if (foundSubject == '') or (foundSubject == foundObject): continue
            last_index = 0
            totalPredicates = srl['V']
            for idx in range(len(self.chunks)):
                if (re.search(srl['V'], str(self.chunks[idx]), re.DOTALL | re.IGNORECASE)):
                    last_index = idx
            for idx in range(last_index-1, -1, -1):
                chunk_words = self.chunks[idx][0]
                chunk_tag = self.chunks[idx][1]
                if (chunk_tag in ['B-VP', 'E-VP', 'I-VP', 'S-VP']):
                    if (chunk_words in foundSubject): 
                        break
                    totalPredicates = chunk_words + ' ' + totalPredicates
                else:
                    break
            if (last_index + 1 < len(self.chunks)):
                if (srl['V'] not in ['am', 'is', 'are', 'was', 'were', 'be']):
                    chunk_words = self.chunks[last_index + 1][0]
                    chunk_tag = self.chunks[last_index + 1][1]
                    if (chunk_tag == 'S-PRT') and (chunk_words not in foundSubject): 
                        totalPredicates = chunk_words + ' ' + totalPredicates
            if (totalPredicates.startswith('to ')):
                totalPredicates= totalPredicates[3:]

            if 'ARGM-CAU' in srl.keys():
                current_result = QDeconstructionResult()
                current_result.predicate = totalPredicates
                current_result.object = foundObject
                current_result.subject = foundSubject
                extraFieldsString = ''
                if ('ARGM-LOC' in srl.keys()) and (srl['ARGM-LOC'] not in totalPredicates):
                    extraFieldsString = srl['ARGM-LOC']
                if ('ARGM-TMP' in srl.keys()) and (srl['ARGM-TMP'] not in totalPredicates):
                    extraFieldsString = extraFieldsString + ' ' + srl['ARGM-TMP'] 
                current_result.extraField = extraFieldsString
                current_result.type = 'why'
                deconstruction_result.append(current_result)
            if 'ARGM-PNC' in srl.keys():
                current_result = QDeconstructionResult()
                current_result.predicate = totalPredicates
                current_result.object = foundObject
                current_result.subject = foundSubject
                extraFieldsString = ''
                if ('ARGM-LOC' in srl.keys()) and (srl['ARGM-LOC'] not in totalPredicates):
                    extraFieldsString = srl['ARGM-LOC']
                if ('ARGM-TMP' in srl.keys()) and (srl['ARGM-TMP'] not in totalPredicates):
                    extraFieldsString = extraFieldsString + ' ' + srl['ARGM-TMP'] 
                current_result.extraField = extraFieldsString
                current_result.type = 'purpose'
                deconstruction_result.append(current_result)
            if 'ARGM-MNR' in srl:
                current_result = QDeconstructionResult()
                current_result.predicate = totalPredicates
                current_result.object = foundObject
                current_result.subject = foundSubject
                extraFieldsString = ''
                if ('ARGM-LOC' in srl.keys()) and (srl['ARGM-LOC'] not in totalPredicates):
                    extraFieldsString = srl['ARGM-LOC']
                if ('ARGM-TMP' in srl.keys()) and (srl['ARGM-TMP'] not in totalPredicates):
                    extraFieldsString = extraFieldsString + ' ' + srl['ARGM-TMP'] 
                current_result.extraField = extraFieldsString
                current_result.type = 'ARG_MANNER_TYPE'
                current_result.key_answer = srl['ARGM-MNR']
                deconstruction_result.append(current_result)
            if 'ARGM-TMP' in srl:
                current_result = QDeconstructionResult()
                current_result.predicate = totalPredicates
                current_result.object = foundObject
                current_result.subject = foundSubject
                extraFieldsString = ''
                if ('ARGM-LOC' in srl.keys()) and (srl['ARGM-LOC'] not in totalPredicates):
                    extraFieldsString = srl['ARGM-LOC']
                current_result.extraField = extraFieldsString
                current_result.type = 'ARG_TEMPORAL_TYPE'
                current_result.key_answer = srl['ARGM-TMP']
                deconstruction_result.append(current_result)
            if 'ARGM-LOC' in srl:
                current_result = QDeconstructionResult()
                current_result.predicate = totalPredicates
                current_result.object = foundObject
                current_result.subject = foundSubject
                extraFieldsString = ''
                if ('ARGM-TMP' in srl.keys()) and (srl['ARGM-TMP'] not in totalPredicates):
                    extraFieldsString = srl['ARGM-TMP']
                current_result.extraField = extraFieldsString
                current_result.type = 'LOC'
                deconstruction_result.append(current_result)

            if ('V' not in srl.keys()):
                continue
            foundSubject = self.checkForAppropriateObjOrSub(srl, 0)
            foundObject = self.checkForAppropriateObjOrSub(srl, 1)
            if (foundSubject == '') or (foundSubject == foundObject) or (foundObject == ''):
                continue
            last_index = 0
            totalPredicates = srl['V']
            for idx in range(len(self.chunks)):
                if (re.search(srl['V'], str(self.chunks[idx]), re.DOTALL | re.IGNORECASE)):
                    last_index = idx
            for idx in range(last_index-1, -1, -1):
                chunk_words = self.chunks[idx][0]
                chunk_tag = self.chunks[idx][1]
                if (chunk_tag in ['B-VP', 'E-VP', 'I-VP', 'S-VP']):
                    if (chunk_words in foundSubject): 
                        break
                    totalPredicates = chunk_words + ' ' + totalPredicates
                else:
                    break
            if (last_index + 1 < len(self.chunks)):
                if (srl['V'] not in ['am', 'is', 'are', 'was', 'were', 'be']):
                    chunk_words = self.chunks[last_index + 1][0]
                    chunk_tag = self.chunks[last_index + 1][1]
                    if (chunk_tag == 'S-PRT') and (chunk_words not in foundSubject): 
                        totalPredicates = chunk_words + ' ' + totalPredicates
            if totalPredicates.startswith('to '):
                totalPredicates= totalPredicates[3:]
            current_result = QDeconstructionResult()
            current_result.predicate = totalPredicates
            current_result.object = foundObject
            current_result.subject = foundSubject
            extraFieldsString = ''
            if ('ARGM-LOC' in srl.keys()) and (srl['ARGM-LOC'] not in totalPredicates):
                extraFieldsString = srl['ARGM-LOC']
            if ('ARGM-TMP' in srl.keys()) and (srl['ARGM-TMP'] not in totalPredicates):
                extraFieldsString = extraFieldsString + ' ' + srl['ARGM-TMP'] 
            current_result.extraField = extraFieldsString
            current_result.type = 'direct'
            deconstruction_result.append(current_result)
        
        return deconstruction_result
