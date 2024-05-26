from typing import List, Tuple

import spacy
import spacy.tokenizer
import spacy.tokens
from spacy.tokens import Token
from helper import Helper


class QDeconstructionResult:
    def __init__(
            self, 
            predicate: List[Token] = [], 
            subject: List[Token] = [], 
            object: List[Token] = [], 
            extra_field: List[Token] = [], 
            type='', 
            key_answer: List[Token] = [], 
        ):
        self.predicate = predicate
        self.subject = subject
        self.object = object
        self.extra_field = extra_field
        self.type = type
        self.key_answer = key_answer

    def __str__(self) -> str:
        predicate_text = Helper.merge_tokens(self.predicate)
        subject_text = Helper.merge_tokens(self.subject)
        
        object_text = Helper.merge_tokens(self.object)
        extra_field_text = Helper.merge_tokens(self.extra_field)
        key_answer_text = Helper.merge_tokens(self.key_answer)
        return f"Predicate: {predicate_text}\nSubject: {subject_text}\nObject: {object_text}\nExtra Field: {extra_field_text}\nType: {self.type}\nKey Answer: {key_answer_text}"

class QDeconstructor:

    def __init__(self, doc: spacy.tokens.Doc, idx_srls: List[dict], verbose=False):
        self.doc = doc
        self.idx_srls = idx_srls
        self._prepare_data(verbose)
            
    
    def _prepare_data(self, verbose=False):
        self.ners = []
        for ent in self.doc.ents:
            if verbose:
                print(ent.text, ":", ent.label_)
            ner_tokens = []
            for token in self.doc:
                if ent.start_char <= token.idx < ent.end_char:
                    ner_tokens.append(token)
            ner_label = ''
            if (ent.label_ == 'DATE' and ent.text.find('year old') == -1 and ent.text.find('years old') == -1 ):
                ner_label = 'DATE'
            elif ent.label_ == 'CARDINAL':
                ner_label = 'CARDINAL'
            elif ent.label_ == 'PERSON':
                ner_label = 'PERSON'
            elif ent.label_ == 'FACILITY' or ent.label_ == 'ORG' or ent.label_ == 'GPE' or ent.label_ == 'LOC':
                ner_label = 'LOC'
            if ner_label:
                self.ners.append({
                    'label': ner_label,
                    'tokens': ner_tokens
                })

        self.srls = []
        for srl in self.idx_srls:
            mod_srl = {}
            for k, v in srl.items():
                span = []
                for token in self.doc:
                    if v[0] <= token.idx < v[1]:
                        span.append(token)
                mod_srl[k] = span
            self.srls.append(mod_srl)
            
        self.antecedents = []
        for cluster in self.doc._.coref_clusters:
            spans_in_cluster = []
            for span in cluster:
                tokens_in_span = []
                for token in self.doc:
                    if span[0] <= token.idx < span[1]:
                        tokens_in_span.append(token)
                spans_in_cluster.append(tokens_in_span)
            self.antecedents.append(spans_in_cluster)       
            
            
        self.noun_phrases = []
        for noun_phrase in self.doc.noun_chunks:
            if verbose:
                print(noun_phrase.text)
            span = []
            for token in self.doc:
                if noun_phrase.start_char <= token.idx < noun_phrase.end_char:
                    span.append(token)
            self.noun_phrases.append(span)
        
        
    def _get_antecedent(self, token: Token) -> Tuple[List[Token], List[Token]]:
        """
        Get the antecedent of a token in a coreference cluster. If the token is not in any cluster, return the token itself.
        Return both the antecedent and the span of the token in the cluster.

        Args:
        -----
        token: Token
            Token to check for antecedent
            
        Returns:
        --------
        Tuple[List[Token], List[Token]]
            
        """
        for cluster in self.antecedents:
            for span in cluster:
                if token in span:
                    return cluster[0], span
        return [token], [token]


    def _replace_antecedent(self, token: Token, original_tokens: List[Token]) -> List[Token]:
        antecedent_tokens, span_tokens = self._get_antecedent(token)
        modified_tokens = []
        replaced = False
        for token in original_tokens:
            if token not in span_tokens:
                modified_tokens.append(token)
            elif not replaced:
                modified_tokens.extend(antecedent_tokens)
                replaced = True
        return modified_tokens
    
    
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
    
    
    def deconstruct(self):
        """
        Currently supports the following dependency parse types: 
            dative, dobj, acomp, attr, pcomp, nsubj, nsubjpass
        Curerntly supports the following NER types:
            DATE, LOC, CARDINAL, PERSON (LOC consists of LOC, ORG, GPE, FACILITY)

        Returns:
        --------
        List[QDeconstructionResult]
        """
        deconstruction_result : List[QDeconstructionResult] = []

        # Dependency parsing rules
        for word in self.doc:
            if word.dep_ == 'dative':
                deconstruction_result.extend(self._deconstruct_dative(word))
            elif word.dep_ == 'dobj':
                deconstruction_result.extend(self._deconstruct_dobj(word))
            elif word.dep_ == 'acomp':
                deconstruction_result.extend(self._deconstruct_acomp(word))
            elif word.dep_ == 'attr':
                deconstruction_result.extend(self._deconstruct_attr(word))
            elif word.dep_ == 'pcomp':
                deconstruction_result.extend(self._deconstruct_pcomp(word))
            elif word.dep_ == 'nsubj' or word.dep_ == 'nsubjpass':
                deconstruction_result.extend(self._deconstruct_nsubj(word))
        
        # Named entity recognition rules
        for ner in self.ners:
            if ner['label'] == 'DATE':
                deconstruction_result.extend(self._deconstruct_ner_date(ner))
            elif ner['label'] == 'LOC':
                deconstruction_result.extend(self._deconstruct_ner_loc(ner))
            elif ner['label'] == 'CARDINAL':
                deconstruction_result.extend(self._deconstruct_ner_cardinal(ner))
            elif ner['label'] == 'PERSON':
                deconstruction_result.extend(self._deconstruct_ner_person(ner))

        # Semantic role labeling rules
        for srl in self.srls:
            if ('V' not in srl.keys()):
                continue
            
            found_subject_tokens = self.checkForAppropriateObjOrSub(srl, 0)
            found_object_tokens = self.checkForAppropriateObjOrSub(srl, 1)
            
            found_subject_text = Helper.merge_tokens(found_subject_tokens)
            found_object_text = Helper.merge_tokens(found_object_tokens)
                
            is_passive = False
            if len(srl['V']) == 1:
                full_predicate = Helper.find_full_predicate(srl['V'][0])
                for child in srl['V'][0].children:
                    if child.dep_ == 'auxpass':
                        is_passive = True
                        break
            else:
                for pred_token in srl['V']:
                    if pred_token.tag_.startswith('VB'):
                        for child in pred_token.children:
                            if child.dep_ == 'auxpass':
                                is_passive = True
                                break
                        full_predicate = Helper.find_full_predicate(pred_token)
                        
            full_predicate_text = Helper.merge_tokens(full_predicate)
            
            if found_subject_text and found_object_text and found_subject_text != found_object_text:
                # Direct question
                current_result = QDeconstructionResult()
                current_result.predicate = full_predicate
                if is_passive:
                    current_result.subject = found_object_tokens
                    current_result.object = found_subject_tokens
                else:
                    current_result.object = found_object_tokens
                    current_result.subject = found_subject_tokens
                extra_tokens = []
                if ('ARGM-LOC' in srl.keys()) and (Helper.merge_tokens(srl['ARGM-LOC']) not in full_predicate_text):
                    extra_tokens.extend(srl['ARGM-LOC'])
                if ('ARGM-TMP' in srl.keys()) and (Helper.merge_tokens(srl['ARGM-TMP']) not in full_predicate_text):
                    extra_tokens.extend(srl['ARGM-TMP']) 
                current_result.extra_field = extra_tokens
                current_result.type = 'direct'
                deconstruction_result.append(current_result)

                # Causal question
                if 'ARGM-CAU' in srl.keys():
                    current_result = QDeconstructionResult()
                    current_result.predicate = full_predicate
                    current_result.object = found_object_tokens
                    current_result.subject = found_subject_tokens
                    extra_tokens = []
                    if ('ARGM-LOC' in srl.keys()) and (Helper.merge_tokens(srl['ARGM-LOC']) not in full_predicate_text):
                        extra_tokens.extend(srl['ARGM-LOC'])
                    if ('ARGM-TMP' in srl.keys()) and (Helper.merge_tokens(srl['ARGM-TMP']) not in full_predicate_text):
                        extra_tokens.extend(srl['ARGM-TMP']) 
                    current_result.extra_field = extra_tokens
                    current_result.type = 'srl_causal'
                    current_result.key_answer = srl['ARGM-CAU']
                    deconstruction_result.append(current_result)
                    
                # Purpose question
                if 'ARGM-PNC' in srl.keys() or 'ARGM-PRP' in srl.keys():
                    current_result = QDeconstructionResult()
                    current_result.predicate = full_predicate
                    current_result.object = found_object_tokens
                    current_result.subject = found_subject_tokens
                    extra_tokens = []
                    if ('ARGM-LOC' in srl.keys()) and (Helper.merge_tokens(srl['ARGM-LOC']) not in full_predicate_text):
                        extra_tokens.extend(srl['ARGM-LOC'])
                    if ('ARGM-TMP' in srl.keys()) and (Helper.merge_tokens(srl['ARGM-TMP']) not in full_predicate_text):
                        extra_tokens.extend(srl['ARGM-TMP']) 
                    current_result.extra_field = extra_tokens
                    current_result.type = 'srl_purpose'
                    if 'ARGM-PNC' in srl.keys():
                        current_result.key_answer = srl['ARGM-PNC']
                    elif 'ARGM-PRP' in srl.keys():
                        current_result.key_answer = srl['ARGM-PRP']
                    else:
                        current_result.key_answer = []
                    deconstruction_result.append(current_result)
                    
                # Manner question
                if 'ARGM-MNR' in srl:
                    current_result = QDeconstructionResult()
                    current_result.predicate = full_predicate
                    current_result.object = found_object_tokens
                    current_result.subject = found_subject_tokens
                    extra_tokens = []
                    if ('ARGM-LOC' in srl.keys()) and (Helper.merge_tokens(srl['ARGM-LOC']) not in full_predicate_text):
                        extra_tokens.extend(srl['ARGM-LOC'])
                    if ('ARGM-TMP' in srl.keys()) and (Helper.merge_tokens(srl['ARGM-TMP']) not in full_predicate_text):
                        extra_tokens.extend(srl['ARGM-TMP']) 
                    current_result.extra_field = extra_tokens
                    current_result.type = 'srl_manner'
                    current_result.key_answer = srl['ARGM-MNR']
                    deconstruction_result.append(current_result)
                    
                # Temporal question
                if 'ARGM-TMP' in srl:
                    current_result = QDeconstructionResult()
                    current_result.predicate = full_predicate
                    current_result.object = found_object_tokens
                    current_result.subject = found_subject_tokens
                    extra_tokens = []
                    if ('ARGM-LOC' in srl.keys()) and (Helper.merge_tokens(srl['ARGM-LOC']) not in full_predicate_text):
                        extra_tokens.extend(srl['ARGM-LOC'])
                    current_result.extra_field = extra_tokens
                    current_result.type = 'srl_temporal'
                    current_result.key_answer = srl['ARGM-TMP']
                    deconstruction_result.append(current_result)
                    
                # Locative question
                if 'ARGM-LOC' in srl:
                    current_result = QDeconstructionResult()
                    current_result.predicate = full_predicate
                    current_result.object = found_object_tokens
                    current_result.subject = found_subject_tokens
                    extra_tokens = []
                    if ('ARGM-TMP' in srl.keys()) and (Helper.merge_tokens(srl['ARGM-TMP']) not in full_predicate_text):
                        extra_tokens.extend(srl['ARGM-TMP'])
                    current_result.extra_field = extra_tokens
                    current_result.type = 'srl_locative'
                    current_result.key_answer = srl['ARGM-LOC']
                    deconstruction_result.append(current_result)
       
        return deconstruction_result
    
    
    def _deconstruct_dative(self, dative_word: Token) -> List[QDeconstructionResult]:
        """Deconstructs a dative word (or indirect object)"""
        
        if dative_word.dep_ != 'dative':
            raise ValueError(f"Word {dative_word} is not of type dative")

        deconstruction_result : List[QDeconstructionResult] = []
        for dobj_word in self.doc:
            if dobj_word.dep_ == 'dobj':
                if dobj_word.head != dative_word.head:
                    continue
                for srl in self.srls:
                    if ('V' not in srl.keys()): 
                        continue
                    
                    found_subject_tokens = self.checkForAppropriateObjOrSub(srl, 0)
                    found_object_tokens = self.checkForAppropriateObjOrSub(srl, 1)
                    found_indirect_tokens = self.checkForAppropriateObjOrSub(srl, 2)
                    
                    found_subject_text = Helper.merge_tokens(found_subject_tokens)    
                    found_object_text = Helper.merge_tokens(found_object_tokens)
                    found_indirect_text = Helper.merge_tokens(found_indirect_tokens)
                    
                    if (found_subject_text == '') or (found_object_text == '') or (found_indirect_text == ''):
                        continue
                    if  (found_subject_text == found_object_text) or (found_indirect_text == found_object_text) or (found_indirect_text == found_subject_text):
                        continue
                        
                    if (dobj_word.head in srl['V']) and (dobj_word.text in found_object_text) and (dative_word.text in found_indirect_text):
                        if len(srl['V']) == 1:
                            full_predicate = Helper.find_full_predicate(srl['V'][0])
                        else:
                            for pred_token in srl['V']:
                                if pred_token.tag_.startswith('VB'):
                                    full_predicate = Helper.find_full_predicate(pred_token)
                        full_predicate_text = Helper.merge_tokens(full_predicate)
                        current_result = QDeconstructionResult()
                        current_result.predicate = full_predicate
                        current_result.object = found_object_tokens
                        current_result.subject = found_subject_tokens
                        current_result.key_answer = found_indirect_tokens
                        extra_tokens = []
                        if ('ARGM-LOC' in srl.keys()) and (Helper.merge_tokens(srl['ARGM-LOC']) not in full_predicate_text):
                            extra_tokens.extend(srl['ARGM-LOC'])
                        if ('ARGM-TMP' in srl.keys()) and (Helper.merge_tokens(srl['ARGM-TMP']) not in full_predicate_text):
                            extra_tokens.extend(srl['ARGM-TMP'])
                        current_result.extra_field = extra_tokens
                        current_result.type = "dative_question"
                        deconstruction_result.append(current_result)  
        return deconstruction_result


    def _deconstruct_dobj(self, dobj_word: Token) -> List[QDeconstructionResult]:
        """Deconstructs a direct object"""
        
        if dobj_word.dep_ != 'dobj':
            raise ValueError(f"Word {dobj_word} is not of type dobj")
        
        deconstruction_result : List[QDeconstructionResult] = []
        for srl in self.srls:
            if ('V' not in srl.keys()):
                continue
            found_subject_tokens = self.checkForAppropriateObjOrSub(srl, 0)
            found_object_tokens = self.checkForAppropriateObjOrSub(srl, 1)
            found_subject_text = Helper.merge_tokens(found_subject_tokens)
            found_object_text = Helper.merge_tokens(found_object_tokens)

            if (found_subject_text == '') or (found_object_text == '') or (found_subject_text == found_object_text):
                continue
            
            if (dobj_word.head in srl['V']) and (dobj_word.text in found_object_text):
                if len(srl['V']) == 1:
                    full_predicate = Helper.find_full_predicate(srl['V'][0])
                else:
                    for pred_token in srl['V']:
                        if pred_token.tag_.startswith('VB'):
                            full_predicate = Helper.find_full_predicate(pred_token)
                full_predicate_text = Helper.merge_tokens(full_predicate)
                current_result = QDeconstructionResult()
                current_result.predicate = full_predicate
                current_result.object = found_object_tokens
                current_result.subject = found_subject_tokens
                current_result.key_answer = found_object_tokens
                extra_tokens = []
                if ('ARGM-LOC' in srl.keys()) and (Helper.merge_tokens(srl['ARGM-LOC']) not in full_predicate_text):
                    extra_tokens.extend(srl['ARGM-LOC'])
                if ('ARGM-TMP' in srl.keys()) and (Helper.merge_tokens(srl['ARGM-TMP']) not in full_predicate_text):
                    extra_tokens.extend(srl['ARGM-TMP'])
                current_result.extra_field = extra_tokens
                current_result.type = "dobj_question"
                deconstruction_result.append(current_result)
        return deconstruction_result
    
    
    def _deconstruct_acomp(self, acomp_word: Token) -> List[QDeconstructionResult]:
        """Deconstructs an adjective complement"""
        
        if acomp_word.dep_ != 'acomp':
            raise ValueError(f"Word {acomp_word} is not of type acomp")
        deconstruction_result : List[QDeconstructionResult] = []
        for srl in self.srls:
            if ('V' not in srl.keys()):
                continue
            
            found_subject_tokens = self.checkForAppropriateObjOrSub(srl, 0)
            found_object_tokens = self.checkForAppropriateObjOrSub(srl, 1)
            
            found_subject_text = Helper.merge_tokens(found_subject_tokens)
            found_object_text = Helper.merge_tokens(found_object_tokens)
                        
            if (found_subject_text == '') or (found_object_text == '') or (found_subject_text == found_object_text):
                continue
            
            verb_text = Helper.merge_tokens(srl['V'])
            if (verb_text == acomp_word.head.text) and (acomp_word.text in found_object_text):
                current_result = QDeconstructionResult(
                    predicate=[],
                    object=found_object_tokens,
                    subject=found_subject_tokens,
                    extra_field=[],
                    type="acomp_question",
                    key_answer=[acomp_word]
                )
                extra_tokens = []
                if ('ARGM-LOC' in srl.keys()):
                    extra_tokens.extend(srl['ARGM-LOC'])
                if ('ARGM-TMP' in srl.keys()):
                    extra_tokens.extend(srl['ARGM-TMP'])
                current_result.extra_field = extra_tokens
                deconstruction_result.append(current_result)
        return deconstruction_result
    
    
    def _deconstruct_attr(self, attr_word: Token) -> List[QDeconstructionResult]:
        """Deconstructs an attribute"""
        
        if attr_word.dep_ != 'attr':
            raise ValueError(f"Word {attr_word} is not of type attr")
        
        deconstruction_result : List[QDeconstructionResult] = []
        for srl in self.srls:
            if ('V' not in srl.keys()):
                continue
            
            found_subject_tokens = self.checkForAppropriateObjOrSub(srl, 0)
            found_subject_text = Helper.merge_tokens(found_subject_tokens)
            
            if (found_subject_text == ''):
                continue
            
            verb_text = Helper.merge_tokens(srl['V'])
            if (verb_text == attr_word.head.text):
                for k, v in srl.items():
                    text_v = Helper.merge_tokens(v)
                    if (k != 'V') and (text_v != found_subject_text) and (attr_word.text in text_v):
                        extra_tokens = []
                        if ('ARGM-LOC' in srl.keys()):
                            extra_tokens.extend(srl['ARGM-LOC'])
                        if ('ARGM-TMP' in srl.keys()):
                            extra_tokens.extend(srl['ARGM-TMP'])
                        deconstruction_result.append(
                            QDeconstructionResult(
                                predicate=[],
                                object=found_subject_tokens,
                                subject=[],
                                extra_field=extra_tokens,
                                type="attr_question",
                                key_answer=[attr_word]
                            )
                        )
        return deconstruction_result
    
    
    def _deconstruct_pcomp(self, pcomp_word: Token) -> List[QDeconstructionResult]:
        """Deconstructs a prepositional complement"""
        
        if pcomp_word.dep_ != 'pcomp':
            raise ValueError(f"Word {pcomp_word} is not of type pcomp")
        
        deconstruction_result : List[QDeconstructionResult] = []
        for srl in self.srls:
            if ('V' not in srl.keys()): 
                continue
            found_subject_tokens = self.checkForAppropriateObjOrSub(srl, 0)
            found_subject_text = Helper.merge_tokens(found_subject_tokens)
            if len(srl['V']) == 1:
                full_predicate = Helper.find_full_predicate(srl['V'][0])
            else:
                for pred_token in srl['V']:
                    if pred_token.tag_.startswith('VB'):
                        full_predicate = Helper.find_full_predicate(pred_token)
            if (found_subject_text != ''):
                found_object_tokens = self.checkForAppropriateObjOrSub(srl, 1)
                found_object_text = Helper.merge_tokens(found_object_tokens)
                for k, v in srl.items():
                    if (k != 'V') and (pcomp_word.text in Helper.merge_tokens(v)) and (found_subject_text != ''):
                        # Remove unnecessary words (before pcomp) in pcomp and add them to the extra_field
                        pcomp_idx = v.index(pcomp_word)
                        # check passive voice
                        is_passive = False
                        for subj_tok in found_subject_tokens:
                            if subj_tok.dep_ == 'nsubjpass':
                                is_passive = True
                                break
                        extra_tokens = []
                        for i in range(pcomp_idx):
                            extra_tokens.append(v[i])
                                
                        current_result = QDeconstructionResult()
                        current_result.predicate = full_predicate
                        current_result.object = []
                        current_result.subject = found_subject_tokens
                        if (found_subject_text == Helper.merge_tokens(v)) and (found_object_text != '') and (found_object_text != found_subject_text):
                            current_result.subject = found_object_tokens
                        if not is_passive:
                            v = v[pcomp_idx:]
                        current_result.extra_field = extra_tokens
                        current_result.type = "pcomp_question"
                        current_result.key_answer = v
                        deconstruction_result.append(current_result)  
        return deconstruction_result
    
    
    def _deconstruct_nsubj(self, nsubj_word: Token) -> List[QDeconstructionResult]:
        """Deconstructs a nominal subject"""
        
        if nsubj_word.dep_ not in ['nsubj', 'nsubjpass']:
            raise ValueError(f"Expected nsubj or nsubjpass, got {nsubj_word.text} with type of {nsubj_word.dep_}")
        
        deconstruction_result: List[QDeconstructionResult] = []
        
        predicate = nsubj_word.head
        full_predicate = Helper.find_full_predicate(predicate, include_deps=['advmod'])
        full_subject_tokens = Helper.find_full_subject(nsubj_word)
        full_subject = self._replace_antecedent(nsubj_word, full_subject_tokens)
        
        for token in predicate.children:
            if token.dep_ == 'dobj':
                possible_objects = Helper.find_full_direct_object(token)
                for full_object in possible_objects:
                    full_object_tokens = self._replace_antecedent(token, full_object)
                    current_result = QDeconstructionResult()
                    current_result.predicate = full_predicate
                    current_result.subject = []
                    current_result.object = full_object_tokens
                    current_result.extra_field = []
                    current_result.type = "nsubj_question"
                    current_result.key_answer = full_subject
                    deconstruction_result.append(current_result)
        return deconstruction_result
    
    
    def _deconstruct_ner_date(self, date_ner: dict) -> List[QDeconstructionResult]:
        """Deconstructs a date named entity. Dict should have the following keys: 'label', 'tokens'"""

        deconstruction_result : List[QDeconstructionResult] = []
        date_text = Helper.merge_tokens(date_ner['tokens'])
        for srl in self.srls:
            if ('V' not in srl.keys()):
                continue
            
            found_subject_tokens = self.checkForAppropriateObjOrSub(srl, 0)
            found_object_tokens = self.checkForAppropriateObjOrSub(srl, 1)
            found_subject_text = Helper.merge_tokens(found_subject_tokens)
            found_object_text = Helper.merge_tokens(found_object_tokens)
            
            if (found_subject_text == '') or (found_subject_text == found_object_text): 
                continue
            for k, v in srl.items():
                if (k != 'V') and (k != 'ARGM-TMP'):
                    text_v = Helper.merge_tokens(v)
                    if (date_text in text_v) and (text_v != found_subject_text) and (text_v != found_object_text):
                        if len(srl['V']) == 1:
                            full_predicate = Helper.find_full_predicate(srl['V'][0])
                        else:
                            for pred_token in srl['V']:
                                if pred_token.tag_.startswith('VB'):
                                    full_predicate = Helper.find_full_predicate(pred_token)
                        full_predicate_text = Helper.merge_tokens(full_predicate)
                        current_result = QDeconstructionResult()
                        current_result.predicate = full_predicate
                        current_result.object = found_object_tokens
                        current_result.subject = found_subject_tokens
                        extra_tokens = []
                        if ('ARGM-LOC' in srl.keys()) and (Helper.merge_tokens(srl['ARGM-LOC']) not in full_predicate):
                            extra_tokens.extend(srl['ARGM-LOC'])
                        current_result.extra_field = [tok for tok in extra_tokens if tok not in date_ner['tokens']]
                        current_result.key_answer = date_ner['tokens']
                        current_result.type = 'ner_date_question'
                        deconstruction_result.append(current_result)     
        return deconstruction_result
    
    
    def _deconstruct_ner_loc(self, loc_ner: dict) -> List[QDeconstructionResult]:
        """Deconstructs a location named entity. Dict should have the following keys: 'label', 'tokens'"""
        
        deconstruction_result : List[QDeconstructionResult] = []
        loc_text = Helper.merge_tokens(loc_ner['tokens'])
        for srl in self.srls:
            if ('V' not in srl.keys()):
                continue
            found_subject_tokens = self.checkForAppropriateObjOrSub(srl, 0)
            found_object_tokens = self.checkForAppropriateObjOrSub(srl, 1)
            found_subject_text = Helper.merge_tokens(found_subject_tokens)
            found_object_text = Helper.merge_tokens(found_object_tokens)
            
            if (found_subject_text == '') or (found_subject_text == found_object_text): 
                continue
            
            if len(srl['V']) == 1:
                full_predicate = Helper.find_full_predicate(srl['V'][0])
            else:
                for pred_token in srl['V']:
                    if pred_token.tag_.startswith('VB'):
                        full_predicate = Helper.find_full_predicate(pred_token)
            full_predicate_text = Helper.merge_tokens(full_predicate)
            for k, v in srl.items():
                text_v = Helper.merge_tokens(v)
                if (loc_text in text_v) and (k != 'V') and (k != 'ARGM-LOC') and (text_v != found_subject_text):
                    real_object = []
                    if (found_object_text != '') and (text_v == found_object_text):
                        for l in range(len(self.doc)-1):
                            if (self.doc[l] not in v) or (self.doc[l+1] not in v):
                                continue
                            if (self.doc[l+1] in loc_ner['tokens']) and (self.doc[l].pos_ == 'ADP'):
                                break
                            real_object.append(self.doc[l])
                    else:
                        real_object = found_object_tokens
                    current_result = QDeconstructionResult()
                    current_result.predicate = full_predicate
                    if len(real_object) > 0 and real_object[-1].text == 'the':
                        real_object = real_object[:-1]
                    current_result.object = real_object
                    current_result.subject = found_subject_tokens
                    extra_tokens = []
                    if ('ARGM-TMP' in srl.keys()) and (Helper.merge_tokens(srl['ARGM-TMP']) not in full_predicate_text):
                        extra_tokens.extend(srl['ARGM-TMP'])
                    current_result.extra_field = extra_tokens
                    current_result.type = 'ner_loc_question'
                    current_result.key_answer = loc_ner['tokens']
                    deconstruction_result.append(current_result)
        return deconstruction_result
                    
                    
    def _deconstruct_ner_cardinal(self, cardinal_ner: dict) -> List[QDeconstructionResult]:
        """Deconstructs a cardinal number named entity. Dict should have the following keys: 'label', 'tokens'"""
        
        deconstruction_result : List[QDeconstructionResult] = []
        cardinal_text = Helper.merge_tokens(cardinal_ner['tokens'])
        for srl in self.srls:
            if ('V' not in srl.keys()):
                continue
            found_subject_tokens = self.checkForAppropriateObjOrSub(srl, 0)
            found_object_tokens = self.checkForAppropriateObjOrSub(srl, 1)
            found_subject_text = Helper.merge_tokens(found_subject_tokens)
            found_object_text = Helper.merge_tokens(found_object_tokens)
            
            if (found_subject_text == '') or (found_subject_text == found_object_text):
                continue
            if len(srl['V']) == 1:
                full_predicate = Helper.find_full_predicate(srl['V'][0])
            else:
                for pred_token in srl['V']:
                    if pred_token.tag_.startswith('VB'):
                        full_predicate = Helper.find_full_predicate(pred_token)
            full_predicate_text = Helper.merge_tokens(full_predicate)
            for k, v in srl.items():
                text_v = Helper.merge_tokens(v)
                if (cardinal_text in text_v) and (k != 'V'):
                    first_part = []
                    last_part = []
                    for tok in v:
                        if (tok in cardinal_ner['tokens']):
                            break
                        last_part.append(tok)
                    for tok in v:
                        if (tok in last_part):
                            continue
                        if (tok in cardinal_ner['tokens']):
                            continue
                        first_part.append(tok)
                    if len(last_part) > 0 and last_part[-1].text == 'the':
                        last_part = last_part[:-1]

                    current_result = QDeconstructionResult()
                    current_result.predicate = full_predicate
                    current_result.subject = first_part
                    current_result.extra_field = ''
                    current_result.key_answer = cardinal_ner['tokens']
                    current_result.type = 'ner_cardinal_question'
                    
                    extra_tokens = []
                    if ('ARGM-LOC' in srl.keys()) and (Helper.merge_tokens(srl['ARGM-LOC']) not in full_predicate_text):
                        extra_tokens.extend(srl['ARGM-LOC'])
                    if ('ARGM-TMP' in srl.keys()) and (Helper.merge_tokens(srl['ARGM-TMP']) not in full_predicate_text):
                        extra_tokens.extend(srl['ARGM-TMP']) 
                    
                    if (text_v == found_object_text) and (found_subject_text != ''):
                        # If the cardinal number is the object
                        current_result.object = extra_tokens
                        current_result.extra_field = found_subject_tokens
                    elif (text_v == found_subject_text) and (found_object_text != ''):
                        # If the cardinal number is the subject
                        current_result.object = last_part + found_object_tokens + extra_tokens

                    deconstruction_result.append(current_result)
        return deconstruction_result

    
    def _deconstruct_ner_person(self, person_ner: dict) -> List[QDeconstructionResult]:
        """Deconstructs a person named entity. Dict should have the following keys: 'label', 'tokens'"""
        
        deconstruction_result : List[QDeconstructionResult] = []
        person_text = Helper.merge_tokens(person_ner['tokens'])
        for srl in self.srls:
            if ('V' not in srl.keys()):
                continue
            
            found_subject_tokens = self.checkForAppropriateObjOrSub(srl, 0)
            found_object_tokens = self.checkForAppropriateObjOrSub(srl, 1)
            found_subject_text = Helper.merge_tokens(found_subject_tokens)
            found_object_text = Helper.merge_tokens(found_object_tokens)
            if (found_subject_text == '') or (found_subject_text == found_object_text): 
                continue
            if len(srl['V']) == 1:
                full_predicate = Helper.find_full_predicate(srl['V'][0])
            else:
                for pred_token in srl['V']:
                    if pred_token.tag_.startswith('VB'):
                        full_predicate = Helper.find_full_predicate(pred_token)
            full_predicate_text = Helper.merge_tokens(full_predicate)
            for k, v in srl.items():
                text_v = Helper.merge_tokens(v)
                if (person_text in text_v) and (k != 'V') and (text_v == found_subject_text) and (text_v != found_object_text):
                    is_in_relcl = any(Helper.is_in_relative_clause(token, relpronoun_exclude=["which"]) for token in person_ner['tokens'])
                    if (not is_in_relcl):
                        np_contains_person = []
                        for noun_phrase in self.noun_phrases:
                            contains = True
                            for tok in person_ner['tokens']:
                                if tok not in noun_phrase:
                                    contains = False
                            if contains:
                                np_contains_person = noun_phrase
                                break
                                
                        has_other_nouns = False
                        for tok in np_contains_person:
                            if tok not in person_ner['tokens']:
                                if tok.pos_ in ['NOUN', 'PROPN']:
                                    has_other_nouns = True
                                    break
                    extra_tokens = []
                    if ('ARGM-LOC' in srl.keys()) and (Helper.merge_tokens(srl['ARGM-LOC']) not in full_predicate_text):
                        extra_tokens.extend(srl['ARGM-LOC'])
                    if ('ARGM-TMP' in srl.keys()) and (Helper.merge_tokens(srl['ARGM-TMP']) not in full_predicate_text):
                        extra_tokens.extend(srl['ARGM-TMP']) 
                    if (not has_other_nouns):
                        deconstruction_result.append(
                            QDeconstructionResult(
                                predicate=full_predicate,
                                object=found_object_tokens,
                                subject=[],
                                extra_field=extra_tokens,
                                type='ner_person_question',
                                key_answer=person_ner['tokens']
                            )
                        )
        return deconstruction_result
    
    
    def _deconstruct_srl_direct() -> List[QDeconstructionResult]:
        """Deconstructs a direct SRL"""
        pass