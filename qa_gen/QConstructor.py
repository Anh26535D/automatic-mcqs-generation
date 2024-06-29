from typing import List, Tuple

from QDeconstructor import QDeconstructionResult
from helper import Helper
import utils
from spacy.tokens import Token
from tqdm import tqdm

class QConstructor:
    
    def __init__(self, doc, idx_srls, enhance_level: int = 0):
        self.doc = doc
        self.enhance_level = enhance_level
        
        self.clusters = []
        for cluster in self.doc._.coref_clusters:
            spans_in_cluster = []
            for span in cluster:
                tokens_in_span = []
                for token in self.doc:
                    if span[0] <= token.idx < span[1]:
                        tokens_in_span.append(token)
                spans_in_cluster.append(tokens_in_span)
            self.clusters.append(spans_in_cluster)  
            
        self.srls = []
        for srl in idx_srls:
            mod_srl = {}
            for k, v in srl.items():
                span = []
                for token in self.doc:
                    if v[0] <= token.idx < v[1]:
                        span.append(token)
                mod_srl[k] = span
            self.srls.append(mod_srl) 
            

    def _is_in_same_cluster(self, token1: Token, token2: Token) -> bool:
        """Checks if two tokens are in the same coreference cluster"""
        antecedent1, span1 = self._get_antecedent(token1)
        antecedent2, span2 = self._get_antecedent(token2)
        if Helper.merge_tokens(antecedent1) == Helper.merge_tokens(antecedent2):
            return True
        for tok1 in antecedent1:
            if tok1 in antecedent2:
                return True
        return False
        
        
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
        for cluster in self.clusters:
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
    
    def _resolve_antecedent(self, tokens: List[Token]) -> List[Token]:
        resolved_tokens = []
        previous_resolved_tokens = []
        for token in tokens:
            antecedent_tokens, _ = self._get_antecedent(token)
            # just replace if token is not in the antecedent itself
            if token in antecedent_tokens:
                resolved_tokens.append(token)
            else:
                is_resolved_before = False
                if token in previous_resolved_tokens:
                    is_resolved_before = True
                if not is_resolved_before:
                    resolved_tokens.extend(antecedent_tokens)
                    previous_resolved_tokens = [token for token in antecedent_tokens]
        return resolved_tokens
        
    
    def _enhance_subject(self, subject_input: List[Token]) -> List[str]:
        """
        Enhacing subject by adding the sentences with the same coreference cluster of the span of the subject.
        Enhance_level is the number of sentences to add.
        E.g.1 "Anna saved the cat. She is a hero." ->subject: She -> new subject: the one that saved the cat (enhance_level=1)
        E.g.2 "Anna saved the cat. She also saved the dog. She is a hero." ->subject: She -> new subject: the one that saved the cat and the dog (enhance_level=2)
        """          
        outputs = [Helper.merge_tokens(subject_input)]
        if self.enhance_level == 0:
            return outputs
        
        for tok in subject_input:
            found_cluster_idx = -1
            for idx, cluster in enumerate(self.clusters):
                for span in cluster:
                    if tok in span:
                        found_cluster_idx = idx
                        break
            if found_cluster_idx != -1:
                possible_span_indices = []
                for idx, span in enumerate(self.clusters[found_cluster_idx]):
                    if span[0].sent.start < tok.sent.start: # only consider the previous sentences
                        possible_span_indices.append(idx)
                if len(possible_span_indices) > 0:
                    sorted_possible_span_indices = sorted(possible_span_indices, key=lambda x: self.clusters[found_cluster_idx][x][0].sent.start)
                    possible_srl_indices = []
                    # for each span, loop through the srls to extract the srl sentence such that span has enrolled in
                    for idx in sorted_possible_span_indices:
                        for srl_idx, srl in enumerate(self.srls):
                            if 'V' not in self.srls[srl_idx]:
                                continue
                            subject_tokens = Helper.checkForAppropriateObjOrSub(srl, 0)
                            if len(subject_tokens) > 0 and any([(tok in subject_tokens) for tok in self.clusters[found_cluster_idx][idx]]):
                                if srl_idx not in possible_srl_indices:
                                    possible_srl_indices.append(srl_idx)
                    if len(possible_srl_indices) > 0:
                        srl_idx = possible_srl_indices[0]
                        sent_srl_tokens = []
                        subject_tokens = Helper.checkForAppropriateObjOrSub(self.srls[srl_idx], 0)
                        for k, v in self.srls[srl_idx].items():
                            is_token_in_subject = False
                            for tok_v in v:
                                if tok_v in subject_tokens:
                                    is_token_in_subject = True
                            if not is_token_in_subject:
                                simplified_v = Helper.simplify_dependencies(v)
                                sent_srl_tokens.extend(simplified_v)
                        sorted_sent_srl_tokens = sorted(sent_srl_tokens, key=lambda x: x.idx)
                        srl_text = 'the one that ' + Helper.merge_tokens(sorted_sent_srl_tokens)
                        if self.enhance_level > 1:
                            max_srls = min(self.enhance_level, len(possible_srl_indices) - 1)
                            for srl_idx in possible_srl_indices[:max_srls]:
                                sent_srl_tokens = []
                                subject_tokens = Helper.checkForAppropriateObjOrSub(self.srls[srl_idx], 0)
                                for k, v in self.srls[srl_idx].items():
                                    is_token_in_subject = False
                                    for tok_v in v:
                                        if tok_v in subject_tokens:
                                            is_token_in_subject = True
                                            break
                                    if not is_token_in_subject:
                                        simplified_v = Helper.simplify_dependencies(v)
                                        sent_srl_tokens.extend(simplified_v)
                                sorted_sent_srl_tokens = sorted(sent_srl_tokens, key=lambda x: x.idx)
                                srl_text = srl_text + ' and ' + Helper.merge_tokens(sorted_sent_srl_tokens)
                        outputs.append(srl_text)
        return outputs
    
    
    def _enhance_answer_in_direct_question(self, subject_input: List[Token], answer_input: str) -> List[str]:
        """
        Enhacing answer by adding the sentences with the same coreference cluster of the span of the subject.
        Enhance_level is the number of sentences to add.
        E.g.1 "Anna saved the cat. She is a hero." -> ("Did Anna save the cat?", "Yes") -> ("Did Anna save the cat?", "Yes, and she is a hero")
        E.g.2 "Anna saved the cat. She also saved the dog. She is a hero." -> ("Did Anna save the cat?", "Yes") -> ("Did Anna save the cat?", "Yes, and she also saved the dog and is a hero")
        """
        if self.enhance_level == 0:
            return [answer_input]
        outputs = []
        antecedent_of_subject_input = []
        for tok in subject_input:
            antecedent_of_subject_input, _ = self._get_antecedent(tok)
            if antecedent_of_subject_input[0] != tok:
                break
        if len(antecedent_of_subject_input) == 0:
            antecedent_of_subject_input = subject_input
        for tok in subject_input:
            found_cluster_idx = -1
            for idx, cluster in enumerate(self.clusters):
                for span in cluster:
                    if tok in span:
                        found_cluster_idx = idx
                        break
            if found_cluster_idx != -1:
                possible_span_indices = []
                for idx, span in enumerate(self.clusters[found_cluster_idx]):
                    if span[0].sent.start < tok.sent.start: # only consider the previous sentences
                        possible_span_indices.append(idx)
                if len(possible_span_indices) > 0:
                    sorted_possible_span_indices = sorted(possible_span_indices, key=lambda x: self.clusters[found_cluster_idx][x][0].sent.start)
                    possible_srl_indices = []
                    # for each span, loop through the srls to extract the srl sentence such that span has enrolled in
                    for idx in sorted_possible_span_indices:
                        for srl_idx, srl in enumerate(self.srls):
                            if 'V' not in self.srls[srl_idx]:
                                continue
                            subject_tokens = Helper.checkForAppropriateObjOrSub(srl, 0)
                            if len(subject_tokens) > 0 and any([(tok in subject_tokens) for tok in self.clusters[found_cluster_idx][idx]]):
                                if srl_idx not in possible_srl_indices:
                                    possible_srl_indices.append(srl_idx)
                    # Get unique srls in different sentences
                    srl_indices = possible_srl_indices[::-1]
                            
                    if len(srl_indices) > 0:
                        srl_text = answer_input + ', '
                        is_first_clause = True
                        for srl_idx in srl_indices[:min(self.enhance_level, len(srl_indices))]:
                            sent_srl_tokens = []
                            subject_tokens = Helper.checkForAppropriateObjOrSub(self.srls[srl_idx], 0)
                            for k, v in self.srls[srl_idx].items():
                                is_token_in_subject = False
                                for tok_v in v:
                                    if tok_v in subject_tokens:
                                        is_token_in_subject = True
                                        break
                                if not is_token_in_subject:
                                    simplified_v = Helper.simplify_dependencies(v)
                                    sent_srl_tokens.extend(simplified_v)
                            sorted_sent_srl_tokens = sorted(sent_srl_tokens, key=lambda x: x.idx)
                            if is_first_clause:
                                srl_text = srl_text + 'and ' + Helper.merge_tokens(antecedent_of_subject_input) + ' ' + Helper.merge_tokens(sorted_sent_srl_tokens)
                                is_first_clause = False
                            else:
                                srl_text = srl_text + ' and ' + Helper.merge_tokens(sorted_sent_srl_tokens)
                        outputs.append(srl_text)
        return outputs
    
    
    def _enhance_direct_question_by_multiple_srls(self, subject: List[Token]) -> List[Tuple[str, str]]:
        """
        Enhancing direct question by adding multiple SRLs.
        
        Args:
        -----
        subject: List[Token]
        
        Returns:
        --------
        List[Tuple[str, str]]
            List of tuples of new extra field and answer
        """
        outputs = []
        for srl in self.srls:
            if ('V' not in srl.keys()):
                continue
            # Check for different sentences
            is_diff_sent = False
            for tok1 in subject:
                for tok2 in srl['V']:
                    if tok1.sent.start != tok2.sent.start:
                        is_diff_sent = True
                        break
            if not is_diff_sent:
                continue
            
            found_subject_tokens = Helper.checkForAppropriateObjOrSub(srl, 0)
            found_object_tokens = Helper.checkForAppropriateObjOrSub(srl, 1)
            found_subject_text = Helper.merge_tokens(found_subject_tokens)
            found_object_text = Helper.merge_tokens(found_object_tokens)
            
            if (found_subject_text == '') or (found_object_text == '') or (found_subject_text == found_object_text):
                continue
            
            is_same_cluster = False
            for tok1 in subject:
                for tok2 in found_subject_tokens:
                    if self._is_in_same_cluster(tok1, tok2):
                        is_same_cluster = True
                        break

            # if the two subject are in cluster, yield "yes" question
            if is_same_cluster:
                extra_field = ''
                answer = 'yes'
                # Add sentence 2 to the extra_field
                sent_srl_tokens2 = []
                for k, v in srl.items():
                    is_token_in_subject = False
                    for tok_v in v:
                        if tok_v in found_subject_tokens:
                            is_token_in_subject = True
                    if not is_token_in_subject:
                        simplified_v = Helper.simplify_dependencies(v)
                        sent_srl_tokens2.extend(simplified_v)
                sent_srl_tokens2 = sorted(sent_srl_tokens2, key=lambda x: x.idx)
                extra_field = Helper.merge_tokens(sent_srl_tokens2)
                extra_field = 'and ' + extra_field
                outputs.append((extra_field, answer))
            # Check if the two subject are not in cluster, yield "no" question and correct the answer
            else:
                extra_field = ''
                answer = 'no'
                # Add sentence 2 to the extra_field
                sent_srl_tokens2 = []
                for k, v in srl.items():
                    if k in ['ARGM-MNR']:
                        continue
                    is_token_in_subject = False
                    for tok_v in v:
                        if tok_v in found_subject_tokens:
                            is_token_in_subject = True
                    if not is_token_in_subject:
                        simplified_v = Helper.simplify_dependencies(v)
                        sent_srl_tokens2.extend(simplified_v)
                sent_srl_tokens2 = sorted(sent_srl_tokens2, key=lambda x: x.idx)
                extra_field = Helper.merge_tokens(sent_srl_tokens2)
                resolved_extra_field = self._resolve_antecedent(sent_srl_tokens2)
                resolved_original_subject = self._resolve_antecedent(subject.copy())
                original_subject = Helper.merge_tokens(resolved_original_subject)
                resolve_new_subject = self._resolve_antecedent(found_subject_tokens.copy())
                resolved_extra_field = Helper.merge_tokens(resolved_extra_field)
                answer = f"No, it is the {Helper.merge_tokens(resolve_new_subject)} that {resolved_extra_field}."
                extra_field = ', and ' + resolved_extra_field
                outputs.append((extra_field, answer))
        return outputs
    
    def constructQuestion(
            self, 
            deconstruction_results: List[QDeconstructionResult], 
            limit: int = 100,
            selection_method: str = 'random',
            type_name: str = None,
            verbose: bool = False,):
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
                    elif predArr[0].text in ['am', 'is', 'are', 'was', 'were']:
                        aux_text = predArr[0].text
                        predArr.pop(firstFoundVerbIndex)
                        predicate_strs = Helper.merge_tokens(predArr).split()
                if numOfVerbs == 0 and len(predArr) == 1:
                    mainVerb = predArr[0].text
                    if utils.lemmatizeVerb(predArr[0].text) == predArr[0].text:
                        aux_text = 'do'
                    else:
                        aux_text = 'does'
                    predicate_strs = [mainVerb]
                if numOfVerbs > 1: # More than 1 verb (e.g. 'He is going to school', 'He will be going to school')
                    word = predArr[firstFoundVerbIndex].text
                    if word in ['am', 'is', 'are', 'was', 'were', 'has', 'have', 'had', 'will']:
                        aux_text = word
                        predArr.pop(firstFoundVerbIndex)
                        if not isinstance(predArr, list):
                            predArr = list(predArr)
                        predicate_strs = Helper.merge_tokens(predArr).split()
            else: # having_word_and (e.g. 'He finds and eats the apple'), TODO: handle more complex cases
                pass
                
            isQuestionMarkExist = False
            verbRemainingPart = Helper.merge_strs(predicate_strs)
            question = ''
            type_text = deconstruction_result.type
            # Replace by the antecedent of answer
            resolved_answer = self._resolve_antecedent(deconstruction_result.key_answer.copy())
            answer = Helper.merge_tokens(resolved_answer)
            # Replace by the antecedent of object
            resolved_object = self._resolve_antecedent(deconstruction_result.object.copy())
            object_text = Helper.merge_tokens(resolved_object)
            
            # Replace by the antecedent of extra field
            resolved_extra = self._resolve_antecedent(deconstruction_result.extra_field.copy())
            extra_text = Helper.merge_tokens(resolved_extra)
            if type_text != 'direct':
                for subject_text in self._enhance_subject(deconstruction_result.subject):
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
                    elif type_text == 'acomp_question':
                        isQuestionMarkExist = True  
                        question = 'Indicate characteristics of ' + utils.getObjectPronun(subject_text)
                    elif type_text == 'nsubj_question':
                        question_word = 'What '
                        for ent in self.doc.ents:
                            if (ent.text == answer and ent.label_ == 'PERSON'):
                                question_word = 'Who '
                        question = question_word + predicate_text + ' ' + object_text + ' ' + extra_text
                    elif type_text == 'srl_causal':
                        question = 'Why '+ aux_text + ' ' + negativePart + ' ' + subject_text + ' ' + verbRemainingPart + ' ' + object_text  + ' ' + extra_text
                    elif type_text == 'srl_purpose':
                        question = 'What '+ aux_text + negativePart + ' the purpose of that ' + subject_text + ' ' + verbRemainingPart + ' ' + object_text  + ' ' + extra_text
                    elif type_text == 'srl_manner':
                        question = 'How '+ aux_text + ' ' + negativePart + ' ' + subject_text + ' ' + verbRemainingPart + ' ' + object_text  + ' ' + extra_text
                    elif type_text == 'srl_temporal':
                        # defind question word based on the answer (three types: when, how long, how often)
                        question_word = 'When '
                        if 'for' in answer or 'since' in answer:
                            question_word = 'How long '
                        elif 'every' in answer or 'each' in answer:
                            question_word = 'How often '
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
            else: # type_text == 'direct':
                # Enhance the question by adding multiple SRLs
                extra_fields = self._enhance_direct_question_by_multiple_srls(deconstruction_result.subject)
                for extra_field, answer in extra_fields:
                    aux_text = aux_text[:1].upper() + aux_text[1:]
                    if object_text.endswith('.'):
                        object_text = object_text[:-1]
                    resolved_subject = self._resolve_antecedent(deconstruction_result.subject.copy())
                    subject_text = Helper.merge_tokens(resolved_subject)
                    question = aux_text + ' ' + subject_text + ' ' + verbRemainingPart + ' ' + object_text  + ' ' + extra_field
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
                            'type': 'direct_with_multiple_srls',
                        })
                aux_text= aux_text[:1].upper() + aux_text[1:]
                if object_text.endswith('.'):
                    object_text = object_text[:-1]
                resolved_subject = self._resolve_antecedent(deconstruction_result.subject.copy())
                subject_text = Helper.merge_tokens(resolved_subject)
                question = aux_text + ' ' + subject_text + ' ' + verbRemainingPart + ' ' + object_text  + ' ' + extra_text
                answer = 'Yes' if not negativePart else 'No'
                answers = self._enhance_answer_in_direct_question(deconstruction_result.subject, answer)
                words = question.split()
                formattedQuestion = ''
                for word_idx, word in enumerate(words):
                    if word in ['He', 'She', 'It', 'They', 'We', 'In'] and word_idx != 0:
                        word = word.lower()
                    if word in ['.', ',', '?', '!', ':', ';'] or word == "'s":
                        formattedQuestion = formattedQuestion + word
                    else:
                        formattedQuestion = formattedQuestion + ' ' + word
                if answers:
                    for ans in answers:
                        if formattedQuestion != '':
                            if isQuestionMarkExist == False:
                                formattedQuestion = formattedQuestion + '?'
                            else: 
                                formattedQuestion = formattedQuestion + '.'
                            if verbose:
                                print('\tGenerate final question: ', formattedQuestion)
                            qa_pairs.append({
                                'question': formattedQuestion,
                                'answer': ans,
                                'type': type_text,
                            })
                else:
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

        unique_qa_pairs.sort(key = lambda s: len(s['question']), reverse=True)
        if selection_method == 'random':
            import random
            random.shuffle(unique_qa_pairs)
        elif selection_method == 'only_type':
            if 'type_name' not in locals():
                raise ValueError('Type name is not specified')
            unique_qa_pairs = [qa_pair for qa_pair in unique_qa_pairs if qa_pair['type'].startswith(type_name)]
        elif selection_method == 'longest':
            unique_qa_pairs.sort(key = lambda s: len(s['question']), reverse=True)
        elif selection_method == 'shortest':
            unique_qa_pairs.sort(key = lambda s: len(s['question']), reverse=False)
        elif selection_method == 'alphabetical':
            unique_qa_pairs.sort(key = lambda s: s['question'])
        elif selection_method == 'reverse_alphabetical':
            unique_qa_pairs.sort(key = lambda s: s['question'], reverse=True)
        elif selection_method == 'answer_length':
            unique_qa_pairs.sort(key = lambda s: len(s['answer']))
        elif selection_method == 'reverse_answer_length':
            unique_qa_pairs.sort(key = lambda s: len(s['answer']), reverse=True)
        else:
            raise ValueError('Invalid selection method')
        
        if limit > 0:
            unique_qa_pairs = unique_qa_pairs[:limit]

        result = []
        for qa_pair in tqdm(unique_qa_pairs, total=len(unique_qa_pairs), desc='Paraphrasing QA pairs'):
            result.append({
                'question': qa_pair['question'],
                'answer': qa_pair['answer'],
                'type': qa_pair['type'],
            })        
        return result