from typing import List
from collections import Counter

import spacy
from spacy import displacy
import spacy.tokenizer
import spacy.tokens


"""
Notation of dependencies:
-------------------------
    nsubj: nominal subject
    aux: auxiliary
    auxpass: auxiliary (passive)
    neg: negation
    prt: particle
    ccomp: clausal complement
    xcomp: open clausal complement
    acomp: adjectival complement
    csubj: clausal subject
    csubjpass: clausal subject (passive)
    advcl: adverbial clause modifier
    det: determiner
    amod: adjectival modifier
    nummod: numeric modifier
    poss: possession modifier
    prep: prepositional modifier
    pobj: object of preposition
    acl: clausal modifier of noun
    advmod: adverbial modifier
    cc: coordinating conjunction
    conj: conjunct
    appos: appositional modifier
    compound: compound modifier
    mark: marker
    punct: punctuation
    parataxis: parataxis
    dep: dependent
    ROOT: root
    nmod: nominal modifier
    acl: clausal modifier of noun
"""


class Helper:
    
    def visualize_dependencies(text: str):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        displacy.serve(doc, style="dep")
        
    
    def merge_tokens(tokens: List[spacy.tokens.Token]) -> str:
        merged_text = ""
        for token in tokens:
            if token.text in [".", ",", "!", "?", ":", ";"]:
                merged_text += token.text
            elif token.text == "'s":
                merged_text += token.text
            else:
                merged_text += " " + token.text
        return merged_text.strip()
    
    
    def merge_strs(tokens: List[str]) -> str:
        merged_text = ""
        for token in tokens:
            if token in [".", ",", "!", "?", ":", ";"]:
                merged_text += token
            elif token == "'s":
                merged_text += token
            else:
                merged_text += " " + token
        return merged_text.strip()
    
    
    def checkForAppropriateObjOrSub(srl, occur_no):
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
        List[Token]
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
        return []
    
    
    def find_subject_of_predicate(predicate: spacy.tokens.Token, use_ccomp: bool = False) -> List[spacy.tokens.Token]:
        """
        Find the subject of the predicate
        
        Args:
        -----
        predicate: spacy.tokens.Token
            Predicate token
        use_ccomp: bool
            Use clausal complement (ccomp) to find the subject
            
        Returns:
        --------
        List[spacy.tokens.Token]
            The subject token
        """
        
        def get_subject_from_verb(verb):
            # Look for nominal subject (nsubj), clausal subject (csubj), or passive nominal subject (nsubjpass)
            for child in verb.children:
                if child.dep_ == "nsubj" or child.dep_ == "nsubjpass":
                    return Helper.find_full_subject(child)
                elif child.dep_ == "csubj":
                    return Helper.find_full_subject(child, rel_deps=["mark", "adv", "advmod"])
            return []

        subject = get_subject_from_verb(predicate)
        if subject:
            return subject
        
        # If no subject is found, check if the predicate is part of a conjunction
        if predicate.dep_ == "conj":
            # Check the parent verb for the subject
            parent_verb = predicate.head
            subjects = get_subject_from_verb(parent_verb)
            if subjects:
                return subjects
            
            # Check other verbs in the conjunction chain
            for sibling in parent_verb.children:
                if sibling.dep_ == "conj":
                    subjects = get_subject_from_verb(sibling)
                    if subjects:
                        return subjects
        
        # Check for clausal complements
        if use_ccomp:
            for child in predicate.children:
                if child.dep_ == "ccomp":
                    subjects = get_subject_from_verb(child)
                    if subjects:
                        return subjects
        return []   
    
    
    def simplify_dependencies(original_tokens: List[spacy.tokens.Token]) -> List[spacy.tokens.Token]:
        """
        Simplify dependencies by removing the relative clause if it is not in relative clause itself
        """
        output_dependencies = [tok for tok in original_tokens]
        def remove_all_children(token):
            for child in token.children:
                remove_all_children(child)
                if child in output_dependencies:
                    output_dependencies.remove(child)
                
        for token in original_tokens:
            if token.dep_ == "relcl" and token.head in original_tokens:
                remove_all_children(token)
                output_dependencies.remove(token)
        return output_dependencies
    
    
    def find_full_predicate(
        token: spacy.tokens.Token, 
        rel_deps: List[str] = None,
        include_deps: List[str] = None, 
        exclude_deps: List[str] = None) -> List[spacy.tokens.Token]:
        """
        Find full form of the predicate
        Only consider conj verb if the root token has not direct object (dobj).
        Because in usual case, the direct object is connected to the last conj verb, if not, these verbs are in different clauses.
        
        Args:
        -----
        token: spacy.tokens.Token
            Token to find the full predicate
        rel_deps: List[str]
            Relevant dependencies to consider (substitute the default dependencies)
        include_deps: List[str]
            Include dependencies (append to relevant dependencies)
        exclude_deps: List[str]
            Exclude dependencies (remove from relevant dependencies)
            
        Returns:
        --------
        List[spacy.tokens.Token]    
        """
        predicate_tokens = []
        relevant_deps = {"aux", "auxpass", "neg", "prt"}
        if rel_deps is not None:
            relevant_deps = set(rel_deps)
        if include_deps is not None:
            relevant_deps.update(include_deps)
        if exclude_deps is not None:
            relevant_deps = relevant_deps - set(exclude_deps)
                
        # check if the root token have direct object
        has_dobj = False
        for child in token.children:
            if child.dep_ == "dobj":
                has_dobj = True
                break
        
        if has_dobj:
            if "cc" in relevant_deps and "conj" in relevant_deps:
                relevant_deps.remove("cc")
                relevant_deps.remove("conj")
                
        def add_related_tokens(token):
            predicate_tokens.append(token)
            for child in token.children:
                if child.dep_ in relevant_deps:
                    add_related_tokens(child)

        add_related_tokens(token)
        
        predicate_tokens = sorted(predicate_tokens, key=lambda token: token.i)
        return predicate_tokens
    
    
    def find_full_subject(subj_word: spacy.tokens.Token, rel_deps: List[str] = None, is_append: bool = False) -> List[spacy.tokens.Token]:
        """
        Find full form of the subject
        
        Args:
        -----
        subj_word: spacy.tokens.Token
            Subject word
        rel_deps: List[str]
            Relevant dependencies to consider
        is_append: bool
            Append the relevant dependencies
        
        Returns:
        --------
        List[spacy.tokens.Token]
        """
        subj_tokens = []
        relevant_deps = {"det", "amod", "compound", "nummod", "poss", "cc", "conj", "case"}
        if rel_deps is not None:
            if is_append:
                relevant_deps.update(rel_deps)
            else:
                relevant_deps = rel_deps
                
        def add_related_tokens(token):
            subj_tokens.append(token)
            for child in token.children:
                if child.dep_ in relevant_deps:
                    add_related_tokens(child)

        add_related_tokens(subj_word)

        sorted_tokens = sorted(subj_tokens, key=lambda token: token.i)
        return sorted_tokens
    
    
    def find_full_direct_object(
        dobj_word: spacy.tokens.Token, 
        rel_deps: List[str] = None, 
        include_deps: List[str] = None,
        exclude_deps: List[str] = None,
        use_acl: bool = False,
        ) -> List[spacy.tokens.Token]:
        """
        Find full form of the direct object
        
        Args:
        -----
        dobj_word: spacy.tokens.Token
            Direct object word
        rel_deps: List[str]
            Relevant dependencies to consider (substitute the default dependencies)
        include_deps: List[str]
            Include dependencies (append to relevant dependencies)
        exclude_deps: List[str]
            Exclude dependencies (remove from relevant dependencies)
        use_acl: bool
            Use clausal modifier of noun (acl) to find the direct object
        Returns:
        --------
        List[spacy.tokenizer.Token]
        """        
        dobj_tokens = []
        relevant_deps = {"det", "amod", "compound", "nummod", "poss", "prep", "advmod", "cc", "conj"}
        
        if rel_deps is not None:
            relevant_deps = set(rel_deps)
        if include_deps is not None:
            relevant_deps.update(include_deps)
        if exclude_deps is not None:
            relevant_deps = relevant_deps - set(exclude_deps)
        
        def add_related_tokens(token):
            if token not in dobj_tokens:
                dobj_tokens.append(token)
                for child in token.children:
                    if token.dep_ == "poss" and child.dep_ == "case":
                        dobj_tokens.append(child)
                    elif child.dep_ == "prep" and token == dobj_word:
                        continue
                    elif child.dep_ in relevant_deps:
                        add_related_tokens(child)
        
        add_related_tokens(dobj_word)

        sorted_tokens = sorted(dobj_tokens, key=lambda token: token.i)
        return sorted_tokens
    
    
    def find_full_attribute(
        attribute_word: spacy.tokens.Token, 
        rel_deps: List[str] = None, 
        include_deps: List[str] = None,
        exclude_deps: List[str] = None,
        use_acl: bool = False,
        use_relcl: bool = False,
        ) -> List[spacy.tokens.Token]:
        """
        Find full form of the attribute
        
        Args:
        -----
        attribute_word: spacy.tokens.Token
            Attribute word
        rel_deps: List[str]
            Relevant dependencies to consider (substitute the default dependencies)
        include_deps: List[str]
            Include dependencies (append to relevant dependencies)
        exclude_deps: List[str]
            Exclude dependencies (remove from relevant dependencies)
        use_acl: bool
            Use clausal modifier of noun (acl) to find the attribute
        use_relcl: bool
            Use relative clause (relcl) to find the attribute
        Returns:
        --------
        List[spacy.tokenizer.Token]
        """ 
        attr_tokens = []
        relevant_deps = {"det", "amod", "compound", "nummod", "poss", "prep", "advmod", "cc", "conj"}
        
        if rel_deps is not None:
            relevant_deps = set(rel_deps)
        if include_deps is not None:
            relevant_deps.update(include_deps)
        if exclude_deps is not None:
            relevant_deps = relevant_deps - set(exclude_deps)
        
        if use_acl:
            relevant_deps.add("acl")
        if use_relcl:
            relevant_deps.add("relcl")
        
        def add_related_tokens(token):
            if token not in attr_tokens:
                attr_tokens.append(token)
                for child in token.children:
                    if token.dep_ == "poss" and child.dep_ == "case":
                        attr_tokens.append(child)
                    elif child.dep_ == "prep" and token == attribute_word:
                        continue
                    elif child.dep_ in relevant_deps:
                        add_related_tokens(child)
                        
        add_related_tokens(attribute_word)
        
        sorted_tokens = sorted(attr_tokens, key=lambda token: token.i)
        return sorted_tokens
        
    
    def find_full_prep(prep_word: spacy.tokens.Token, rel_deps: List[str] = None, is_append: bool = None) -> str:
        """
        Find full form of the preposition
        
        Args:
        -----
        prep_word: spacy.tokens.Token
            Preposition word
        rel_deps: List[str]
            Relevant dependencies to consider
        is_append: bool
            Append the relevant dependencies
            
        Returns:
        --------
        str
        
        """
        prep_tokens = []
        relevant_deps = {"pobj", "prep", "advmod", "amod", "det"}
        if rel_deps is not None:
            if is_append:
                relevant_deps.update(rel_deps)
            else:
                relevant_deps = rel_deps
        
        def add_related_tokens(token):
            prep_tokens.append(token)
            for child in token.children:
                if child.dep_ in relevant_deps:
                    add_related_tokens(child)

        add_related_tokens(prep_word)

        sorted_tokens = sorted(prep_tokens, key=lambda token: token.i)
        return " ".join([token.text for token in sorted_tokens])
    
    
    def is_in_relative_clause(token: spacy.tokens.Token, relpronoun_exclude: List[str] = None) -> bool:
        """Determine if the token is part of a relative clause."""
        # Relative clause indicators
        relative_clause_deps = {'relcl', 'acl:relcl'}
        relative_pronouns = {'who', 'whom', 'whose', 'which', 'that'}
        if relpronoun_exclude is not None:
            relative_pronouns = set(relative_pronouns) - set(relpronoun_exclude)
        # Check if the token itself is part of a relative clause
        if token.dep_ in relative_clause_deps:
            return True

        # Check if token is a relative pronoun
        if token.text.lower() in relative_pronouns:
            return True

        # Check ancestors for relative clause relation
        for ancestor in token.ancestors:
            if ancestor.dep_ in relative_clause_deps:
                return True

        return False