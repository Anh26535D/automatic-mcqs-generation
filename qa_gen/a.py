# import os
# import json
# import re

# import warnings
# warnings.filterwarnings("ignore")

# import spacy
# from fastcoref import spacy_component

# from helper import Helper

# from typing import List, Tuple
# from spacy.tokens import Token


# cur_dir = os.getcwd()

# CONTRACTIONS_PATH = os.path.join(cur_dir, 'utility_files', 'contractions.json')

# contractions_dict = json.loads(open(CONTRACTIONS_PATH).read())
# contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

# nlp = spacy.load('en_core_web_sm')
# nlp.add_pipe("fastcoref")

# text = '''
#     The Yanomami live along the rivers of the rainforest in the north of Brazil. 
#     They have lived in the rainforest for about 10,000 years and they use more than 2,000 different plants for food and for medicine. 
#     But in 1988, someone found gold in their forest, and suddenly 45,000 people came to the forest and began looking for gold. 
#     They cut down the forest to make roads. 
#     They made more than a hundred airports. 
#     The Yanomami people lost land and food. 
#     Many died because new diseases came to the forest with the strangers.
#     In 1987, they closed fifteen roads for eight months. 
#     No one cut down any trees during that time. In Panama, the Kuna people saved their forest. 
#     They made a forest park which tourists pay to visit. 
#     The Gavioes people of Brazil use the forest, but they protect it as well. 
#     They find the Brazil nuts which grow on the forest trees.
# '''

# def expandContractions(s, contractions_dict=contractions_dict):
#     def replace(match):
#         return contractions_dict[match.group(0)]
#     return contractions_re.sub(replace, s)


# def clean_text(text: str):
#     text = text.replace('\n', '')
#     text = text.replace('\t', '')
#     text = text.replace('\r', '')
#     text = text.replace('“', '"')
#     text = text.replace('”', '"')
#     text = text.replace("’", "'")
#     text = text.replace("‘", "'")
#     text = text.strip()
#     return text

# text = clean_text(text)
# text = expandContractions(text)
# textList = []
# textList.append(text)

# doc = nlp(u''+text, component_cfg={"fastcoref": {'resolve_text': True}})

# clusters = []
# for cluster in doc._.coref_clusters:
#     spans_in_cluster = []
#     for span in cluster:
#         tokens_in_span = []
#         for token in doc:
#             if span[0] <= token.idx < span[1]:
#                 tokens_in_span.append(token)
#         spans_in_cluster.append(tokens_in_span)
#     clusters.append(spans_in_cluster)  
    
    
# def _get_antecedent(token: Token) -> Tuple[List[Token], List[Token]]:
#     for cluster in clusters:
#         for span in cluster:
#             if token in span:
#                 return cluster[0], span
#     return [token], [token]


# def _resolve_coref(tokens: List[Token]) -> List[Token]:
#     resolved_tokens = []
#     i = 0
#     prev_token = None
#     while i < len(tokens):
#         antecedent, span = _get_antecedent(tokens[i])
#         if prev_token not in span:
#             resolved_tokens.extend(antecedent)
#             prev_token = tokens[i]
#             i = i + len(span)
#         else:
#             resolved_tokens.append(tokens[i])
#             prev_token = tokens[i]
#             i = i + 1
        
#     return resolved_tokens
        
# def _is_in_same_cluster(token1: Token, token2: Token) -> bool:
#     """Checks if two tokens are in the same coreference cluster"""
#     antecedent1, span1 = _get_antecedent(token1)
#     antecedent2, span2 = _get_antecedent(token2)
#     if Helper.merge_tokens(antecedent1) == Helper.merge_tokens(antecedent2):
#         return True
#     for tok1 in antecedent1:
#         if tok1 in antecedent2:
#             return True
#     return False

# resolved_text = _resolve_coref([token for token in doc])
# resolved_text = Helper.merge_tokens(resolved_text)
# print(resolved_text)
