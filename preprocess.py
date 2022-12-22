'''
 @Date  : 12/02/2019
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''

"""
Reads the raw csv files downloaded from the Propara website, then create JSON files which contain
lists of instances in train, dev and test

JSON format: a list of instances
Instance
    |____topic
    |____prompt
    |____paragraph id
    |____paragraph (string)
    |____entity (string)
    |____gold state change sequence (list of labels)
    |____number of words (tokens) in the paragraph
    |____number of sentences in the paragraph
    |____number of location candidates
    |____list of location candidates
    |____gold locations (list of strings, len = sent + 1)
    |____list of sentences
               |____sentence id
               |____number of words
               |____sentence (string)
               |____entity mention position (list of indices)
               |____verb mention position (list of indices)
               |____list of location candidates
                            |____location mention position (list of indices)
                                (list length equal to number of location candidates)
"""

from copyreg import pickle
from typing import Dict, List, Tuple, Set
import pandas as pd
import argparse
import json
import os
import time
import re
pd.set_option('display.max_columns', 50)
total_paras = 0  # should equal to 488 after read_paragraph

import spacy
nlp = spacy.load("en_core_web_sm", disable = ['ner'])
# nlp2 = spacy.load("en_core_web_lg", disable = ['ner'])

# from flair.data import Sentence
# from flair.models import SequenceTagger
# import flair
# pos_tagger = SequenceTagger.load('pos')


def tokenize(paragraph: str):
    """
    Change the paragraph to lower case and tokenize it!
    """
    paragraph = re.sub(' +', ' ', paragraph)  # remove redundant spaces in some sentences.
    para_doc = nlp(paragraph.lower())  # create a SpaCy Doc instance for paragraph
    tokens_list = [token.text for token in para_doc]
    return ' '.join(tokens_list), len(tokens_list)


def lemmatize(paragraph: str):
    """
    Reads a paragraph/sentence/phrase/word and lemmatize it!
    """
    if paragraph == '-' or paragraph == '?':
        return None, paragraph
    para_doc = nlp(paragraph)
    lemma_list = [token.lemma_ if token.lemma_ != '-PRON-' else token.text for token in para_doc]
    return lemma_list, ' '.join(lemma_list)


def coreArguments(doc: spacy.tokens.doc.Doc) -> List[str]:
  # function added by GEE
  core_args = []
  for token in doc:  
    if token.head.pos_ in ['VERB', 'AUX'] and token.pos_ in ['NOUN', 'PROPN']:      
      core_args += [token.text]
      if token.lemma_ != token.text:
        core_args += [token.lemma_]
      left_dependents = [x for x in token.lefts]
      if left_dependents and left_dependents[0].pos_ == 'DET':
        left_dependents = left_dependents[1:]
      right_dependents = [x for x in token.rights]
      # take care of "of" on the right
      if "of" in [x.text for x in right_dependents]:
        of_token = [x for x in token.rights if x.text == "of"][0]
        if [y for y in of_token.rights]:
          right_edge = [y for y in of_token.rights][-1].i
          of_idx = of_token.i
          right_dependents += [y for y in doc[of_idx+1:right_edge+1]]
        
      if right_dependents and right_dependents[-1].dep_ == 'prep':
        right_dependents = right_dependents[:-1]
      span = [x for x in left_dependents] + [token] + [x for x in right_dependents]     
      np = [x.text for x in span]
      np3 = [x.lemma_ for x in span]
      core_args += [' '.join(np)]
      core_args += [' '.join(np3)]
      # print('coreArguments 1:', ' '.join(np))
      if span[-1].lemma_ != span[-1].text:
        np2 = [x.text for x in span[:-1]] + [span[-1].lemma_]
        core_args += [' '.join(np2)]        
        # print('coreArguments 2:', ' '.join(np2))
  return list(set(core_args))


def locativeADV(doc: spacy.tokens.doc.Doc) -> List[str]: 
  loc = []
  for token in doc:
    if token.head.pos_ in ['VERB', 'AUX'] and token.head.lemma_ in ['be', 'go'] and token.dep_ in ['xcomp', 'acomp', 'advmod'] and token.pos_ == 'ADV':
      loc += [token.text]
      loc += [token.lemma_]
      left_dependents = [x.text for x in token.lefts]
      lefter_dependents = [y.text for x in token.lefts for y in x.lefts]
      right_dependents = [x.text for x in token.rights]
      righter_dependents = [y.text for x in token.rights for y in x.rights]      
      loc += [' '.join(lefter_dependents + left_dependents + [token.text] + right_dependents + righter_dependents)]      
  return list(set(loc))
  
def taggingMistake(doc: spacy.tokens.doc.Doc) -> List[str]:
  ent = []
  for token in doc:
    if token.pos_ in ['NOUN', 'PROPN'] and token.dep_ == 'ROOT':
      ent += [token.lemma_]    
    # ('turbine', 13, 'PROPN', 'NNP', 'ROOT', 'turbine', 13),
  return list(set(ent))  
  
def extractSubtree(doc: spacy.tokens.doc.Doc, idx: int) -> str:
  prep_token_object = doc[idx:idx+1]
  # prep_id = idx  
  # np_beginning = prep_id + 1
  subtree = []
  if list(prep_token_object.rights):
    pobj = list(prep_token_object.rights)[-1]
    subtree += [prep_token_object.text] + [descendant.text for descendant in pobj.subtree]
  else:
    subtree += [prep_token_object.lemma_]
  return ' '.join(subtree)

def purifytree(tree: List[spacy.tokens.token.Token]) -> List[spacy.tokens.token.Token]:
  if tree and tree[0].pos_ == 'DET': # or tree[0].dep_ == 'prep'
    tree = tree[1:]
  return tree  

def extractSubtree_experimental(doc: spacy.tokens.doc.Doc, idx: int):
  prep_token_object = doc[idx:idx+1] 
  subtree_object = [descendant for descendant in prep_token_object.subtree]
  if subtree_object:
    purified = purifytree(subtree_object)
    while purified != subtree_object:
      subtree_object = purified
      purified = purifytree(subtree_object)
    subtree_object = purified
  if subtree_object:
    subtree = [t.text for t in subtree_object]
    last_token_lemmatized = [t.text for t in subtree_object[:-1]] + [subtree_object[-1].lemma_]
    all_tokens_lemmatized = [t.lemma_ for t in subtree_object]
    return ' '.join(subtree), subtree_object, ' '.join(last_token_lemmatized), ' '.join(all_tokens_lemmatized)
  else: 
    return '', subtree_object, '', ''

def findPP(doc: spacy.tokens.doc.Doc) -> List[str]:
  locative_prepositions = ['in', 'at', 'on', 'above', 'below', 'behind', 'under', 'around', 'bottom', 'top', 'near', 'inside', 'outside', 'through']
  res = []
  for token in doc:
    if token.tag_ == ['IN'] or token.pos_ == 'ADP' or token.dep_ == 'prep':
      if token.text in locative_prepositions:
        if token.head.dep_ == 'ROOT':
          res += [extractSubtree(doc, token.i)]
          token_nominal_children = [t for t in list(token.children) if t.pos_ in ['NOUN', 'PROPN']]
          if token_nominal_children:
            for tnc in token_nominal_children:
              res += [tnc.text]
              x1, _, x2, x3 = extractSubtree_experimental(doc, tnc.i)
              res += [x1, x2, x3]
          n1, _, n2, n3 = extractSubtree_experimental(doc, token.i)          
          res += [n1, n2, n3]
        else:
          n1, n_obj, n2, n3 = extractSubtree_experimental(doc, token.i)
          if n_obj and n_obj[0].pos_ == 'ADP':
            nominal_child = [x for x in list(n_obj[0].children) if x.pos_ in ['NOUN', 'PROPN']]
            if nominal_child:
              focus = nominal_child[0]
              x1, xobj, x2, x3 = extractSubtree_experimental(doc, focus.i)
              res += [x1, x2, x3]
          res += [n1, n2, n3]
      else:
        n1, n_obj, n2, n3 = extractSubtree_experimental(doc, token.i)
        if n_obj and n_obj[0].pos_ == 'ADP':
          nominal_child = [x for x in list(n_obj[0].children) if x.pos_ in ['NOUN', 'PROPN']]
          if nominal_child:
            focus = nominal_child[0]
            x1, xobj, x2, x3 = extractSubtree_experimental(doc, focus.i)
            res += [x1, x2, x3]
            following_prep_check = [x for x in xobj if x.pos_ == 'ADP' and x.i > focus.i]
            if following_prep_check:
              # print('following_prep_check:', following_prep_check)
              for p in following_prep_check:
                new_end_idx1 = p.i - xobj[0].i
                new_find1 = [x for x in xobj[:new_end_idx1]]
                # print('new_find1:', new_find1)
                new_beginning_idx = p.i + 1 - xobj[0].i
                new_find2 = [x for x in xobj[new_beginning_idx:]]
                # print('new_find2:', new_find2)
                res += [' '.join([x.text for x in new_find1])] + [' '.join([x.text for x in new_find2])]
                res += [' '.join([x.lemma_ for x in new_find1])] + [' '.join([x.lemma_ for x in new_find2])]
                if new_find1:
                  res += [' '.join([x.text for x in new_find1[:-1]] + [new_find1[-1].lemma_])]
                if new_find2:
                  res += [' '.join([x.text for x in new_find2[:-1]] + [new_find1[-1].lemma_])]
  return list(set(res))

def generalExtraction(doc: spacy.tokens.doc.Doc) -> List[str]:
  import re
  res = []  
  root_conj_tokens = []
  root_tokens = [token for token in doc if token.dep_ == 'ROOT' and token.pos_ in ['VERB', 'AUX']]
  if root_tokens:
    for r in root_tokens:
      root_conj_tokens += [(r, token) for token in doc if token.head.i == r.i and token.dep_ == 'conj' and token.pos_ in ['VERB', 'AUX']]
    #   if r.text == 'happen':
    #     print(root_conj_tokens)
  for token in doc:
    if token in root_tokens:
      root_token = token
      root_children = list(root_token.children)
      for child in root_children:
        if child.pos_ in ['NOUN', 'PROPN'] or child.dep_ == 'prep':
          interim_text, interim_obj, last_token_lemmatized, all_tokens_lemmatized = extractSubtree_experimental(doc, child.i)
          res += [interim_text] + [last_token_lemmatized] + [all_tokens_lemmatized]
          if ' of ' in interim_text:
            of_token = [t for t in interim_obj if t.text == 'of'][0]
            of_id = of_token.i
            of_dependent_token = [t for t in interim_obj if t.head.i == of_id and t.pos_ in ['NOUN', 'PROPN']]
            if of_dependent_token:
              res += [of_dependent_token[0].text]
              of_dependent_id = of_dependent_token[0].i
              res += [extractSubtree_experimental(doc, of_dependent_id)[0]]
          if [x for x in interim_obj if x.dep_ == 'prep' and x.text != 'of']:
            preposition_token = [x for x in interim_obj if x.dep_ == 'prep'][0]
            interim_text_p, interim_obj_p, last_token_lemmatized_p, all_tokens_lemmatized_p = extractSubtree_experimental(doc, preposition_token.i)
            res += [interim_text_p] + [last_token_lemmatized_p] + [all_tokens_lemmatized_p]
          rel = [x for x in interim_obj if x.dep_ == 'relcl']
          
          if rel:
            _, rel_subtree_obj, _, _ = extractSubtree_experimental(doc, rel[0].i)
            if rel_subtree_obj:
              # print('rel_subtree_obj:', rel_subtree_obj)
              rel_beginning_point = rel_subtree_obj[0].i - interim_obj[0].i
              new_find = [x for x in interim_obj[:rel_beginning_point]]
              if new_find:
                # print('new_find:', new_find)
                res += [' '.join([x.text for x in new_find])] + [' '.join([x.lemma_ for x in new_find])] + [' '.join([x.text for x in new_find[:-1]] + [new_find[-1].lemma_])]
              

        if (root_token, child) in root_conj_tokens:
          sister_token_tuples = [x for x in root_conj_tokens if x[0] == root_token]
          for sister_token_tuple in sister_token_tuples:
            sister_token = sister_token_tuple[1]
            root_sister_children = list(sister_token.children)
            for sister_child in root_sister_children:
              if sister_child.pos_ in ['NOUN', 'PROPN'] or sister_child.dep_ == 'prep':
                interim_text, interim_obj, last_token_lemmatized, all_tokens_lemmatized = extractSubtree_experimental(doc, sister_child.i)                
                res += [interim_text] + [last_token_lemmatized] + [all_tokens_lemmatized]
                res += [sister_child.text]
                if ' of ' in interim_text:
                  of_token = [t for t in interim_obj if t.text == 'of'][0]
                  of_id = of_token.i
                  of_dependent_token = [t for t in interim_obj if t.head.i == of_id and t.pos_ in ['NOUN', 'PROPN']]
                  if of_dependent_token:
                    res += [of_dependent_token[0].text]
                    of_dependent_id = of_dependent_token[0].i
                    res += [extractSubtree_experimental(doc, of_dependent_id)[0]] 
                if [x for x in interim_obj if x.dep_ == 'prep' and x.text != 'of']:
                  preposition_token = [x for x in interim_obj if x.dep_ == 'prep'][0]
                  interim_text_p, interim_obj_p, last_token_lemmatized_p, all_tokens_lemmatized_p = extractSubtree_experimental(doc, preposition_token.i)
                  res += [interim_text_p] + [last_token_lemmatized_p] + [all_tokens_lemmatized_p]     
  return list(set(res))
  
def findADV(doc: spacy.tokens.doc.Doc) -> List[str]:
  res = []
  for token in doc:
    if token.dep_ in ['advmod', 'npadvmod'] and token.head.dep_ == 'ROOT' and token.i > token.head.i:
      interim_text, interim_obj, last_token_lemmatized, all_tokens_lemmatized = extractSubtree_experimental(doc, token.i)
      res += [interim_text] + [last_token_lemmatized] + [all_tokens_lemmatized]
  return list(set(res))
  
def anythingNominal(doc: spacy.tokens.doc.Doc) -> List[str]:
  res = []
  for token in doc:
    if token.pos_ in ['NOUN', 'PROPN']:
      res += [token.text] + [token.lemma_]
      comps = [t for t in doc if t.head.i == token.i and t.dep_ == 'compound']
      if comps:
        comps_idx = [t.i for t in comps] + [token.i]
        beginning, end = min(comps_idx), max(comps_idx)
        kk = doc[beginning:end+1]
        if kk:
          res += [' '.join([x.text for x in kk])] + [' '.join([x.lemma_ for x in kk])] + [' '.join([x.text for x in kk[:-1]] + [kk[-1].lemma_])]
  return list(set(res))  


def find_loc_candidate(paragraph: spacy.tokens.doc.Doc) -> List[str]:
    """
    function added by GEE.
    cases:
    NOUN or PROPN (and its dependents) w/ head being the VERB or AUX (core arguments)
    ADVMOD (and its dependents) w/ head being the VERB or AUX (adverbs of location)
    NOUN or PROPN (and its dependents) w/ head being an ADP whose head is the VERB or AUX. (objects of preposition)
    for all the above, when reading dependents, don't keep the DET
    """
    core_args = coreArguments(paragraph)
    loc = locativeADV(paragraph)
    pps = findPP(paragraph)
    ents = taggingMistake(paragraph)
    adv = findADV(paragraph)
    ge = generalExtraction(paragraph)
    nom = anythingNominal(paragraph)
    return list(set(core_args + loc + ents + pps + adv + ge + nom))


def commonsense_location(tokenized_paragraph, entity):
    import pickle
    nlp2 = spacy.load("en_core_web_lg", disable = ['ner'])
    sims = []   
    with open(os.path.join('/data/ghazaleh/datasets/knowledge_pickles/', 'located.pickle'), 'rb') as rf:
      locateddict = pickle.load(rf)     
    doc1 = nlp2(tokenized_paragraph)
    # doc1 = nlp(tokenized_paragraph)
    for tp in locateddict[entity]:
        word, weight = tp
        if word.strip():
            doc2 = nlp2(word)
            # doc2 = nlp(word)
            if doc2 and doc2.vector_norm:
                sims += [(word, doc1.similarity(doc2)*float(weight))]
    if sims:
        sims = sorted(sims, key=lambda x:x[1], reverse=True)
        # print('For ', entity, ': found commonsense location:', sims[0])
        return [sims[0][0]]
    else:
        return sims

def find_loc_candidate3(paragraph_doc: spacy.tokens.doc.Doc, entity_list: List[str]) -> List[str]:
  nouns, subtrees, adv = [], [], [] 
  # tokenized_paragraph = ' '.join([t.text for t in paragraph_doc])
  # commonsense_locations = []
  # for e in entity_list:
  #   commonsense_locations = commonsense_location(tokenized_paragraph, e)
  # commonsense_locations = list(set(commonsense_locations))
  for token in paragraph_doc:
    if token.pos_ == 'NOUN':
      if not token.dep_ == 'ROOT' and not (token.dep_ == 'conj' and token.head.dep_ == 'ROOT'):
        nouns.append(token.text)
        nouns.append(token.lemma_)
        subtree = ' '.join([t.text for t in token.subtree])
        subtrees += [subtree]
    elif token.pos_ == 'ADV' and (token.head.dep_ == 'ROOT' and token.head.pos_ in ['VERB', 'AUX']):
      from nltk import pos_tag, word_tokenize
      adv_pos = pos_tag(word_tokenize(token.text))
      if adv_pos:
        adv_pos = adv_pos[0][1]
        if adv_pos.startswith('N'):
          adv += [token.text]
          adv += [' '.join([t.text for t in token.subtree])]      
  return list(set(nouns + subtrees + adv)) # commonsense_locations

def find_loc_candidate4(paragraph: spacy.tokens.doc.Doc) -> List[str]:
    """
    paragraph: the paragraph after tokenization and lower-case transformation
    return: the location candidates found in this paragraph
    """    
    pos_list = [(token.text, token.pos_, token.dep_) for token in paragraph]
    loc_list = []

    # extract nouns (including 'noun + noun' phrases)
    for i in range(len(pos_list)):
        if pos_list[i][1] == 'NOUN':
            candidate = pos_list[i][0]
            for k in range(1, i+1):
                if pos_list[i-k][1] == 'ADJ':
                    candidate = pos_list[i-k][0] + ' ' + candidate
                elif pos_list[i-k][1] == 'NOUN':
                    loc_list.append(candidate)
                    candidate = pos_list[i-k][0] + ' ' + candidate
                else:
                    break
            loc_list.append(candidate)

    # extract 'noun + and/or + noun' phrase
    for i in range(2, len(pos_list)):
        if pos_list[i][1] == 'NOUN' \
            and (pos_list[i-1][0] == 'and' or pos_list[i-1][0] == 'or') \
                and pos_list[i-2][1] == 'NOUN':
            loc_list.append(pos_list[i-2][0] + ' ' + pos_list[i-1][0] + ' ' + pos_list[i][0])

    # noun + of + noun phrase
    for i in range(2, len(pos_list)):
        if pos_list[i][1] == 'NOUN' \
            and pos_list[i-1][0] == 'of' \
                and pos_list[i-2][1] == 'NOUN':
            loc_list.append(pos_list[i-2][0] + ' ' + pos_list[i-1][0] + ' ' + pos_list[i][0])

    # noun + of + a/an/the + noun phrase
    for i in range(3, len(pos_list)):
        if pos_list[i][1] == 'NOUN' \
            and pos_list[i-1][1] == 'DET' \
                and pos_list[i-2][0] == 'of' \
                    and pos_list[i-3][1] == 'NOUN':
            loc_list.append(pos_list[i-3][0] + ' ' + pos_list[i-2][0] + ' ' + pos_list[i-1][0] + ' ' + pos_list[i][0])
    
    for token in paragraph:
      if token.pos_ == 'ADV' and (token.head.dep_ == 'ROOT' and token.head.pos_ in ['VERB', 'AUX']):
        from nltk import pos_tag, word_tokenize
        adv_pos = pos_tag(word_tokenize(token.text))
        if adv_pos:
          adv_pos = adv_pos[0][1]
          # nominal adv
          if adv_pos.startswith('N'):
            loc_list += [token.text]
            loc_list += [' '.join([t.text for t in token.subtree])]      
      # pp
      elif token.pos_ == 'ADP' and token.head.dep_ == 'ROOT' and token.head.pos_ in ['VERB', 'AUX']:
        loc_list += [' '.join([t.text for t in token.subtree][1:])]
      

    # lemmatization
    for i in range(len(loc_list)):
        _, location = lemmatize(loc_list[i])
        loc_list[i] = location
    
    return loc_list


# TODO: Maybe we shouldn't perform lemmatization to location candidates for the test set
#       in order to generate raw spans in the paragraph while filling the grids.
#       (candidate masks are still computed after masking both the candidate and the paragraph)
# def find_loc_candidate(paragraph: spacy.tokens.doc.Doc) -> Set[str]:
#     """
#     paragraph: the paragraph after tokenization and lower-case transformation
#     return: the location candidates found in this paragraph
#     """
#     pos_tagger.predict(paragraph)
#     pos_list = [(token.text, token.get_labels('pos')) for token in paragraph]
#     loc_list = []

#     # extract nouns (including 'noun + noun' phrases)
#     for i in range(len(pos_list)):
#         if pos_list[i][1] == 'NOUN':
#             candidate = pos_list[i][0]
#             for k in range(1, i+1):
#                 if pos_list[i-k][1] == 'ADJ':
#                     candidate = pos_list[i-k][0] + ' ' + candidate
#                 elif pos_list[i-k][1] == 'NOUN':
#                     loc_list.append(candidate)
#                     candidate = pos_list[i-k][0] + ' ' + candidate
#                 else:
#                     break
#             loc_list.append(candidate)

#     # extract 'noun + and/or + noun' phrase
#     for i in range(2, len(pos_list)):
#         if pos_list[i][1] == 'NOUN' \
#             and (pos_list[i-1][0] == 'and' or pos_list[i-1][0] == 'or') \
#                 and pos_list[i-2][1] == 'NOUN':
#             loc_list.append(pos_list[i-2][0] + ' ' + pos_list[i-1][0] + ' ' + pos_list[i][0])

#     # noun + of + noun phrase
#     for i in range(2, len(pos_list)):
#         if pos_list[i][1] == 'NOUN' \
#             and pos_list[i-1][0] == 'of' \
#                 and pos_list[i-2][1] == 'NOUN':
#             loc_list.append(pos_list[i-2][0] + ' ' + pos_list[i-1][0] + ' ' + pos_list[i][0])

#     # noun + of + a/an/the + noun phrase
#     for i in range(3, len(pos_list)):
#         if pos_list[i][1] == 'NOUN' \
#             and pos_list[i-1][1] == 'DET' \
#                 and pos_list[i-2][0] == 'of' \
#                     and pos_list[i-3][1] == 'NOUN':
#             loc_list.append(pos_list[i-3][0] + ' ' + pos_list[i-2][0] + ' ' + pos_list[i-1][0] + ' ' + pos_list[i][0])
    
#     # lemmatization
#     for i in range(len(loc_list)):
#         _, location = lemmatize(loc_list[i])
#         loc_list[i] = location
    
#     return set(loc_list)


def find_mention(paragraph: List[str], phrase: str, norm: bool) -> List:
    """
    Judge whether a phrase is a span of the paragraph (or sentence) and return the span
    norm: whether the sentence should be normalized first
    """
    phrase = phrase.strip().split()
    phrase_len = len(phrase)
    span_list = []

    # perform lemmatization on both the paragraph and the phrase
    if norm:
        paragraph, _ = lemmatize(' '.join(paragraph))
        phrase, _ = lemmatize(' '.join(phrase))

    for i in range(0, len(paragraph) - phrase_len):
        sub_para = paragraph[i: i+phrase_len]
        if sub_para == phrase:
            span_list.extend(list(range(i, i+phrase_len)))
    return span_list


def log_existence(paragraph: str, para_id: int, entity: str, loc_seq: List[str], log_file):
    """
    Record the entities and locations that does not match any span in the paragraph.
    """
    entity_list = re.split('; |;', entity)
    paragraph = paragraph.strip().split()
    for ent in entity_list:
        if not find_mention(paragraph, ent, norm = False) and not find_mention(paragraph, ent, norm = True):
            print(f'[WARNING] Paragraph {para_id}: entity "{ent}" is not a span in paragraph.', file=log_file)
    
    for loc in loc_seq:
        if loc == '-' or loc == '?':
            continue
        if not find_mention(paragraph, loc, norm = True):
            print(f'[WARNING] Paragraph {para_id}: location "{loc}" is not a span in paragraph.', file=log_file)


def get_entity_mask(sentence: str, entity: str, pad_bef_len: int, pad_aft_len: int) -> List[int]:
    """
    return the masked vector pertaining to a certain entity in the paragraph
    """
    sentence = sentence.strip().split()
    sent_len = len(sentence)
    entity_list = re.split('; |;', entity)
    span_list = []
    for ent_name in entity_list:
        span_list.extend(find_mention(sentence, ent_name, norm = False) or find_mention(sentence, ent_name, norm = True))
    
    entity_mask = [1 if i in span_list else 0 for i in range(sent_len)]
    padding_before = [0 for _ in range(pad_bef_len)]
    padding_after = [0 for _ in range(pad_aft_len)]

    return padding_before + entity_mask + padding_after


def get_verb_mask(sentence: str, pad_bef_len: int, pad_aft_len: int) -> List[int]:
    """
    return the masked vector pertaining to the verb in the sentence
    """
    sentence = nlp(sentence)
    sent_len = len(sentence)
    pos_list = [(token.text, token.pos_) for token in sentence]
    span_list = [i for i in range(sent_len) if pos_list[i][1] == 'VERB']
    
    verb_mask = [1 if i in span_list else 0 for i in range(sent_len)]
    padding_before = [0 for _ in range(pad_bef_len)]
    padding_after = [0 for _ in range(pad_aft_len)]

    return padding_before + verb_mask + padding_after


def get_location_mask(sentence: str, location: str, pad_bef_len: int, pad_aft_len: int) -> List[int]:
    """
    return the masked vector pertaining to a certain location in the paragraph
    """
    sentence = sentence.strip().split()
    sent_len = len(sentence)
    span_list = find_mention(sentence, location, norm = True)
    
    loc_mask = [1 if i in span_list else 0 for i in range(sent_len)]
    padding_before = [0 for _ in range(pad_bef_len)]
    padding_after = [0 for _ in range(pad_aft_len)]

    return padding_before + loc_mask + padding_after


def compute_state_change_seq(gold_loc_seq: List[str]) -> List[str]:
    """
    Compute the state change sequence for the certain entity.
    Note that the gold location sequence contains an 'initial state'.
    State change labels: O_C, O_D, E, M, C, D
    """
    num_states = len(gold_loc_seq)
    # whether the entity has been created. (if exists from the beginning, then it should be True)
    # GEE: this line says that if gold_loc_seq[0] == '-' (and create == False), then this entity did not exist and will be created later in this process. Else (if create == True), then this entity existed already.
    create = False if gold_loc_seq[0] == '-' else True
    gold_state_seq = []

    for i in range(1, num_states):
        if gold_loc_seq[i] == '-':  # could be O_C, O_D or D
            if create == True and gold_loc_seq[i-1] == '-':
                gold_state_seq.append('O_D')
            elif create == True and gold_loc_seq[i-1] != '-':
                gold_state_seq.append('D')
            else:
                gold_state_seq.append('O_C')

        elif gold_loc_seq[i] == gold_loc_seq[i-1]:
            # E means exists: no state change
            gold_state_seq.append('E')

        else:  # location change, could be C or M
            if gold_loc_seq[i-1] == '-':
                create = True
                gold_state_seq.append('C')
            else:
                gold_state_seq.append('M')
    
    assert len(gold_state_seq) == len(gold_loc_seq) - 1
    
    return gold_state_seq


def read_paragraph(filename: str) -> Dict[int, Dict]:

    csv_data = pd.read_csv(filename)
    paragraph_result = {}
    max_sent = len(csv_data.columns) - 3  # should equal to 10 in this case
    # print(max_sent)
    for _, row in csv_data.iterrows():
        para_id = int(row['Paragraph ID'])
        # print(para_id)
        topic = row['Topic']
        prompt = row['Prompt']
        sent_list = []

        for i in range(1, max_sent + 1):  
            # print(i)
            sent = row[f'Sentence{i}']
            if pd.isna(sent):
                break
            sent_list.append(sent)

        text = ' '.join(sent_list)
        paragraph_result[para_id] = {'id': para_id,
                                     'topic': topic,
                                     'prompt': prompt,
                                     'paragraph': text,
                                     'total_sents': len(sent_list)}
    
    total_paras = len(paragraph_result)
    print(f'Paragraphs read: {total_paras}')
    return paragraph_result


def read_paragraph_from_sentences(csv_data: pd.DataFrame, begin_row_index: int, total_sents: int) -> str:
    """
    Read the paragraph from State_change_annotations.csv.
    This is because the paragraph in this file and the original Paragraphs.csv may be different and will cause problems.
    """
    row_index = begin_row_index + 3  # row index of the first sentence
    sent_list = []

    for i in range(total_sents):  # read each sentence
        row = csv_data.iloc[row_index]
        assert row['sent_id'] == f'event{i + 1}'
        sentence = row['sentence']
        sent_list.append(sentence)
        row_index += 2

    return ' '.join(sent_list)


def read_annotation(filename: str, paragraph_result: Dict[int, Dict],
                    log_file, test: bool) -> List[Dict]:
    """
    1. read csv
    2. get the entities
    3. tokenize the paragraph and change to lower case
    3. extract location candidates
    4. for each entity, create an instance for it
    5. read the entity's initial state
    5. read each sentence, give it an ID
    6. compute entity mask (length of mask vector = length of paragraph)
    7. extract the nearest verb to the entity, compute verb mask
    8. for each location candidate, compute location mask
    9. read entity's state at current timestep
    10. for the train/dev sets, if gold location is not extracted in step 3,
        add it to the candidate set (except for '-' and '?'). Back to step 6
    11. reading ends, compute the number of sentences
    12. get the number of location candidates
    13. infer the gold state change sequence
    """

    data_instances = []
    column_names = ['para_id', 'sent_id', 'sentence', 'ent1', 'ent2', 'ent3',
                    'ent4', 'ent5', 'ent6', 'ent7', 'ent8']
    max_entity = 8

    csv_data = pd.read_csv(filename, header = None, names = column_names)
    row_index = 0
    para_index = 0

    # variables for computing the accuracy of location prediction
    total_loc_cnt = 0
    total_err_cnt = 0

    start_time = time.time()

    while True:

        row = csv_data.iloc[row_index]
        if pd.isna(row['para_id']):  # skip empty lines
            row_index += 1
            continue

        para_id = int(row['para_id'])
        if para_id not in paragraph_result:  # keep the dataset split
            row_index += 1
            continue
        
        # the number of lines we need to read is relevant to 
        # the number of sentences in this paragraph
        total_sents = paragraph_result[para_id]['total_sents']
        total_lines = 2 * total_sents + 3
        begin_row_index = row_index  # first line of this paragraph in csv
        end_row_index = row_index + total_lines - 1  # last line

        # tokenize, lower cased
        raw_paragraph = read_paragraph_from_sentences(csv_data = csv_data, begin_row_index = begin_row_index, total_sents = total_sents)
        paragraph, total_tokens = tokenize(raw_paragraph)
        prompt, _ = tokenize(paragraph_result[para_id]['prompt'])

        

        # process data in this paragraph
        # first, figure out how many entities it has
        entity_list = []
        for i in range(1, max_entity + 1):
            entity_name = row[f'ent{i}']
            if pd.isna(entity_name):
                break
            entity_list.append(entity_name)
        
        # find location candidates
        loc_cand_set = set(find_loc_candidate4(nlp(paragraph)))
        print(f'Paragraph {para_id}: \nLocation candidate set: ', loc_cand_set, file=log_file)        


        total_entities = len(entity_list)
        verb_mention_per_sent = [None for _ in range(total_sents)]

        # sets for computing the accuracy of location prediction
        total_loc_set = set()
        total_err_set = set()

        for i in range(total_entities):
            entity_name = entity_list[i]

            instance = {'id': para_id,
                        'topic': paragraph_result[para_id]['topic'],
                        'prompt': prompt,
                        'paragraph': paragraph,
                        'total_tokens': total_tokens,
                        'total_sents': total_sents,
                        'entity': entity_name}
            gold_loc_seq = []  # list of gold locations
            sentence_list = []
            sentence_concat = []

            # read initial state, skip the prompt line
            row_index += 2
            row = csv_data.iloc[row_index]
            assert row['sent_id'] == 'state1'
            # GEE: if you want to lemmatize the gold location, you need to do it for location candidates as well. 
            # or, conversely, you can skip lemmatization for both. Just be consistent.      
            _, gold_location = lemmatize(row[f'ent{i+1}'])
            gold_loc_seq.append(gold_location)

            # for each sentence, read the sentence and the entity location
            for j in range(total_sents):

                # read sentence
                row_index += 1
                row = csv_data.iloc[row_index]
                assert row['sent_id'] == f'event{j+1}'
                sentence, num_tokens_in_sent = tokenize(row['sentence'])
                sentence_concat.append(sentence)
                sent_id = j + 1

                # read gold state
                row_index += 1
                row = csv_data.iloc[row_index]
                assert row['sent_id'] == f'state{j+2}'
                # GEE: if you want to lemmatize the gold location, you need to do it for location candidates as well. 
                # or, conversely, you can skip lemmatization for both. Just be consistent.  
                _, gold_location = lemmatize(row[f'ent{i+1}'])
                gold_loc_seq.append(gold_location)

                if gold_location != '-' and gold_location != '?':
                    total_loc_set.add(gold_location)


                # whether the gold location is in the candidates (training only)
                if gold_location not in loc_cand_set \
                    and gold_location != '-' and gold_location != '?':
                    # GEE: THIS IS VERY IMPORTANT! IF IN TRAIN OR DEV SET, THEN DON'T HESTITATE TO COPY GOLD LOCATION FROM DATA. 
                    # THEN OUR MODEL WILL HAVE TO LEARN TO PICK THAT.
                    if not test:
                        loc_cand_set.add(gold_location)
                    # GEE: This is to keep track of where our heuristics for extracting location candidates failed.
                    total_err_set.add(gold_location)
                    # GEE: This will be useful later for error analysis.
                    print(f'[INFO] Paragraph {para_id}: gold location "{gold_location}" not included in candidate set.',
                         file=log_file)

                sentence_list.append({'id': sent_id, 
                                      'sentence': sentence, 
                                      'total_tokens': num_tokens_in_sent})
            
            assert len(sentence_list) == total_sents
            loc_cand_list = list(loc_cand_set)
            total_loc_candidates = len(loc_cand_list)
            # record the entities and locations that does not match any span in the paragraph
            # GEE: This captures world knowledge
            entity_name, _ = tokenize(entity_name)
            log_existence(paragraph, para_id, entity_name, gold_loc_seq, log_file)
            
            words_read = 0  # how many words have been read
            for j in range(total_sents):

                sent_dict = sentence_list[j]
                sentence = sent_dict['sentence']
                num_tokens_in_sent = sent_dict['total_tokens']

                # compute the masks
                entity_mask = get_entity_mask(sentence, entity_name, words_read, total_tokens - words_read)
                entity_mention = [idx for idx in range(len(entity_mask)) if entity_mask[idx] == 1]

                if not verb_mention_per_sent[j]:
                    verb_mask = get_verb_mask(sentence, words_read, total_tokens - words_read)
                    assert len(entity_mask) == len(verb_mask)
                    verb_mention = [idx for idx in range(len(verb_mask)) if verb_mask[idx] == 1]
                    verb_mention_per_sent[j] = verb_mention
                else:
                    verb_mention = verb_mention_per_sent[j]
                
                loc_mention_list = []
                for loc_candidate in loc_cand_list:
                    loc_mask = get_location_mask(sentence, loc_candidate, words_read, total_tokens - words_read)
                    assert len(entity_mask) == len(loc_mask)
                    loc_mention = [idx for idx in range(len(loc_mask)) if loc_mask[idx] == 1]
                    loc_mention_list.append(loc_mention)

                sent_dict['entity_mention'] = entity_mention
                sent_dict['verb_mention'] = verb_mention
                sent_dict['loc_mention_list'] = loc_mention_list
                sentence_list[j] = sent_dict
                words_read += num_tokens_in_sent

            assert words_read == total_tokens # after reading all sentences, sum(len(sentences tokens)) should == the len(paragraph tokens)
            assert len(gold_loc_seq) == len(sentence_list) + 1
            instance['sentence_list'] = sentence_list
            instance['loc_cand_list'] = loc_cand_list
            instance['total_loc_candidates'] = total_loc_candidates
            instance['gold_loc_seq'] = gold_loc_seq
            instance['gold_state_seq'] = compute_state_change_seq(gold_loc_seq)
            # print(instance)
            assert paragraph == ' '.join(sentence_concat), f'at paragraph #{para_id}'

            # pointer backward, construct instance for next entity
            row_index = begin_row_index
            data_instances.append(instance)

        total_loc_cnt += len(total_loc_set)
        total_err_cnt += len(total_err_set)
        # print(total_loc_set)
        # print(total_err_set)


        row_index = end_row_index + 1
        para_index += 1

        if para_index % 10 == 0:
            end_time = time.time()
            print(f'[INFO] {para_index} paragraphs processed. Time elapsed: {end_time - start_time}s')
        if para_index >= len(paragraph_result):
            end_time = time.time()
            print(f'[INFO] All {para_index} paragraphs processed. Time elapsed: {end_time - start_time}s')
            break
    
    # compute accuracy of location prediction
    loc_accuracy = 1 - total_err_cnt / total_loc_cnt
    print(f'[DATA] Recall of location prediction: {loc_accuracy} ({total_loc_cnt - total_err_cnt}/{total_loc_cnt})')

    return data_instances


def read_split(filename: str, paragraph_result: Dict[int, Dict]):

    train_para, dev_para, test_para = {}, {}, {}
    csv_data = pd.read_csv(filename)

    for _, row in csv_data.iterrows():

        para_id = int(row['Paragraph ID'])
        para_data = paragraph_result[para_id]
        partition = row['Partition']
        if partition == 'train':
            train_para[para_id] = para_data
        elif partition == 'dev':
            dev_para[para_id] = para_data
        elif partition == 'test':
            test_para[para_id] = para_data
        
    print('Number of train paragraphs: ', len(train_para))
    print('Number of dev paragraphs: ', len(dev_para))
    print('Number of test paragraphs: ', len(test_para))

    return train_para, dev_para, test_para


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-para_file', type=str, default='/data/ghazaleh/neuralsymbolic/data/corefed_propara/Paragraphs.csv', help='path to the paragraph csv')
    parser.add_argument('-state_file', type=str, default='/data/ghazaleh/neuralsymbolic/data/corefed_propara/State_change_annotations.csv', 
                        help='path to the state annotation csv')
    parser.add_argument('-split_file', type=str, default='data/Train_Dev_Test.csv',
                        help='path to the csv that annotates the train/dev/test split')
    parser.add_argument('-log_dir', type=str, default='logs',
                        help='directory to store the intermediate outputs')
    parser.add_argument('-store_dir', type=str, default='data/', 
                        help='directory that you would like to store the generated instances')
    opt = parser.parse_args()

    print('Received arguments:')
    print(opt)
    print('-' * 50)

    paragraph_result = read_paragraph(opt.para_file)
    train_para, dev_para, test_para = read_split(opt.split_file, paragraph_result)

    log_file = open(f'{opt.log_dir}/info.log', 'w+', encoding='utf-8')
    # save the instances to JSON files
    print('Dev Set......')
    dev_instances = read_annotation(opt.state_file, dev_para, log_file, test = False)
    json.dump(dev_instances, open(os.path.join(opt.store_dir, 'dev.json'), 'w', encoding='utf-8'),
                ensure_ascii=False, indent=4)
                
    print('Testing Set......')
    test_instances = read_annotation(opt.state_file, test_para, log_file, test = True)
    json.dump(test_instances, open(os.path.join(opt.store_dir, 'test.json'), 'w', encoding='utf-8'),
                ensure_ascii=False, indent=4)

    print('Training Set......')
    train_instances = read_annotation(opt.state_file, train_para, log_file, test = False)
    json.dump(train_instances, open(os.path.join(opt.store_dir, 'train.json'), 'w', encoding='utf-8'),
                ensure_ascii=False, indent=4)

    print('[INFO] JSON files saved successfully.')

    log_file.close()
