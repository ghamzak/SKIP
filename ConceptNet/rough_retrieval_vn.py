'''
 @Date  : 01/11/2022
 @Author: Ghazaleh Kazeminejad
 @mail  : ghka9436@colorado.edu
'''
import json, os, re, sys
from tqdm import tqdm
import argparse
from typing import Set, List, Tuple
import spacy
nlp = spacy.load("en_core_web_sm", disable = ['ner'])
# from spacy.lang.en import STOP_WORDS
# STOP_WORDS = set(STOP_WORDS) - {'bottom', 'serious', 'top', 'alone', 'around', 'used', 'behind', 'side', 'mine', 'well'}
from Stemmer import PorterStemmer
stemmer = PorterStemmer()


parser = argparse.ArgumentParser()
parser.add_argument('-train', type=str, default='../data/train.json', help='path to the training set')
parser.add_argument('-dev', type=str, default='../data/dev.json', help='path to the dev set')
parser.add_argument('-test', type=str, default='../data/test.json', help='path to the test set')
parser.add_argument('-semparse', type=str, default='/data/ghazaleh/datasets/propara_parsed', help='path to propara parsed json files')
parser.add_argument('-output', type=str, default='./rough_retrieval_vn.json', help='file to store the vn output text')
opt = parser.parse_args()


thematic_roles = ['Trajectory', 'Agent', 'Destination', 'Asset', 'Affector', 'Attribute', 'Axis', \
    'Beneficiary', 'Causer', 'Cause', 'Circumstance', 'Co-Agent', 'Co-Patient', 'Co-Theme', 'Duration', \
        'Eventuality', 'Experiencer', 'Goal', 'Initial Location', 'Initial State', 'Instrument', \
            'Location', 'Maleficiary', 'Manner', 'Material', 'Path', 'Patient', 'Pivot', 'Product', \
                'Recipient', 'Reflexive', 'Result', 'Source', 'Stimulus', 'Theme', 'Topic', 'Value', \
                    'Co_Agent', 'Co_Patient', 'Co_Theme', 'Initial_Location', 'Initial_State']
thematic_roles = [t.lower() for t in thematic_roles]

def stem(word: str) -> str:
    """
    Stem a single word
    """
    word = word.lower().strip()
    return stemmer.stem(word)

def lemmatize(phrase: str) -> str:
    """
    lemmatize a phrase
    """
    phrase = phrase.lower().strip()
    doc = nlp(phrase)
    return ' '.join([token.lemma_ for token in doc])

def overlaps(entity, argument):    
    return entity in argument

def accepted_preds() -> List[str]:
    pf = '/home/ghazaleh/dissertation/neural_symbolic/LEXIS3/knowledgebased/predicate2sent.txt'
    with open(pf, 'r') as rf:
        p2s = [x.strip() for x in rf.readlines()]
    preds = [x.split('(')[0] for x in p2s]
    return list(set(preds)) #+ ['has_state']

def getPPhead(phrase:str) -> str:
    """
    to capture nominal in PP, e.g. in the sediment, on top of the original sediment -> sediment
    """
    doc = nlp(phrase)
    if len(doc) <= 1:
        return phrase
    if doc[0].pos_ == 'ADP':        
        np = [[x for x in t.subtree] for t in doc[0].rights if t.pos_ == 'NOUN']
        if np and np[0]:
            if not 'of' in [t.text for t in np[0]]:
                keep = doc[1:]
                # print(keep)
                return getPPhead(' '.join([t.text for t in keep])) 
            else:
                of_index = [t.i for t in doc if t.text == 'of'][0]
                keep = doc[of_index+1:]
                # print(keep)
                return getPPhead(' '.join([t.text for t in keep]))
        else:
            keep = doc[1:]
            # print(keep)
            return getPPhead(' '.join([t.text for t in keep]))

    elif doc[0].pos_ in ['DET', 'ADJ', 'ADP', 'ADV']:
        keep = doc[1:]
        # print(keep)
        return getPPhead(' '.join([t.text for t in keep]))    
    else:
        return phrase



def search_triple(entity:str, parse_dir:str, sidlist:List[str]) -> List[str]:
    """
    Search matched triples in ConceptNet to the given entity.
    """
    global thematic_roles
    # global cpnet
    # entity = remove_stopword(entity)
    ok_preds = accepted_preds()
    stem_entity = lemmatize(entity)# ''.join(map(stem, entity))
    subevent_list = []
    parse_file_list = [fname for sid in sidlist for fname in os.listdir(parse_dir) if re.sub('sentence', '', fname.split('.')[0]) == str(sid)]
    for fname in parse_file_list:
        with open(os.path.join(parse_dir, fname), 'r') as rf:
            s_parse_dict = json.load(rf)
        if type(s_parse_dict) == dict and 'props' in s_parse_dict.keys() and s_parse_dict['props']:
            for event in s_parse_dict['props']:
                # all_sempreds = []
                for subevent_group in event['events']:
                    for subevent in subevent_group['predicates']:
                        predicateType = '_'.join(subevent['predicateType'].lower().split(' '))                                            
                        polarity = subevent['polarity']
                        args = subevent['args']
                        keep = False
                        for arg in args:
                            if arg['value'] and (overlaps(entity, arg['value'].lower()) or overlaps(stem_entity, arg['value'].lower())):
                                keep = True
                                break
                        if keep:
                            prefix = ''
                            if not polarity:
                                prefix = '!'
                            if prefix + predicateType.lower() in ok_preds: #  or predicateType.lower() == 'has_state' and 'result' in [x['type'].lower() for x in args]
                                triple = prefix + predicateType.lower() + '('
                                args_list = []
                                for arg in args:
                                    if arg['value'].strip() and arg['value'].lower() not in thematic_roles and '_' not in arg['value']:
                                        if len(arg['value'].strip().lower().split(' ')) > 1:
                                            if ',' not in arg['value']:
                                                x = getPPhead(arg['value'].strip().lower())
                                                # x = arg['value'].strip().lower()
                                                if x.strip():
                                                    args_list += ['argvalue:'+x.strip()]
                                                else:
                                                    args_list += ['argvalue:'+'-']
                                            else:
                                                x = re.sub(',', '', arg['value'].strip().lower())
                                                x = re.sub(r'\s+', ' ', x)
                                                x = getPPhead(x)
                                                args_list += ['argvalue:'+x]
                                                # args_list += ['argvalue:'+arg['value'].strip().lower()]
                                        elif len(arg['value'].strip().lower().split(' ')) == 1:
                                            args_list += ['argvalue:'+arg['value'].strip().lower()]

                                    else:
                                        args_list += ['argvalue:'+'-']
                                if len(args_list) in [2, 3]:
                                    triple += ', '.join(args_list)
                                elif len(args_list) == 1:
                                    triple += args_list[0]
                                triple += ')'
                                if predicateType.lower() == 'motion':                                    
                                    subevent_list += [triple]
                                elif 'argvalue:'+'-' not in args_list:
                                    subevent_list += [triple]
    return subevent_list



def retrieve(datapath:str) -> Tuple[List[dict], int]:
    """
    Retrieve all triples (subevents) from SemParse to the given entity.
    Args:
        datapath - path to the input dataset
        fout - file object to store output
    """
    triple_cnt = 0
    result = []
    dataset = json.load(open(datapath, 'r', encoding='utf8'))
    split_name = re.sub(r'.json$','',datapath.split('/')[-1])

    for instance in tqdm(dataset):
        para_id = instance['id']
        entity_name = instance['entity']
        topic = instance['topic']
        paragraph = instance['paragraph']
        prompt = instance['prompt']
        parse_dir = os.path.join(opt.semparse, split_name, str(para_id))
        # print(parse_dir)
        triples = []
        sid_list = [s['id'] for s in instance['sentence_list'] if s['entity_mention']]
        parse_file_list = [fname for sid in sid_list for fname in os.listdir(parse_dir) if re.sub('sentence', '', fname.split('.')[0]) == str(sid)]
        
        for ent in entity_name.split(';'):
            # find relevant semantic predicates where this entity is mentioned
            x = search_triple(ent, parse_dir, sid_list)
            x = [t for t in x if t not in triples]
            triples.extend(x)
        triple_cnt += len(triples)

        result.append({'id': para_id,
                       'entity': entity_name,
                       'topic': topic,
                       'prompt': prompt,
                       'paragraph': paragraph,
                       'knowledge': triples
                       })

    return result, triple_cnt




if __name__ == '__main__':

    # if os.path.exists(opt.output):
    #     result = json.load(open(opt.output, 'r', encoding='utf8'))
    # else:
    #     result = []
    result = []

    print('Dev')
    dev_result, triple_cnt = retrieve(opt.dev)    
    result.extend(dev_result)
    print(triple_cnt)
    json.dump(result, open(opt.output, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    print(f'{len(dev_result)} data instances acquired.')
    print(f'Average number of VN triples found: {triple_cnt / len(dev_result)}')

    print('Test')
    test_result, triple_cnt = retrieve(opt.test)
    result.extend(test_result)
    json.dump(result, open(opt.output, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    print(f'{len(test_result)} data instances acquired.')
    print(f'Average number of VN triples found: {triple_cnt / len(test_result)}')

    print('Train')
    train_result, triple_cnt = retrieve(opt.train)
    result.extend(train_result)
    json.dump(result, open(opt.output, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    print(f'{len(train_result)} data instances acquired.')
    print(f'Average number of VN triples found: {triple_cnt / len(train_result)}')