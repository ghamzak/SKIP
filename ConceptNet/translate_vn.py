'''
 @Date  : 01/11/2022
 @Author: Ghazaleh Kazeminejad
 @mail  : ghka9436@colorado.edu
'''
from ast import arg
import json
from tqdm import tqdm
import argparse
import re
from typing import List, Dict, Set, Tuple

# from dissertation.neural_symbolic.LEXIS3.KOALA.ConceptNet.rough_retrieval_vn import stem
# import nltk
# nltk.download('stopwords')
# nltk_stopwords = nltk.corpus.stopwords.words('english')
# nltk_stopwords += ["like", "gone", "did", "going", "would", "could", "get", "in", "up", "may"]
# from spacy.lang.en import STOP_WORDS
# STOP_WORDS = set(STOP_WORDS) - {'bottom', 'serious', 'top', 'alone', 'around', 'used', 'behind', 'side', 'mine', 'well'}
from Stemmer import PorterStemmer
stemmer = PorterStemmer()


def stem(word: str) -> str:
    """
    Stem a single word
    """
    word = word.lower().strip()
    return stemmer.stem(word)

def accepted_preds() -> List[str]:
    pf = '/home/ghazaleh/dissertation/neural_symbolic/LEXIS3/knowledgebased/predicate2sent.txt'
    with open(pf, 'r') as rf:
        p2s = [x.strip() for x in rf.readlines()]
    preds = [x.split(':')[0] for x in p2s]
    return list(set(preds))
pred_keys = accepted_preds()

def read_relation(filename: str) -> Dict[str, str]:
    file = open(filename, 'r', encoding='utf8')
    rel_rules = {}

    for line in file:
        rule = line.strip().split(': ')
        relation, direction = rule
        rel_rules[relation] = direction

    return rel_rules


def read_transform(filename: str) -> Dict[str, str]:
    file = open(filename, 'r', encoding='utf8')
    trans_rules = {}

    for line in file:
        relation, sentence = line.strip().split(': ')
        trans_rules[relation] = sentence.strip()

    return trans_rules

def triple2sent(raw_triples: List[str], trans_rules: Dict[str, str]) -> List[str]:
    """
    Turn the vn subevents into natural language sentences.
    """
    global pred_keys
    result = []    
    intransitive = ['alive', '!alive', 'appear', 'be','!be', 'degradation_material_integrity', 'destroyed', 'develop']
    transitive = ['covered', 'attached','!attached', 'contact', 'contain', 'create_image', 'emit', 'full_of', 'give_birth', 'has_location','!has_location', 'reside', 'together']
    for line in raw_triples:
        pred = line.split('(')[0]
        args = [x.strip() for x in re.sub(r'\)', '', line.split('(')[1]).split('argvalue:') if x.strip()]
        args = [re.sub(r'\,', '', x) for x in args]
        args = [re.sub(r'\s+', ' ', x) for x in args]
        sentence = ''
        if pred in intransitive:
            A = args[0]
        elif pred in transitive:
            A, B = args
        elif pred == 'motion':
            if len(args) == 1:
                A = args[0]
            elif len(args) == 2:
                A, B = args
        
        if pred == 'motion':
            if (len(args) == 2 and B == '?') or (len(args) == 1):
                sentence = A + ' moves'
        # elif pred == 'has_state':
        #     A = args[1]
        #     sentence = A + ' is created'
        #     args = [args[1]]
        elif pred == 'has_location' and len(args) == 2:
            if '!' + line in raw_triples:
                positive_index, negative_index = raw_triples.index(line), raw_triples.index('!'+line)
                if positive_index < negative_index:
                    sentence = A + ' leaves the initial location ' + B
                else:
                    sentence = A + ' moves towards destination location ' + B # enters?
            else:
                if len(args) == 2:
                    sentence = A + ' is located in location ' + B # enters?
        elif pred == 'be' and len(args) == 1:            
            if '!' + line in raw_triples:
                positive_index, negative_index = raw_triples.index(line), raw_triples.index('!'+line)
                if positive_index < negative_index:
                    sentence = A + ' is destroyed'
                else:
                    sentence = A + ' is created'

        elif pred == 'attached' and len(args) == 2:
            if '!' + line in raw_triples:
                positive_index, negative_index = raw_triples.index(line), raw_triples.index('!'+line)
                if positive_index < negative_index:
                    sentence = A + ' is detached from ' + B
                else:
                    sentence = A + ' is attached to ' + B
        elif pred == 'alive' and len(args) == 1:
            if '!' + line in raw_triples:
                positive_index, negative_index = raw_triples.index(line), raw_triples.index('!'+line)
                if positive_index < negative_index:
                    sentence = A + ' is dead'
                else:
                    sentence = A + ' is alive'
        elif pred.startswith('!'):
            continue
        else:
            if pred not in trans_rules.keys():
                pk = [x for x in pred_keys if x.startswith(pred)]
                if pk:
                    p_key = pk[0]
            else:
                p_key = pred
            template_sentence = trans_rules[p_key]
            if len(args) == 1:
                sentence = re.sub('A', A, template_sentence)
            elif len(args) == 2:
                sentence = re.sub('B', B, re.sub('A', A, template_sentence))
            # elif len(args) == 3:
            #     sentence = re.sub('C', C, re.sub('B', B, re.sub('A', A, template_sentence)))
        if sentence:
            if sentence.endswith('destroyed') and pred == 'be':
                pred = 'destroyed'
            if sentence.endswith('dead') and pred == 'alive':
                pred = 'dead'
            if sentence.endswith('apart') and pred == 'attached':
                pred = 'detached'
            if 'away from' in sentence and pred == 'has_location':
                pred = 'leave'
            
            if len(args) == 2:
                this_line_list = [pred, '_'.join(stem(A).split(' ')), '_'.join(A.split(' ')), 'n', '_'.join(stem(B).split(' ')), '_'.join(B.split(' ')), 'n', '4.0', 'LEFT', 'VN', sentence]
            elif len(args) == 1:
                this_line_list = [pred, '_'.join(stem(A).split(' ')), '_'.join(A.split(' ')), 'n', '-', '-', '-', '4.0', 'LEFT', 'VN', sentence]
            result.append(', '.join(this_line_list))

    return result



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # we currently don't have the rough_retrieval.json file. It should be generated using the rough_retrieval.py file
    parser.add_argument('-input', type=str, default='./rough_retrieval_vn.json', help='path to the rough vn triples')
    # generate retrieval.json here first so it doesn't overwrite the last version 
    parser.add_argument('-output', type=str, default='./result/retrieval_vn.json', help='path to store the generated graph')
    parser.add_argument('-relation', type=str, default='./relation_direction.txt', help='path to the relation rules')
    parser.add_argument('-transform', type=str, default='./predicate2sent.txt',
                        help='path to the file that describes the rules to transform vn subevents into natural language')
    parser.add_argument('-max', type=int, default=10, help='how many triples to collect')
    opt = parser.parse_args()

    data = json.load(open(opt.input, 'r', encoding='utf8'))
    rel_rules = read_relation(opt.relation)
    trans_rules = read_transform(opt.transform)
    result = []
    less_cnt, total_relevance, total_score = 0, 0, 0

    for instance in tqdm(data):
        selected_triples = triple2sent(raw_triples = instance['knowledge'], trans_rules = trans_rules)

        result.append({'id': instance['id'],
                       'entity': instance['entity'],
                       'topic': instance['topic'],
                       'prompt': instance['prompt'],
                       'paragraph': instance['paragraph'],
                       'knowledge': selected_triples
                       })

    json.dump(result, open(opt.output, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

    total_instances = len(result)
    print(f'Total instances: {total_instances}')
    # print(f'Instances with less than {opt.max} ConceptNet triples collected: {less_cnt} ({(less_cnt/total_instances)*100:.2f}%)')
    # print(f'Average number of relevance-based triples: {total_relevance / total_instances:.2f}')
    # print(f'Average number of score-based triples: {total_score / total_instances:.2f}')
    print(f'{len(result)} instances finished.')
