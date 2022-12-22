'''
 @Date  : 11/03/2022 
 @Author: Ghazaleh Kazeminejad
 @mail  : ghka9436@colorado.edu
'''
import json
import argparse
from mimetypes import knownfiles
from tqdm import tqdm
import os
import time
from typing import List, Tuple

def insertionSortForTriples(triple_list):
  for index in range(1,len(triple_list)):
    currentvalue = triple_list[index]
    currentweight = float(currentvalue.split(', ')[7])
    if currentvalue.split(', ')[9] == 'RELEVANCE' and currentvalue.split(', ')[0] == 'relatedto':
      currentweight = currentweight / 3.0
    position = index
    while position > 0 and float(triple_list[position-1].split(', ')[7]) > currentweight:
      triple_list[position] = triple_list[position-1]
      position = position - 1
    triple_list[position] = currentvalue
  triple_list.reverse()
  return triple_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cn', type=str, default='./result/retrieval_0.json',
                        help='file containing retrieved triple from ConceptNet')
    # parser.add_argument('-bert', type=str, default='./result/retrieval_bert_word.json',
    #                     help='file containing retrieved triple using bert embedding')
    parser.add_argument('-output', type=str, default='./result/retrieval.json')
    parser.add_argument('-max_num', type=int, default=10, help='max number of retrieved triples')
    parser.add_argument('-semparse', type=str, default='./result/retrieval_vn.json')
    opt = parser.parse_args()

    cn_data = json.load(open(opt.cn, 'r', encoding='utf8'))
    vn_data = json.load(open(opt.semparse, 'r', encoding='utf8'))
    print(len(cn_data), len(vn_data))
    assert len(cn_data) == len(vn_data)
    result = []
    total_cn_cnt, total_vn_cnt = 0, 0
    for i in tqdm(range(len(cn_data))):
        cn_inst = cn_data[i]
        vn_inst = vn_data[i]
        para_id = cn_inst['id']
        assert para_id == vn_inst['id']
        entity = cn_inst['entity']
        assert entity == vn_inst['entity']
        topic = cn_inst['topic']
        paragraph = cn_inst['paragraph']
        prompt = cn_inst['prompt']
        knowledge = list(set(vn_inst['knowledge']))
        all_triples = cn_inst['cpnet'] + knowledge
        sorted_triples = insertionSortForTriples(all_triples)
        for t in sorted_triples[:opt.max_num]:
          if len(t.split(', ')) not in [11, 13]:
            print(t)
            assert len(t.split(', ')) in [11, 13]
        # cn_triples = cn_inst['cpnet']
        # vn_subevents = vn_inst['knowledge']
        
        # total_cn_cnt += len(cn_triples)
        # total_vn_cnt += len(vn_subevents)
        
        
        result.append(
            {'id': para_id,
            'entity': entity,
            'topic': topic,
            'prompt': prompt,
            'paragraph': paragraph,
            'knowledge': sorted_triples[:opt.max_num]
            # 'cpnet': cn_triples,
            # 'semparse': vn_subevents
            }
        )

    json.dump(result, open(opt.output, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    print(f'Total instances: {len(cn_data)}')
    # print(f'Average number of ConceptNet-retrieved triples: {total_cn_cnt/len(cn_data)}')
    # print(f'Average number of SemParse-retrieved triples: {total_vn_cnt/len(cn_data)}')
    # print(f'Instances with less than {opt.max_num} ConceptNet triples collected: {less_cnt} ({(less_cnt / len(cn_data)) * 100:.2f}%)')
    # print(f'Average number of BERT-retrieved triples: {total_bert_cnt / data_len}')
    # print(f'Average number of rule-retrieved triples: {total_rule_cnt / data_len}')
    print('Finished.')


if __name__ == '__main__':
    main()

