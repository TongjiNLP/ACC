import json
import copy
from eval_script import get_entities
import random
import os
from bert_score import BERTScorer
from tqdm import tqdm
from transformers import RobertaTokenizerFast

overlap_prop_threshold = 0.25
bert_score_threshold = 0.6

max_answer_len=20

def read_json(src):
    with open(src,"r",encoding="utf-8") as f:
        data=json.load(f)
    return data

def write_json(data,dst,indent=4):
    with open(dst,"w",encoding="utf-8",newline="") as f:
        json.dump(data,f,indent=indent,ensure_ascii=False)

def compute_word_overlap(pred:str,gold:str):
    pred_words=pred.lower().split(" ")
    gold_words=gold.lower().split(" ")
    # 使用集合操作找出两个句子中的重复单词
    repeated_words = set(pred_words) & set(gold_words)

    # 返回重复单词的数量
    return len(repeated_words)/max(len(pred_words),len(gold_words))

def find_str_positions(main:str, sub:str):
    positions = []
    start = 0
    while True:
        index = main.find(sub, start)
        if index == -1:
            break
        positions.append(index)
        start = index + 1
    return positions

def has_position_overlap(pred:str,gold:str,pred_index:list,gold_index:list):
    p_len=len(pred)
    g_len=len(gold)
    for p_st in pred_index:
        for g_st in gold_index:
            p_end=p_st+p_len-1
            g_end=g_st+g_len-1
            if (p_st <= g_end and g_st <= p_end) :
                return g_st  # 重叠
    return -1

def judge_answer_type(preds:list,golds:list,context:str,Scorer:BERTScorer):
    all_answers={
        "true answer":[],
        "partial answer":[],
        "wrong answer":[],
        "bad answer":[]
    }
    '''
    分类标准：
    1. 正确答案要求: 与某个gold answer完全一致
    2. 半正确答案要求: 存在某个gold answer, 符合以下条件: 
       - gold answer 与正确答案的重叠程度超过 min_overlap_prop;
       - gold answer 与正确答案的语义相似度超过 bert_score_threshold;
       - 半正确答案与正确答案在上下文的位置相近;
       - 半正确答案中至多对应一个gold answer, 不能包含超过两个及以上的gold answer
    3. 错误答案要求: 除去正确答案和半正确答案的情况, 且不符合2.4的条件

    '''
    golds_indexs={gold:find_str_positions(context,gold) for gold in golds}

    partial_candidates=[]

    for pred in preds:

        # if len(pred.split(" "))>max_answer_len:
        #     all_answers["bad answer"].append(pred)
        #     continue

        pred_index=find_str_positions(context,pred)

        if pred_index==[]:
            all_answers["bad answer"].append(pred)
            continue

        best_overlap_prop=0.0
        similar_answer=""
        similar_answer_index=-1
        has_answer_num=0
        is_true_answer=False

        for gold in golds:
            gold_index=golds_indexs[gold]
            if gold_index==[]:
                continue
            if gold==pred:
                is_true_answer=True
                break
            # 统计是否包含gold answer
            if gold in pred or pred in gold:
                has_answer_num+=1
            
            # g_st=has_position_overlap(pred,gold,pred_index,gold_index)
            # if g_st==-1:
            #     continue

            g_st=gold_index[0]

            # 求出重叠程度
            overlap_prop=compute_word_overlap(pred,gold)
            if overlap_prop > best_overlap_prop:
                best_overlap_prop = overlap_prop
                similar_answer = gold
                similar_answer_index=g_st

        if is_true_answer:
            all_answers["true answer"].append([pred,pred_index[0]])
        # elif has_answer_num>=2:
        #     all_answers['bad answer'].append([pred,pred_index[0]])
        elif best_overlap_prop>=overlap_prop_threshold:
            partial_candidates.append([pred,pred_index[0],similar_answer,similar_answer_index])
        elif best_overlap_prop==0.0:
            all_answers['wrong answer'].append([pred,pred_index[0]])
        else:
            all_answers["bad answer"].append([pred,pred_index[0]])
    
    if len(partial_candidates)>0:
        bert_scores=Scorer.score(
            cands=[each[0] for each in partial_candidates],
            refs=[each[2] for each in partial_candidates],
        )[-1].tolist()

        for cands,score in zip(partial_candidates,bert_scores):
            if score >= bert_score_threshold:
                all_answers["partial answer"].append(cands)
            else:
                # all_answers["wrong answer"].append(cands[:2])
                all_answers["bad answer"].append(cands[:2])

    return all_answers

def random_split(span_list:list,k:int):
    # 首先，创建一个包含索引的列表，用于表示列表a的元素位置
    indices = list(range(len(span_list)))
    
    # 随机打乱索引列表
    random.shuffle(indices)
    
    # 将打乱后的索引列表分为长度为k和m-k的部分
    indices1 = indices[:k]
    indices2 = indices[k:]
    
    # 根据索引列表创建列表1和列表b
    list1 = [span_list[i] for i in indices1]
    list2 = [span_list[i] for i in indices2]
    
    return list1, list2