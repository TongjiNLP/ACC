import json
import copy
from eval_script import get_entities
import random
import os
from bert_score import BERTScorer
from tqdm import tqdm
from utils import *
import argparse

random.seed(1111)

max_cls_answer_num=120000
max_cor_answer_num=200000

def get_label_answers(train_data_fp,pred_fp,cls_dst,cor_dst):
    Scorer=BERTScorer("../../MODEL/bert")
    dev_data=read_json(train_data_fp)["data"]
    pred_data=None
    temp_pred_data=read_json(pred_fp)
    if pred_data is None:
        pred_data={key:[] for key in temp_pred_data.keys()}
    for key,all_answers in temp_pred_data.items():
        for ans in all_answers:
            ans=ans.strip().lstrip()
            if not ans in pred_data[key]:
                pred_data[key].append(ans)

    pred_data_new={
        id:{
            "true answer":[],
            "partial answer":[],
            "wrong answer":[]
        } for id in temp_pred_data.keys()
    }

    all_true_answers=[]
    all_partial_answers=[]
    all_wrong_answers=[]

    for example in tqdm(dev_data):
        example_id=example["id"]
        # (span, st, ed)
        context=" ".join(example["context"])

        gold_answers = get_entities(example["label"],example["context"])
        golds = [gold[0] for gold in gold_answers]
        golds_index=[find_str_positions(context,each)[0] for each in golds]
        preds = pred_data[example_id]

        answer_type=judge_answer_type(preds,golds,context,Scorer)

        all_true_answers+=[[example_id,gold,gold_idx] for (gold,gold_idx) in zip(golds,golds_index)]
        all_partial_answers+=[[example_id]+each for each in answer_type["partial answer"]]
        all_wrong_answers+=[[example_id]+each for each in answer_type["wrong answer"]]

    sample_num=min([len(all_true_answers),len(all_partial_answers),len(all_wrong_answers)])
    # sample_num=len(all_true_answers)

    # all_partial_answers=sorted(all_partial_answers,key=lambda x:abs(x[-1]-mean_f1))

    print(f"original answers")
    print(f"number of true answers : {len(all_true_answers)}")
    print(f"number of partial answers : {len(all_partial_answers)}")
    print(f"number of wrong answers : {len(all_wrong_answers)}")

    all_true_answers_cls=random.sample(all_true_answers,sample_num) if len(all_true_answers)>sample_num else all_true_answers
    all_partial_answers_cls=random.sample(all_partial_answers,sample_num) if len(all_partial_answers)>sample_num else all_partial_answers
    all_wrong_answers_cls=random.sample(all_wrong_answers,sample_num) if len(all_wrong_answers)>sample_num else all_wrong_answers
    
    for true_answer in all_true_answers_cls:
        example_id=true_answer[0]
        pred_data_new[example_id]["true answer"].append(true_answer[1:])
    for partial_answer in all_partial_answers_cls:
        example_id=partial_answer[0]
        pred_data_new[example_id]["partial answer"].append(partial_answer[1:])
    for wrong_answer in all_wrong_answers_cls:
        example_id=wrong_answer[0]
        pred_data_new[example_id]["wrong answer"].append(wrong_answer[1:])

    print(f"file: {pred_fp}")
    print(f"number of true answers : {len(all_true_answers_cls)}")
    print(f"number of partial answers : {len(all_partial_answers_cls)}")
    print(f"number of wrong answers : {len(all_wrong_answers_cls)}")
    print("")
    write_json(pred_data_new,cls_dst)

    all_true_answers_cor=all_true_answers

    sample_num=2*len(all_true_answers_cls)
    all_partial_answers_cor=random.sample(all_partial_answers,sample_num) if len(all_partial_answers)>sample_num else all_partial_answers

    pred_data_new={
        id:{
            "true answer":[],
            "partial answer":[]
        } for id in temp_pred_data.keys()
    }

    for true_answer in all_true_answers_cor:
        example_id=true_answer[0]
        pred_data_new[example_id]["true answer"].append(true_answer[1:])
    for partial_answer in all_partial_answers_cor:
        example_id=partial_answer[0]
        pred_data_new[example_id]["partial answer"].append(partial_answer[1:])

    print(f"file: {pred_fp}")
    print(f"number of true answers : {len(all_true_answers_cor)}")
    print(f"number of partial answers : {len(all_partial_answers_cor)}")
    print("")
    write_json(pred_data_new,cor_dst)

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    parser.add_argument("--prediction_fp",type=str,default="../data/MSQA_merge_prediction.json")
    parser.add_argument("--train_data_fp",type=str,default="../data/MultiSpanQA_data/train.json")
    parser.add_argument("--tokenizer_fp",type=str,default="../MODEL/roberta")
    parser.add_argument("--cls_dst",type=str,default="../predictions/all_answer_cls.json")
    parser.add_argument("--cor_dst",type=str,default="../predictions/all_answer_cor.json")

    args=parser.parse_args()

    tokenizer=RobertaTokenizerFast.from_pretrained(args.tokenizer_fp)

    get_label_answers( 
        train_data_fp=args.train_data_fp,
        pred_fp=args.prediction_fp,
        cls_dst=args.cls_dst,
        cor_dst=args.cor_dst
    )