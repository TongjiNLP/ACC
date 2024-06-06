import json
import os
from eval_script import *
import random
from tqdm import tqdm
from eda import eda
import copy

data_path="../data/MultiSpanQA_data/train.json"
max_train_true_answers=100000
max_dev_train_true_answers=10000

def find_all_positions(s:str, sub:str):
    positions = []
    start = 0
    while True:
        index = s.find(sub, start)
        if index == -1:
            break
        positions.append(index)
        start = index + 1
    return positions

def read_json(src):
    with open(src,"r",encoding="utf-8") as f:
        data=json.load(f)
    return data

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

def judge_answers(predict_answers,true_answers:list):
    def compute_overlap(pred:str,gold:str):
        pred_list=pred.split(" ")
        gold_list=gold.split(" ")
        count = 0
        for pred_span in pred_list:
            if pred_span in gold_list:
                count += 1
        return count

    all_answers={
        "true_answers":[],
        "partial_answers":[],
        "wrong_answers":[],
        "missing_answers":copy.deepcopy(true_answers)
    }
    similar_answer=None
    for pred in predict_answers:
        max_overlap=-1.0
        for answer in true_answers:
            if pred==answer:
                max_overlap=1.0
                break
            elif pred in answer or answer in pred:
                max_overlap=0.5
                similar_answer=answer
                break
            else:
                overlap_num=compute_overlap(pred,answer)
                if overlap_num >= 2:
                    max_overlap=0.5
                    similar_answer=answer
                    break

                    
        if max_overlap==1:
            all_answers["true_answers"].append(pred)
            all_answers["missing_answers"].remove(pred)
        elif 0.5<=max_overlap<1:
            all_answers["partial_answers"].append((pred,similar_answer))
            # all_answers["missing_answers"].remove(similar_answer)
        else:
            all_answers["wrong_answers"].append(pred)
    return all_answers

def prepare_squad_data_correct(data_dir:str,ans_dir:str,dst_dir:str,eda_prop=0.0):

    dataset=read_json(data_dir)["data"]
    sample_answers=read_json(ans_dir)

    def get_dataset(ori_dataset,sample_answers,prefix=""):
        new_datasets=[]
        true_answers_num=0
        partial_answers_num=0
        for example in tqdm(ori_dataset):
            example_id=example["id"]

            predict_answers=sample_answers[example_id]
            context=" ".join(example["context"])
            question=" ".join(example["question"])
            # 获取真实答案
            # true_answers = get_entities(example["label"],example["context"])
            # true_answers = [(each[0],each[1]) for each in true_answers]

            partial_answers = predict_answers["partial answer"]
            true_answers=predict_answers["true answer"]


            # if len(partial_answers)>max_partial_answer_num:
            #     partial_answers=random.sample(partial_answers,max_partial_answer_num)

            # max_true_answer_num=int(len(partial_answers)/2)

            # if len(true_answers)>max_true_answer_num:
            #     true_answers=random.sample(true_answers,max_true_answer_num)

            true_answers_num+=len(true_answers)
            partial_answers_num+=len(partial_answers)
            
            qas=[]

            for i,(ans,idx) in enumerate(true_answers):
                qas.append({
                    "id":example_id+f"_t{i}",
                    "question":f"Based on the prediction ` {ans} ` , "+question,
                    "answers":[
                        {
                            "text":ans,
                            "answer_start":idx
                        } 
                    ],
                    "is_impossible":False
                })
            for i,(ans,idx,similar_ans,similar_idx) in enumerate(partial_answers):
                qas.append({
                    "id":example_id+f"_p{i}",
                    "question":f"Based on the prediction ` {ans} ` , "+question,
                    "answers":[
                        {
                            "text":similar_ans,
                            "answer_start":similar_idx
                        } 
                    ],
                    "is_impossible":False
                })
            if len(qas)>0:
                new_datasets.append({
                    "titles":"",
                    "paragraphs":[{
                        "qas":qas,
                        "context":context
                    }]
                })
        
        print(f"===== infomation of {prefix} =====")
        print(f"number of true answers : {true_answers_num}")
        print(f"number of partial answers : {partial_answers_num}")

        return {
            "data":new_datasets
        }    

    train_data,dev_data=random_split(dataset,int(0.95*len(dataset)))
    train_data_new=get_dataset(train_data,sample_answers,prefix="train data")
    dev_data=get_dataset(dev_data,sample_answers,prefix="dev data")

    os.mkdir(dst_dir) if not os.path.exists(dst_dir) else None
    with open(os.path.join(dst_dir,"train.json"),"w",encoding="utf-8",newline="") as f:
        json.dump(train_data_new,f,indent=None,ensure_ascii=False)
    with open(os.path.join(dst_dir,"dev.json"),"w",encoding="utf-8",newline="") as f:
        json.dump(dev_data,f,indent=None,ensure_ascii=False)    

if __name__=="__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--train_data_fp",type=str,default="../data/MultiSpanQA_data/train.json")
    parser.add_argument("--answer_fp",type=str,default="../predictions/all_answer_cor.json")
    parser.add_argument("--dst_dir",type=str,default="../data/cor_data_new")

    args=parser.parse_args()

    prepare_squad_data_correct(
        data_dir=args.train_data_fp,
        ans_dir=args.answer_fp,
        dst_dir=args.dst_dir,
    )