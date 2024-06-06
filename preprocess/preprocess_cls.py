import json
import os
from eval_script import *
import random
from tqdm import tqdm
import csv
from eda import eda
import copy
from utils import *
import argparse

# max_answer_num=10
data_path="../data/MultiSpanQA_data/train.json"

def prepare_cls_data_triple(data_dir:str,ans_dir:str,dst_dir:str,eda_prop=0):

    dataset=read_json(data_dir)["data"]
    sample_answers=read_json(ans_dir)

    def get_example(answers,id,question,context,label):
        examples=[]
        for i,ans in enumerate(answers):
            examples.append({
                "id":id+"_"+label[0]+str(i),
                "context":context,
                "question":question,
                "answer":ans,
                "label":label
            })
        return examples

    def use_eda(datasets,eda_prop):
        datasets_new=[]
        for example in random.sample(datasets,int(eda_prop*len(datasets))):
            new_contexts=eda(sentence=example["context"],num_aug=3)
            example["context"]=random.sample(new_contexts,1)[0]
            datasets_new.append(example)
        datasets+=datasets_new
        return datasets

    def get_dataset(ori_dataset,sample_answers,prefix="",eda_prop=0):

        true_datasets=[]
        partial_datasets=[]
        wrong_datasets=[]

        for example in tqdm(ori_dataset):
            example_id=example["id"]

            samples=sample_answers[example_id]

            true_answers=[ans for ans in samples["true answer"]]
            # true_answers=[ans[0] for ans in get_entities(example["label"],example["context"])]
            partial_answers=[ans[:2] for ans in samples["partial answer"]]
            wrong_answers=[ans for ans in samples["wrong answer"]]

            question=" ".join(example["question"])
            context=" ".join(example["context"])
            
            true_datasets+=get_example(true_answers,example_id,question,context,"true answer")
            partial_datasets+=get_example(partial_answers,example_id,question,context,"partial answer")
            wrong_datasets+=get_example(wrong_answers,example_id,question,context,"wrong answer")


        if eda_prop > 0:
            true_datasets=use_eda(true_datasets,eda_prop)
            partial_datasets=use_eda(partial_datasets,eda_prop)
            wrong_datasets=use_eda(wrong_datasets,eda_prop)

        # sample_number=min([len(true_datasets),len(partial_datasets),len(wrong_datasets)])
        # true_datasets=random.sample(true_datasets,sample_number)
        # partial_datasets=random.sample(partial_datasets,sample_number)
        # wrong_datasets=random.sample(wrong_datasets,sample_number)

        print(f"===== infomation of {prefix} =====")
        print(f"number of true answers : {len(true_datasets)}")
        print(f"number of partial answers : {len(partial_datasets)}")
        print(f"number of wrong answers : {len(wrong_datasets)}")

        datasets=true_datasets+partial_datasets+wrong_datasets
        random.shuffle(datasets)
        return datasets


    train_data,dev_data=random_split(dataset,int(0.92*len(dataset)))
    train_data_new=get_dataset(train_data,sample_answers,prefix="train data",eda_prop=eda_prop)
    dev_data_new=get_dataset(dev_data,sample_answers,prefix="dev data",eda_prop=0)



    os.mkdir(dst_dir) if not os.path.exists(dst_dir) else None
    with open(os.path.join(dst_dir,"train.json"),"w",encoding="utf-8",newline="") as f:
        json.dump(train_data_new,f,indent=4,ensure_ascii=False)
    with open(os.path.join(dst_dir,"valid.json"),"w",encoding="utf-8",newline="") as f:
        json.dump(dev_data_new,f,indent=4,ensure_ascii=False)    

    print("")
    print(f"train data number:{len(train_data)}")
    print(f"dev data examples:{len(dev_data)}")

def prepare_cls_data_binary(data_dir:str,ans_dir:str,dst_dir:str,eda_prop=0):

    dataset=read_json(data_dir)["data"]
    sample_answers=read_json(ans_dir)

    def get_dataset(ori_dataset,sample_answers,prefix=""):
        new_datasets=[]
        true_answers_num=0
        partial_answers_num=0
        wrong_answers_num=0
        for example in tqdm(ori_dataset):
            example_id=example["id"]

            samples=sample_answers[example_id]

            true_answers=[ans[:2] for ans in samples["true answer"]]
            # true_answers=[ans[0] for ans in get_entities(example["label"],example["context"])]
            wrong_answers=[ans[:2] for ans in samples["wrong answer"]]

            new_datasets.append({
                "id":example_id,
                "context":" ".join(example["context"]),
                "question":" ".join(example["question"]),
                "true_answers":true_answers,
                "wrong_answers":wrong_answers,
            })
            true_answers_num+=len(true_answers)
            wrong_answers_num+=len(wrong_answers)  

        print(f"===== infomation of {prefix} =====")
        print(f"number of true answers : {true_answers_num}")
        print(f"number of wrong answers : {wrong_answers_num}")
        return new_datasets     

    train_data,dev_data=random_split(dataset,int(0.92*len(dataset)))
    train_data_new=get_dataset(train_data,sample_answers,prefix="train data")
    dev_data_new=get_dataset(dev_data,sample_answers,prefix="dev data")

    if eda_prop > 0:
        print(f"add {eda_prop * 100} % eda data")
        for example in random.sample(train_data_new,int(eda_prop*len(train_data_new))):
            new_contexts=eda(sentence=example["context"],num_aug=5)
            example["context"]=random.sample(new_contexts,1)[0]
            train_data_new.append(example)
        train_data_new+=train_data_new
        random.shuffle(train_data_new)
        print(f"number of train example:{len(train_data_new)}")

    os.mkdir(dst_dir) if not os.path.exists(dst_dir) else None
    with open(os.path.join(dst_dir,"train.json"),"w",encoding="utf-8",newline="") as f:
        json.dump(train_data_new,f,indent=4,ensure_ascii=False)
    with open(os.path.join(dst_dir,"valid.json"),"w",encoding="utf-8",newline="") as f:
        json.dump(dev_data_new,f,indent=4,ensure_ascii=False)    

    print("")
    print(f"train data number:{len(train_data)}")
    print(f"dev data examples:{len(dev_data)}")

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    parser.add_argument("--train_data_fp",type=str,default="../data/MultiSpanQA_data/train.json")
    parser.add_argument("--answer_fp",type=str,default="../predictions/all_answer_cls.json")
    parser.add_argument("--dst_dir",type=str,default="../data/cls_data_new")

    args=parser.parse_args()

    prepare_cls_data_triple(
        data_dir=args.train_data_fp,
        ans_dir=args.answer_fp,
        dst_dir=args.dst_dir,
        eda_prop=0
    )