import json
import random
import copy
import os
from utils import *
import argparse

def mask_dev_answers(dataset):
    for i,example in enumerate(dataset["data"]):
        new_context=[]
        new_labels=[]
        for token,label in zip(example["context"],example["label"]):
            if label=="B" or label=="I":
                continue
            new_context.append(token)
            new_labels.append(label)
        example["context"]=new_context
        example["label"]=new_labels
        dataset["data"][i]=example
    return dataset

def split_list_random(lst, k):
    lst_copy=copy.deepcopy(lst)
    random.shuffle(lst_copy)
    n = len(lst_copy)
    avg = n // k
    remainder = n % k
    result = []
    index = 0

    for i in range(k):
        if i < remainder:
            result.append(lst_copy[index:index+avg+1])
            index += avg + 1
        else:
            result.append(lst_copy[index:index+avg])
            index += avg

    return result

def split_train_data(src,dst,dev_name="valid.json",split_num=3):
    train_data=read_json(src)["data"]

    train_datas=split_list_random(train_data,split_num)

    os.mkdir(dst) if not os.path.exists(dst) else None

    for i in range(split_num):
        split_dir=os.path.join(dst,f"split_{i+1}")
        os.mkdir(split_dir) if not os.path.exists(split_dir) else None
        train_data_1={"data":[]}
        dev_data_1={"data":train_datas[i]}
        for j in range(split_num):
            if i == j:
                continue
            train_data_1["data"]+=train_datas[j]
        write_json(train_data_1,os.path.join(split_dir,"train.json"),indent=None)
        write_json(dev_data_1,os.path.join(split_dir,dev_name),indent=None)

if __name__=="__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--original_file",type=str,default="../data/MultiSpanQA_data/train.json")
    parser.add_argument("--split_dir",type=str,default="../data/MSQA_split")

    args=parser.parse_args()

    split_train_data(
        src=args.original_file,
        dst=args.split_dir,
    )
    