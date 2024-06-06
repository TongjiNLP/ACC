import os
import json
from typing import List
from transformers import PreTrainedTokenizerFast
import random
from modeling_cls import *

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from sklearn.metrics import accuracy_score, recall_score, f1_score, matthews_corrcoef

DEBUG_NUM=50

LABEL2NUM={
    "true answer":2,
    "partial answer":1,
    "wrong answer":0
}

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def compute_metrics(pred:list,gold:list):
    accuracy=accuracy_score(gold,pred)
    return {
        "acc":accuracy,
    }

class BaseProcessor(object):
    def get_input_text(self,example):
        raise NotImplementedError

    def get_train_examples(self,data_dir,debug=False):
        raise NotImplementedError

    def get_dev_examples(self,data_dir,debug=False):
        raise NotImplementedError

class SpanClassifyExample(object):
    def __init__(
        self,
        example_id:str=None,
        question:str=None,
        context:str=None,
        answer:str=None,
        idx:int=None,
        label:int=None
    ):
        self.example_id=example_id
        self.question=question
        self.context=context
        self.answer=answer
        self.idx=idx
        self.label=label

    def __repr__(self) -> str:
        return f"example_id:{self.example_id}, question:{self.question}, context:{self.context}" + \
        f", answer:{self.answer}, idx:{self.idx}, label:{self.label}"

class BoolqExample(object):
    def __init__(
        self,
        example_id:str=None,
        question:str=None,
        context:str=None,
        label:int=None
    ):
        self.example_id=example_id
        self.question=question
        self.context=context
        self.label=label

    def __repr__(self) -> str:
        return f"example_id:{self.example_id}, question:{self.question}, context:{self.context}" + \
        f", label:{self.label}"   

class SpanClassifyFeatures(object):
    def __init__(
        self,
        feature_id=None,
        input_ids=None,
        input_masks=None,
        segment_ids=None,
        q_range=None,
        p_range=None,
        a_range=None,
        label=None
    ):
        self.feature_id=feature_id
        self.input_ids=input_ids
        self.input_masks=input_masks
        self.segment_ids=segment_ids
        self.label=label
        self.q_range=q_range
        self.p_range=p_range
        self.a_range=a_range

class SpanClassifyProcessor(BaseProcessor):
    def get_input_text(self, example:SpanClassifyExample):
        return(
            f"Is {example.answer} the answer for {example.question} ?",
            example.context
        ) 
    
    def read_files(self,fp,shuffle=False):
        examples=[]
        with open(fp,"r",encoding="utf-8") as f:
            data=json.load(f)
            for example in data:
                    examples.append(
                        SpanClassifyExample(
                            example_id= example["id"],
                            question=   example["question"],
                            context=    example["context"],
                            answer=     example["answer"][0],
                            idx=        example["answer"][1],
                            label=      LABEL2NUM[example["label"]]
                        )
                    )
        if shuffle:
            random.shuffle(examples)
        return examples

    def get_train_examples(self,data_dir,debug=False):
        data=self.read_files(os.path.join(data_dir,"train.json"),shuffle=True)
        return data[:DEBUG_NUM] if debug==True else data

    def get_dev_examples(self,data_dir,debug=False):
        data=self.read_files(os.path.join(data_dir,"valid.json"))
        return data[:DEBUG_NUM] if debug==True else data
        # return data

class BoolqProcessor(BaseProcessor):
    def get_input_text(self, example:BoolqExample):
        return(
            example.question,
            example.context
        ) 
    
    def read_files(self,fp,shuffle=False):
        examples=[]
        with open(fp,"r",encoding="utf-8") as f:
            id=0
            for line in f.readlines():
                data=json.loads(line)
                examples.append(
                    BoolqExample(
                        example_id=id,
                        question=data["question"],
                        context=data["passage"],
                        label=data["answer"]==True
                    )
                )
                id+=1
        if shuffle:
            random.shuffle(examples)
        return examples

    def get_train_examples(self,data_dir,debug=False):
        data=self.read_files(os.path.join(data_dir,"train.jsonl"),shuffle=True)
        return data[:DEBUG_NUM] if debug==True else data

    def get_dev_examples(self,data_dir,debug=False):
        data=self.read_files(os.path.join(data_dir,"dev.jsonl"))
        return data[:DEBUG_NUM] if debug==True else data

processors={
    'SpanClassify':SpanClassifyProcessor,
    'Boolq':BoolqProcessor
}

def get_features(
    examples: list,
    processor:BaseProcessor,
    max_length:int,
    tokenizer: PreTrainedTokenizerFast,
    truncation_strategy='longest_first',
    do_predict=False
):
    features=[]
    for i,example in enumerate(examples):
        if i%1000==0 and len(examples)>1000:
            print(f"Writing example {i} of {len(examples)}")
        # qo=f"Is {example.answer} the answer for {example.question} ?"
        # p=example.context
        qo,p=processor.get_input_text(example)
        # <s> Q </s> <s> P </s>
        encoding=tokenizer(
            qo,
            p,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            truncation_strategy=truncation_strategy,
            return_token_type_ids=True,
            return_offsets_mapping=True,
            padding="max_length"
        )

        offset_maping = encoding['offset_mapping']
        ans_idx_st , ans_idx_ed = example.idx , example.idx+len(example.answer)

        end1, end2 = -1 , -1
        count = 0
        ans_st, ans_ed= -1 , -1
        if isinstance(tokenizer,RobertaTokenizerFast):
            for i,(token_beg,token_end) in enumerate(offset_maping):
                if token_beg == 0 and token_end == 0:
                    if count == 0 or count == 2:
                        count += 1
                    elif count == 1:
                        end1 = i
                        count += 1
                    elif count == 3:
                        end2 = i
                elif end1!=-1:
                    if token_beg <= ans_idx_st and ans_idx_st < token_end and ans_st == -1:
                        ans_st = i
                    if token_beg <= ans_idx_ed and ans_idx_ed <= token_end and ans_ed == -1:
                        ans_ed = i
                
                # if i==0: 
                #     continue
                # if token_beg == 0 and token_end == 0 and end1 == -1:
                #     end1 = i
                # elif token_beg == 0 and token_end == 0 and end1 > -1:
                #     end2 = i
                #     break

            if end2 == -1:
                end2 = len(offset_maping) - 1
            # qo: [1     ,end1-1]
            # p:  [end1+2,end2-1]
            q_range=[ 1 , end1 ]
            p_range=[ end1 + 2 , end2 ]
            a_range=[ ans_st , ans_ed + 1 ]
            if ans_st==-1 or ans_ed==-1:
                a_range=[0,1]
            # print(tokenizer.convert_ids_to_tokens(encoding.input_ids[ans_st : ans_ed + 1]))
            # print(example.answer)
            # input()

        else:
            for i,(token_beg,token_end) in enumerate(offset_maping):
                if token_beg == 0 and token_end == 0:
                    if count == 0:
                        count += 1
                    elif count == 1:
                        end1 = i
                        count += 1
                    elif count == 2:
                        end2 = i
                else:
                    if token_beg <= ans_idx_st and ans_idx_st < token_end and ans_st == -1:
                        ans_st = i
                    if token_beg <= ans_idx_ed and ans_idx_ed <= token_end and ans_ed == -1:
                        ans_ed = i

            if end2 == -1:
                end2 = len(offset_maping) - 1
            # qo: [1     ,end1-1]
            # p:  [end1+2,end2-1]
            q_range=[ 1 , end1 ]
            p_range=[ end1 + 1 , end2 ]
            a_range=[ ans_st , ans_ed + 1 ]

        # if not do_predict:
        features.append(
            SpanClassifyFeatures(
                feature_id=example.example_id,
                input_ids=encoding.input_ids,
                input_masks=encoding.attention_mask,
                segment_ids=encoding.token_type_ids,
                label=example.label,
                p_range=p_range,
                q_range=q_range,
                a_range=a_range
            )
        )

    return features

def load_datasets(model_args,data_args,task,tokenizer,prefix="train", debug=False):
    assert prefix in ["train","dev"]
    processor=processors[task]()
    cached_features_file = os.path.join(data_args.data_dir, 'cached_{}_{}_{}_{}'.format(
        prefix,
        list(filter(None, model_args.model_name_or_path.split('/'))).pop(),
        str(data_args.max_seq_length),
        str(task)))
    if prefix=="dev":
        examples = processor.get_dev_examples(data_args.data_dir,debug=debug)
    # elif prefix=="test":
    #     examples = processor.get_test_examples(data_args.data_dir)
    else:
        examples = processor.get_train_examples(data_args.data_dir,debug=debug)
    
    if os.path.exists(cached_features_file) and not data_args.overwrite_cache:
        features = torch.load(cached_features_file)
    else:
        features = get_features(
            examples,
            processor,
            data_args.max_seq_length,
            tokenizer,
            do_predict=(prefix!="train"),
            truncation_strategy='longest_first'
        )
        torch.save(features, cached_features_file)
    

    all_input_ids=torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_masks=torch.tensor([f.input_masks for f in features], dtype=torch.long)
    all_segment_ids=torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_labels=torch.tensor([f.label for f in features], dtype=torch.long)
    all_p_ranges=torch.tensor([f.p_range for f in features], dtype=torch.long)
    all_q_ranges=torch.tensor([f.q_range for f in features], dtype=torch.long)
    all_a_ranges=torch.tensor([f.a_range for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids,all_input_masks,all_segment_ids,
        all_labels,all_p_ranges,all_q_ranges,all_a_ranges
    )

    return dataset,examples