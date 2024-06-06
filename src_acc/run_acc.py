import os
import argparse
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union, Dict, Any
import copy

import torch
import numpy as np
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

import transformers
from transformers import (
    RobertaConfig,T5Config,
    RobertaTokenizerFast,T5TokenizerFast,BertTokenizerFast
)
from modeling_answer import RobertaTaggerForMultiSpanQA
from modeling_cls import (
    RobertaDUMASpanClassifier,
    RobertaForSequenceClassification,
    BertDUMASpanClassifier,
    RobertaSpanClassfier
)
from modeling_gen import T5GenerationModel
from modeling_cor import RobertaForQuestionAnswering,BertForQuestionAnswering

from util_acc import *
import csv
import time

MODELS={
    "bert":(BertTokenizerFast,BertDUMASpanClassifier,BertForQuestionAnswering),
    "roberta":(RobertaTokenizerFast,RobertaDUMASpanClassifier,RobertaForQuestionAnswering),
    "roberta_sc":(RobertaTokenizerFast,RobertaSpanClassfier,RobertaForQuestionAnswering),
}

CLS_LABELS=["wrong answer","partial answer","true answer"]

class ACCModel():
    def __init__(
        self,
        args
    ):
        tokenizer,cls_model,cor_model=MODELS[args.model_type]
        print(tokenizer,cls_model,cor_model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # tokenizer
        self.answer_tokenizer=tokenizer.from_pretrained(
            args.cls_model_path,
            use_fast=True,
            add_prefix_space=True,
        )
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(
            args.gen_tokenizer_path,
            use_fast=True,
            add_prefix_space=True,
        ) if args.use_gen else None
        # model from checkpoints
        # self.answer_model=RobertaTaggerForMultiSpanQA.from_pretrained(
        #     args.answer_model_path
        # ) if args.use_ans else None
        if args.use_cls_vanilla:
            self.cls_model=RobertaForSequenceClassification.from_pretrained(
                args.cls_model_path
            ) if args.use_cls else None
        else:
            self.cls_model=cls_model.from_pretrained(
                args.cls_model_path
            ) if args.use_cls else None
        # self.gen_model=T5GenerationModel.from_pretrained(
        #     args.gen_model_path
        # ) if args.use_gen else None
        if args.use_gen:
            self.gen_model=T5GenerationModel(
                args.gen_tokenizer_path
            )
            init_checkpoint = f'{args.gen_model_path}/pytorch_model.bin'
            checkpoint = torch.load(init_checkpoint, map_location='cpu')
            model_dict = checkpoint['model_state_dict']
            self.gen_model.load_state_dict(model_dict, False)

        else:
            self.gen_model=None

        self.cor_model=cor_model.from_pretrained(
            args.cor_model_path
        ) if args.use_cor or args.LLM_cls else None

        # Processor
        self.processor = processors[args.task_name]()
        
        self.dev_examples=self.processor.get_dev_examples(
            args.data_dir,
            debug=args.debug
        ) if args.do_eval else None
        self.test_examples=self.processor.get_test_examples(
            args.data_dir,
            debug=args.debug
        ) if args.do_predict else None
        self.gold_answers = read_gold(
            os.path.join(args.data_dir,"valid.json") if not args.debug else os.path.join(args.data_dir,"debug.json")
        )

    def _check_answer(self,examples,predictions:dict):
        for id,context in zip(examples["id"],examples["context"]):
            context=" ".join(context)
            predictions[id]=[each for each in predictions[id] if each in context]

    def _get_metrics(self,predictions):
        predictions_copy=copy.deepcopy(predictions)
        golds=copy.deepcopy(self.gold_answers)
        metrics=multi_span_evaluate(
            preds = predictions_copy,
            golds = golds
        )
        return metrics

    def write_cls_model_output(self,prediction_labels:dict,dst:str):
        all_spans=[]
        for example_id,spans in prediction_labels.items():
            for (span,label) in spans:
                all_spans.append({
                    "id":example_id,
                    "text":span,
                    "label":CLS_LABELS[label]
                })
        json_dump(dst,all_spans)

    def write_cor_model_output(self,cor_outputs:dict,dst:dir):
        json_dump(dst,cor_outputs)

    @torch.no_grad()
    def _answer(self,args,prefix="dev"):

        if args.pred_dir is not None and not args.use_ans:
            with open(args.pred_dir,"r",encoding="utf-8") as f:
                predictions = json.load(f)
            

            self._check_answer(
                self.dev_examples if prefix=="dev" else self.test_examples,
                predictions
            )
            return predictions
        
        dataset,features = get_answermodel_features(
            self.dev_examples if prefix=="dev" else self.test_examples,
            self.answer_tokenizer,
            args
        )
        eval_batch_size = args.answer_batch_size
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        print(f"***** Running Answering Model on {prefix} data*****")
        print(f"  Num examples = {len(dataset)}")
        print(f"  Batch size = {eval_batch_size}")

        self.answer_model.to(self.device)

        self.answer_model.eval()
        all_logits=[]

        for batch in eval_dataloader:
            batch = tuple(t.to(self.device) for t in batch)

            # 'input_ids', 'token_type_ids', 'attention_mask',
            inputs={
                "input_ids":batch[0],
                "token_type_ids":batch[1],
                "attention_mask":batch[2]
            }

            outputs=self.answer_model(**inputs)
            all_logits += to_list(outputs[0])
        
        predictions = postprocess_tagger_predictions(
            examples = self.dev_examples if prefix=="dev" else self.test_examples,
            features = features,
            outputs = all_logits
        )    

        return predictions
        
    @torch.no_grad()
    def _classify(self,args,last_predictions,prefix="dev"):
        start_time=time.time()
        dataset,features,reserve_list=get_clsmodel_features(
            self.dev_examples if prefix=="dev" else self.test_examples,
            self.answer_tokenizer,
            last_predictions,
            args
        )
        eval_batch_size = args.cls_batch_size
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        print(f"***** Running Classify Model on {prefix} data*****")
        print(f"  Num examples = {len(dataset)}")
        print(f"  Batch size = {eval_batch_size}")

        self.cls_model.to(self.device)

        self.cls_model.eval()
        all_logits=[]

        for batch in eval_dataloader:
            batch = tuple(t.to(self.device) for t in batch)

            # 'input_ids', 'token_type_ids', 'attention_mask',
            if isinstance(self.cls_model,RobertaDUMASpanClassifier) or isinstance(self.cls_model,BertDUMASpanClassifier):
                inputs={
                    "input_ids":batch[0],
                    "attention_mask":batch[1],
                    "token_type_ids":batch[2],
                    "p_ranges":batch[3],
                    "q_ranges":batch[4],
                }
            elif isinstance(self.cls_model,RobertaSpanClassfier):
                inputs={
                    "input_ids":batch[0],
                    "attention_mask":batch[1],
                    "token_type_ids":batch[2],
                    "q_ranges":batch[4],
                    "a_ranges":batch[5]
                }
            else:
                inputs={
                    "input_ids":batch[0],
                    "attention_mask":batch[1],
                    "token_type_ids":batch[2]
                }

            outputs=self.cls_model(**inputs)
            all_logits += to_list(torch.softmax(outputs.logits,dim=-1))
        
        predictions,prediction_labels = postprocess_cls_predictions(
            examples = self.dev_examples if prefix=="dev" else self.test_examples,
            features = features,
            outputs = all_logits,
            reserve_list=reserve_list,
            reserve_answer=args.reserve_answer
        )    

        end_time=time.time()
        eval_time=end_time-start_time

        return (
            predictions,
            prediction_labels,
            {
                "cls_eval_time(s)":round(eval_time,2),
                "cls_dataset_len":len(dataset),
                "cls_batch_size":eval_batch_size,
                "cls_eval_time_per_example(ms)":round(eval_time/len(dataset)*1000,2)
            }
        )

    @torch.no_grad()
    def _generate(self,args,last_predictions,prediction_labels,prefix="dev"):
        if prediction_labels is None:
            prediction_labels=collections.defaultdict(list)
            for key,value in last_predictions.items():
                for ans in value:
                    prediction_labels[key].append((ans,1))
        
        dataset,features=get_genmodel_features(
            self.dev_examples if prefix=="dev" else self.test_examples,
            self.t5_tokenizer,
            prediction_labels,
            args
        ) 

        eval_batch_size = args.cls_batch_size
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        print(f"***** Running Complement Model on {prefix} data*****")
        print(f"  Num examples = {len(dataset)}")
        print(f"  Batch size = {eval_batch_size}")

        self.gen_model.to(self.device)

        self.gen_model.eval()
        all_outputs=[]
        
        for batch in eval_dataloader:
            batch = tuple(t.to(self.device) for t in batch)

            inputs={
                "input_ids":batch[0],
                "input_masks":batch[1],
            }

            outputs=self.gen_model(**inputs)
            # predictions=postprocess_gen_predictions(
            #     self.dev_examples if prefix=="dev" else self.test_examples,
            #     features,
            #     outputs
            # )
            predictions = self.t5_tokenizer.batch_decode(
                outputs, 
                skip_special_tokens=True
            )
            all_outputs+=predictions
        
        all_predictions=postprocess_gen_predictions(
            self.dev_examples if prefix=="dev" else self.test_examples,
            features,
            all_outputs,
            prediction_labels,
        )

        return all_predictions

    @torch.no_grad()
    def _correct(self,args,last_predictions,prediction_labels,prefix="dev"):
        start_time=time.time()
        if prediction_labels is None:
            prediction_labels=collections.defaultdict(list)
            for key,value in last_predictions.items():
                for ans in value:
                    prediction_labels[key].append((ans,1))
        
        dataset,features,examples=get_cormodel_features(
            self.dev_examples if prefix=="dev" else self.test_examples,
            self.answer_tokenizer,
            prediction_labels,
            args
        ) 

        eval_batch_size = args.cor_batch_size
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        print(f"***** Running Correct Model on {prefix} data*****")
        print(f"  Num examples = {len(dataset)}")
        print(f"  Batch size = {eval_batch_size}")

        self.cor_model.to(self.device)

        self.cor_model.eval()
        
        # 存放结果
        all_results=[]

        # 开始测试
        # for batch in tqdm(eval_dataloader, desc="Evaluating"):
        for batch in eval_dataloader:
            # 准备模型输入
            self.cor_model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            token_type_ids=None
            if args.model_type in ['bert']:
                token_type_ids=batch[2]
            # 计算

            inputs = {
                'input_ids':      batch[0],
                'attention_mask': batch[1],
                'token_type_ids': token_type_ids
            }
            example_indices = batch[3]
            outputs = self.cor_model(**inputs)
            # 处理结果
            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                result = RawResult(unique_id    = unique_id,
                                start_logits = to_list(outputs[0][i]),
                                end_logits   = to_list(outputs[1][i]))
                all_results.append(result)
        
        cor_outputs,all_predictions=postprocess_cor_predictions(
            examples,
            features,
            all_results,
            prediction_labels,
            args
        )

        end_time=time.time()
        eval_time=end_time-start_time

        return (
            cor_outputs,
            all_predictions,
            {
                "cor_eval_time(s)":round(eval_time,2),
                "cor_dataset_len":len(dataset),
                "cor_batch_size":eval_batch_size,
                "cor_eval_time_per_example(ms)":round(eval_time/len(dataset)*1000,2)
            }
        )

    def eval_cls_first(self,args):       

        if (args.use_gen and args.use_cor):
            raise ValueError(
                "You should only choose 'use_gen' or 'use_cor', not both of them."
            )

        all_metrics={}
        eval_time_info={}

        os.mkdir(args.output_dir) if not os.path.exists(args.output_dir) else None
        predictions_file=os.path.join(args.output_dir,"predictions_ans.json")

        last_predictions=self._answer(args,prefix="dev")

        answer_metrics=self._get_metrics(last_predictions)
        all_metrics["ans only"]=answer_metrics

        json_dump(predictions_file,last_predictions)

        prediction_labels=None

        if args.use_cls:
            last_predictions,prediction_labels,cls_eval_info=self._classify(args,last_predictions)
            eval_time_info.update(cls_eval_info)
            cls_metrics=self._get_metrics(last_predictions)
            all_metrics[f"cls only"]=cls_metrics
            predictions_file=os.path.join(args.output_dir,"predictions_cls.json")
            self.write_cls_model_output(prediction_labels,os.path.join(args.output_dir,"cls_model_output.json"))
            json_dump(predictions_file,last_predictions)

        if args.use_gen:
            last_predictions=self._generate(args,last_predictions,prediction_labels)
            gen_metrics=self._get_metrics(last_predictions)
            all_metrics[f"cls+gen "]=gen_metrics
            predictions_file=os.path.join(args.output_dir,"predictions_gen.json")
            json_dump(predictions_file,last_predictions)
        
        if args.use_cor:
            cor_outputs,last_predictions,cor_eval_info=self._correct(args,last_predictions,prediction_labels)
            eval_time_info.update(cor_eval_info)
            cor_metrics=self._get_metrics(last_predictions)
            all_metrics[f"cls+cor "]=cor_metrics
            predictions_file=os.path.join(args.output_dir,"predictions_cor.json")
            self.write_cor_model_output(cor_outputs,os.path.join(args.output_dir,"cor_model_output.json"))
            json_dump(predictions_file,last_predictions)
        
        # save golds
        new_gold_answers={}
        for id,answers in self.gold_answers.items():
            new_gold_answers[id]=list(answers)
        # # save predictions
        gold_file=os.path.join(args.output_dir,"golds.json")
        json_dump(gold_file,new_gold_answers)

        json_dump(os.path.join(args.output_dir,"eval_info.json"),eval_time_info)

        return all_metrics

    def eval_cor_first(self,args):
        if (args.use_gen and args.use_cor):
            raise ValueError(
                "You should only choose 'use_gen' or 'use_cor', not both of them."
            )

        all_metrics={}
        eval_time_info={}

        os.mkdir(args.output_dir) if not os.path.exists(args.output_dir) else None
        predictions_file=os.path.join(args.output_dir,"predictions_ans.json")

        last_predictions=self._answer(args,prefix="dev")

        answer_metrics=self._get_metrics(last_predictions)
        all_metrics["ans only"]=answer_metrics

        json_dump(predictions_file,last_predictions)

        prediction_labels=None

        if args.use_gen:
            last_predictions=self._generate(args,last_predictions,prediction_labels)
            gen_metrics=self._get_metrics(last_predictions)
            all_metrics[f"gen only"]=gen_metrics
            predictions_file=os.path.join(args.output_dir,"predictions_gen.json")
            json_dump(predictions_file,last_predictions)
        
        if args.use_cor:
            cor_outputs,last_predictions,cor_eval_info=self._correct(args,last_predictions,prediction_labels)
            eval_time_info.update(cor_eval_info)
            cor_metrics=self._get_metrics(last_predictions)
            all_metrics[f"cor only"]=cor_metrics
            predictions_file=os.path.join(args.output_dir,"predictions_cor.json")
            self.write_cor_model_output(cor_outputs,os.path.join(args.output_dir,"cor_model_output.json"))
            json_dump(predictions_file,last_predictions)

        if args.use_cls:
            last_predictions,prediction_labels,cls_eval_info=self._classify(args,last_predictions)
            eval_time_info.update(cls_eval_info)
            cls_metrics=self._get_metrics(last_predictions)
            all_metrics[f"cor+cls"]=cls_metrics
            predictions_file=os.path.join(args.output_dir,"predictions_cls.json")
            self.write_cls_model_output(prediction_labels,os.path.join(args.output_dir,"cls_model_output.json"))
            json_dump(predictions_file,last_predictions)
        
        # save golds
        new_gold_answers={}
        for id,answers in self.gold_answers.items():
            new_gold_answers[id]=list(answers)
        # # save predictions
        gold_file=os.path.join(args.output_dir,"golds.json")
        json_dump(gold_file,new_gold_answers)

        json_dump(os.path.join(args.output_dir,"eval_info.json"),eval_time_info)

        return all_metrics

    def predict(self,args):
        pass

    def eval_after_LLM(self,args):
        all_metrics={}
        eval_time_info={}

        LABEL2NUM={LABEL:i for i,LABEL in enumerate(CLS_LABELS)}

        os.mkdir(args.output_dir) if not os.path.exists(args.output_dir) else None
        prediction_labels=collections.defaultdict(list)
        answer_dir=os.path.join(args.LLM_output_dir,f"cls/answers.json")
        predictions_dir=os.path.join(args.LLM_output_dir,f"cls/predictions.json")
        with open(answer_dir,"r",encoding="utf-8") as f:
            LLM_answers = json.load(f)
        with open(predictions_dir,"r",encoding="utf-8") as f:
            predictions = json.load(f)
        for pred in LLM_answers:
            prediction_labels[pred["id"]].append((pred["text"],LABEL2NUM[pred["label"]]))

        answer_metrics=self._get_metrics(predictions)
        all_metrics["LLM cls"]=answer_metrics

        cor_outputs,last_predictions,cor_eval_info=self._correct(args,predictions,prediction_labels)
        eval_time_info.update(cor_eval_info)
        cor_metrics=self._get_metrics(last_predictions)
        all_metrics[f"LLM cls+cor "]=cor_metrics
        predictions_file=os.path.join(args.output_dir,"predictions_gen.json")
        self.write_cor_model_output(cor_outputs,os.path.join(args.output_dir,"cor_model_output.json"))
        json_dump(predictions_file,last_predictions)        

        return all_metrics

def parse_args():
    parser = argparse.ArgumentParser()

    # data dir
    parser.add_argument("--data_dir",default=None,type=str)
    parser.add_argument("--output_dir",default=None,type=str)
    parser.add_argument("--output_name",default="result.csv",type=str)
    parser.add_argument("--task_name",default=None,type=str)
    parser.add_argument("--pred_dir",default=None,type=str)

    # models and tokenizers
    parser.add_argument("--answer_tokenizer_path",default=None,type=str)
    parser.add_argument("--gen_tokenizer_path",default=None,type=str)
    parser.add_argument("--answer_model_path",default=None,type=str)
    parser.add_argument("--cls_model_path",default=None,type=str)
    parser.add_argument("--gen_model_path",default=None,type=str)
    parser.add_argument("--cor_model_path",default=None,type=str)
    parser.add_argument("--model_type",default=None,type=str)

    # inference arguments
    parser.add_argument("--answer_batch_size",default=32,type=int)
    parser.add_argument("--cls_batch_size",default=32,type=int)
    parser.add_argument("--gen_batch_size",default=32,type=int)
    parser.add_argument("--cor_batch_size",default=32,type=int)

    # max_length
    parser.add_argument("--max_length",default=512,type=int)
    parser.add_argument("--max_query_length",default=64,type=int)
    parser.add_argument("--max_ans_length",default=75,type=int)
    parser.add_argument("--doc_stride",default=128,type=int)

    # usage
    parser.add_argument("--use_ans", action='store_true')
    parser.add_argument("--use_cls", action='store_true')
    parser.add_argument("--use_gen", action='store_true')
    parser.add_argument("--use_cor", action='store_true')
    parser.add_argument("--cor_first", action='store_true')
    parser.add_argument("--use_cls_vanilla", action='store_true')
    parser.add_argument("--use_binary", action='store_true')
    parser.add_argument("--reserve_answer",action='store_true')

    # running on dev / test
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_predict", action='store_true')

    # others
    parser.add_argument("--seed",default=26,type=int)
    parser.add_argument("--debug",action='store_true')
    
    # LLM cls/cor
    parser.add_argument("--LLM_cls",action='store_true')
    parser.add_argument("--LLM_output_dir",type=str)

    args = parser.parse_args()
    return args

def write_csv(data:dict,dst):
    with open(dst,"w",encoding="utf-8",newline="") as f:
        writer=csv.writer(f,delimiter="\t")
        writer.writerow(["models   ","EM P","EM R","EM F1","PM P","PM R","PM F1"])
        for key,val in data.items():
            row=[key]+[round(each,2) for each in val.values()]
            writer.writerow(row)

def main():
    args = parse_args()
    print(args)
    acc_model=ACCModel(args)

    if args.do_eval:
        if args.LLM_cls:
            metrics=acc_model.eval_after_LLM(args)
        elif args.cor_first:
            metrics=acc_model.eval_cor_first(args)
        else:
            metrics=acc_model.eval_cls_first(args)

        metrics_path=os.path.join(args.output_dir,args.output_name)
        print(metrics)
        write_csv(metrics,metrics_path)
    
if __name__=="__main__":
    main()
