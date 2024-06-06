import os
import json
import collections
from typing import List,Tuple
from transformers import PreTrainedTokenizerFast
import random
import copy

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from datasets import load_dataset
from eval_script import *
from util_extract import *

DEBUG_NUM=20

def json_dump(dst,data):
    with open(dst,"w",encoding="utf-8",newline="") as f:
        json.dump(data,f,indent=4,ensure_ascii=False)     

def to_list(tensor):
    return tensor.detach().cpu().tolist()

class BaseProcessor(object):
    def get_input_text(self,example):
        raise NotImplementedError

    def get_train_examples(self,data_dir,debug=False):
        raise NotImplementedError

    def get_dev_examples(self,data_dir,debug=False):
        raise NotImplementedError

class MultiSpanQAProcessor(object):
    def get_dev_examples(self,data_dir,debug=False):  
        data_file={
            "valid":os.path.join(data_dir,"valid.json") if not debug else os.path.join(data_dir,"debug.json")
        }
        data=load_dataset('json', field='data', data_files=data_file)
        return data["valid"]

    def get_test_examples(self,data_dir,debug=False):
        data_file={
            "test":os.path.join(data_dir,"test.json") if not debug else os.path.join(data_dir,"debug.json")
        }
        data=load_dataset('json', field='data', data_files=data_file)
        return data["test"]
        # return data

processors={
    "MultiSpanQA":MultiSpanQAProcessor
}

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])

def get_answermodel_features(
    examples,
    tokenizer,
    args
):
    def prepare_validation_features(examples):
        tokenized_examples = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=args.max_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,
            padding="max_length",
            is_split_into_words=True,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []
        tokenized_examples["word_ids"] = []
        tokenized_examples["sequence_ids"] = []

        for i, sample_index in enumerate(sample_mapping):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            word_ids = tokenized_examples.word_ids(i)
            # One example can give several spans, this is the index of the example containing this span of text.
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            tokenized_examples["word_ids"].append(word_ids)
            tokenized_examples["sequence_ids"].append(sequence_ids)
        return tokenized_examples

    features = examples.map(
        prepare_validation_features,
        batched=True,
        remove_columns=examples.column_names,
        desc="Running answer_model's tokenizer on validation dataset",
    )

    all_input_ids=torch.tensor([each["input_ids"] for each in features],dtype=torch.long)
    all_token_type_ids=torch.tensor([each["token_type_ids"] for each in features],dtype=torch.long)
    all_att_masks=torch.tensor([each["attention_mask"] for each in features],dtype=torch.long)
    dataset=TensorDataset(
        all_input_ids,all_token_type_ids,all_att_masks
    )
    return dataset,features

def get_clsmodel_features(
    examples,
    tokenizer,
    last_predictions,
    args
):
    def prepare_validation_features(examples):
        features={
            "example_ids":[],
            "answers":[],
            "input_ids":[],
            "input_masks":[],
            "segment_ids":[],
            "p_ranges":[],
            "q_ranges":[],
            "a_ranges":[]
        }

        for example_id,question,context in zip(examples["id"],examples["question"],examples["context"]):
            p =" ".join(context)
            q =" ".join(question)
            predict_answers=last_predictions[example_id]
            for answer in predict_answers:
                qo = f"Is {answer} the answer for {q} ?"
                encoding=tokenizer(
                    qo,
                    p,
                    add_special_tokens=True,
                    max_length=args.max_length,
                    truncation=True,
                    truncation_strategy="longest_first",
                    return_token_type_ids=True,
                    return_offsets_mapping=True,
                    padding="max_length"
                )


                offset_maping = encoding['offset_mapping']


                ans_idx_st = p.index(answer)
                ans_idx_ed = ans_idx_st + len(answer)

                end1,end2 = -1 , -1
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
                    p_range=[ end1+2 , end2]
                    # q_range=[ end1+2 , end2]
                    # p_range=[ 1 , end1 ]
                    a_range=[ ans_st , ans_ed + 1 ]

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
                                break
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
                    p_range=[ end1+1 , end2]
                    a_range=[ ans_st , ans_ed + 1 ]
            
                if ans_st==-1 or ans_ed==-1:
                        a_range=[0,1]

                features["example_ids"].append(example_id)
                features["answers"].append(answer)
                features["input_ids"].append(encoding.input_ids)
                features["input_masks"].append(encoding.attention_mask)
                features["segment_ids"].append(encoding.token_type_ids)
                features["p_ranges"].append(p_range)
                features["q_ranges"].append(q_range)
                features["a_ranges"].append(a_range)
        
        return features

    features = examples.map(
        prepare_validation_features,
        batched=True,
        remove_columns=examples.column_names,
        desc="Running cls_model's tokenizer on validation dataset",
    )
    
    all_input_ids=torch.tensor(features["input_ids"], dtype=torch.long)
    all_input_masks=torch.tensor(features["input_masks"], dtype=torch.long)
    all_segment_ids=torch.tensor(features["segment_ids"], dtype=torch.long)
    all_p_ranges=torch.tensor(features["p_ranges"], dtype=torch.long)
    all_q_ranges=torch.tensor(features["q_ranges"], dtype=torch.long)
    all_a_ranges=torch.tensor(features["a_ranges"], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids,all_input_masks,all_segment_ids,
        all_p_ranges,all_q_ranges,all_a_ranges
    )

    if args.reserve_answer:
        reserve_list=[]
        for key,prediction in last_predictions.items():
            if len(prediction)<=1:
                reserve_list.append(key)
    else:
        reserve_list=None

    return dataset,features,reserve_list 

def get_genmodel_features(
    examples,
    tokenizer,
    prediction_labels,
    # all_wrong_answers,
    args        
):
    features={}
    input_list=[]
    input_list_id=[]
    all_original_answers=[]
    # other_answers_list=[]
    for example_id,question,context in zip(examples["id"],examples["question"],examples["context"]):
        preditions=prediction_labels[example_id]
        context=" ".join(context)
        for (pred,label) in preditions:
            if label==1:
                new_question=f"Based on the prediction ` {pred} ` , "+" ".join(question)

                input_list.append(f"Question: {new_question} Context: {context}")
                input_list_id.append(example_id)
                all_original_answers.append(pred)
            # other_answers_list.append(" # ".join(other_answers).lower())
        
    
    encoding = tokenizer(
        input_list,
        padding='longest',
        max_length=args.max_length,
        truncation='longest_first'
    )

    features["example_id"]=input_list_id
    features["input_ids"]=encoding.input_ids
    features["input_masks"]=encoding.attention_mask
    features["original_answer"]=all_original_answers
    # features["other_answers"]=other_answers_list

    all_input_ids=torch.tensor(features["input_ids"], dtype=torch.long)
    all_input_masks=torch.tensor(features["input_masks"], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids,all_input_masks
    )

    return dataset,features

def get_cormodel_features(
    examples,
    tokenizer,
    prediction_labels,
    args        
):
    all_partial_answers={}
    # 获取上一步预测结果中的所有部分正确答案
    for example_id in examples["id"]:
        preditions=prediction_labels[example_id]
        partial_answers=[]
        for (pred,label) in preditions:
            if label==1:
                partial_answers.append(pred)
            if args.use_binary and label==2:
                partial_answers.append(pred)
        all_partial_answers[example_id]=partial_answers

    # 获取SquadExamples
    squad_examples=get_squad_examples(
        examples,
        all_partial_answers
    )

    squad_features=convert_examples_to_features(
        examples=squad_examples,
        tokenizer=tokenizer,
        cls_token=tokenizer.cls_token,
        sep_token=tokenizer.sep_token,
        max_seq_length=args.max_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=False  
    )

    all_input_ids = torch.tensor([f.input_ids for f in squad_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in squad_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in squad_features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in squad_features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in squad_features], dtype=torch.float)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index, all_cls_index, all_p_mask)
    
    return dataset, squad_features, squad_examples

def postprocess_tagger_predictions(
    examples,
    features,
    outputs,
):
    """
    Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
    original contexts. This is the base postprocessing functions for models that only return start and end logits.

    Args:
        examples: The non-preprocessed dataset (see the main script for more information).
        features: The processed dataset (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
            first dimension must match the number of elements of :obj:`features`.
        output_dir (:obj:`str`, `optional`):
            If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if
            :obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null
            answers, are saved in `output_dir`.
        prefix (:obj:`str`, `optional`):
            If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
        log_level (:obj:`int`, `optional`, defaults to ``logging.WARNING``):
            ``logging`` log level (e.g., ``logging.WARNING``)
    """
    label_list = ["B", "I", "O"]
    id2label = {i: l for i, l in enumerate(label_list)}

    all_logits = np.array(outputs)
    all_labels = np.argmax(all_logits, axis=2)

    if -100 not in id2label.values():
        id2label[-100] = 'O'

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()

    # Let's loop over all the examples!
    # for example_index, example in enumerate(tqdm(examples)):
    for example_index, example in enumerate(examples):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]
        prelim_predictions = []
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            sequence_ids = features[feature_index]['sequence_ids']
            word_ids = features[feature_index]['word_ids']
            logits = [l for l in all_logits[feature_index]]
            labels = [id2label[l] for l in all_labels[feature_index]]
            prelim_predictions.append(
                {
                    "logits": logits,
                    "labels": labels,
                    "word_ids": word_ids,
                    "sequence_ids": sequence_ids
                }
            )

        previous_word_idx = -1
        ignored_index = []  # Some example tokens will disappear after tokenization.
        valid_labels = []
        valid_logits = []
        for x in prelim_predictions:
            logits = x['logits']
            labels = x['labels']
            word_ids = x['word_ids']
            sequence_ids = x['sequence_ids']

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            for word_idx, label, lo in list(zip(word_ids,labels,logits))[token_start_index:]:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    continue
                # We set the label for the first token of each word.
                elif word_idx > previous_word_idx:
                    ignored_index += range(previous_word_idx+1, word_idx)
                    valid_labels.append(label)
                    valid_logits.append(lo)
                    previous_word_idx = word_idx
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    continue

        context = example["context"]
        for i in ignored_index[::-1]:
            context = context[:i] + context[i+1:]

        assert len(context) == len(valid_labels)

        predict_entities = get_entities(valid_labels, context)
        predictions = [x[0] for x in predict_entities]
        all_predictions[example["id"]] = predictions

    return all_predictions

def postprocess_cls_predictions(
    examples,
    features,
    outputs,
    reserve_list=None,
    reserve_answer=False
):
    all_logits=np.array(outputs)
    all_labels=np.argmax(all_logits,axis=1)

    predictions=collections.defaultdict(list)
    prediction_labels=collections.defaultdict(list)
    for label,example_id,answer in zip(all_labels,features["example_ids"],features["answers"]):
        # true answer : 2
        # partial answer : 1
        # wrong answer : 0
        if reserve_answer and example_id in reserve_list:
            if label==2 and not answer in predictions[example_id]:
                predictions[example_id].append(answer)
                prediction_labels[example_id].append((answer,2))
            else:
                predictions[example_id].append(answer)
                prediction_labels[example_id].append((answer,1))
        else:
            if label==2 and not answer in predictions[example_id]:
                predictions[example_id].append(answer)
                prediction_labels[example_id].append((answer,2))
            elif label==1:
                predictions[example_id].append(answer)
                prediction_labels[example_id].append((answer,1))
            else:
                prediction_labels[example_id].append((answer,0))

    for id in examples["id"]:
        if not id in predictions:
            predictions[id]=[]
            prediction_labels[id]=[]

    # print(f"true answer number: {true_answer_num}")
    # print(f"partial answer number: {partial_answer_num}")
    # print(f"wrong answer number: {wrong_answer_num}")
    # input()

    return predictions,prediction_labels

def postprocess_gen_predictions(
    examples,
    features,
    outputs,   
    prediction_labels
):
    split_word=" # "
    prediction_labels_copy=copy.deepcopy(prediction_labels)
    all_predictions=collections.defaultdict(list)
    for key,value in prediction_labels_copy.items():
        for (pred,label) in value:
            if label==2:
                all_predictions[key].append(pred)
    
    for context,example_id,original_answer,output in zip(examples["context"],features["example_id"],features["original_answer"],outputs):
        answer_list=output.split(split_word)
        context=" ".join(context)
        new_answer_list=[]
        for answer in answer_list:
            answer=answer.strip().lstrip()
            answer_lower=answer.lower()     
            if answer_lower=="" or not answer_lower in context.lower():
                continue
            if answer in all_predictions[example_id]:
                continue
            new_answer_list.append(answer)
        if new_answer_list==[]:
            new_answer_list=[original_answer]
        all_predictions[example_id]+=new_answer_list
    
    for id in examples["id"]:
        if not id in all_predictions:
            all_predictions[id]=[]

    return all_predictions

def postprocess_cor_predictions(
    examples,
    features,
    outputs,   
    prediction_labels,
    args
):
    prediction_labels_copy=copy.deepcopy(prediction_labels)
    all_predictions=collections.defaultdict(list)
    for key,value in prediction_labels_copy.items():
        all_predictions[key]=[]
        for (pred,label) in value:
            if label==2:
                all_predictions[key].append(pred)

    

    all_predictions_new=write_predictions(
        examples,
        features,
        outputs,
        n_best_size=10,
        max_answer_length=args.max_ans_length,
        do_lower_case=True,
    )

    cor_model_outputs=[]

    for i,(key,ans) in enumerate(all_predictions_new.items()):
        # qas_id=example_id+"_"+str(i),
        original_text=examples[i].input_span
        example_id=key.split("_")[0]
        all_predictions[example_id].append(ans) if not ans in all_predictions[example_id] else None
        cor_model_outputs.append({
            "id":example_id,
            "original_text":original_text,
            "new_text":ans
        })

    for id in prediction_labels.keys():
        if not id in all_predictions:
            all_predictions[id]=[]

    return cor_model_outputs,all_predictions
