import math
from typing import Optional, Tuple, List, Union, Dict, Any

import torch
import numpy as np
import torch.nn as nn
from torch.nn import CrossEntropyLoss,BCELoss,MultiLabelSoftMarginLoss,MultiMarginLoss

import transformers
from transformers import (
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    set_seed,
)

from transformers import (
    RobertaPreTrainedModel,RobertaModel,BertModel,BertPreTrainedModel,BertConfig,BertTokenizerFast,
    RobertaConfig,RobertaTokenizerFast,RobertaForSequenceClassification
)
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
from transformers.modeling_outputs import MultipleChoiceModelOutput

class MulticlassHingeLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(MulticlassHingeLoss, self).__init__()
        self.margin = margin

    def forward(self, scores, targets):
        batch_size = scores.size(0)
        correct_scores = scores[torch.arange(batch_size), targets].unsqueeze(1)
        margins = scores - correct_scores + self.margin
        margins[torch.arange(batch_size), targets] = 0
        loss = torch.sum(torch.max(margins, torch.zeros_like(margins))) / batch_size
        return loss

def split_context_query(sequence_output, p_ranges, q_ranges):
    context_max_len = sequence_output.size(1)
    query_max_len = sequence_output.size(1)

    context_sequence_output = sequence_output.new(
        torch.Size((sequence_output.size(0), context_max_len, sequence_output.size(2)))).zero_()
    query_sequence_output = sequence_output.new_zeros(
        (sequence_output.size(0), query_max_len, sequence_output.size(2)))
    query_attention_mask = sequence_output.new_zeros((sequence_output.size(0), query_max_len))
    context_attention_mask = sequence_output.new_zeros((sequence_output.size(0), context_max_len))
    for i in range(0, sequence_output.size(0)):
        q_st, q_end = q_ranges[i]
        p_st, p_end = p_ranges[i]
        context_sequence_output[i, :min(context_max_len, p_end - p_st)] = sequence_output[i,
                                                                    p_st: min(context_max_len, p_end)]
        query_sequence_output[i, :min(query_max_len, q_end - q_st)] = sequence_output[i,
                                                                    q_st: min(query_max_len,q_end)]
        query_attention_mask[i, :min(query_max_len, q_end - q_st )] = sequence_output.new_ones(
            (1, query_max_len))[0, :min(query_max_len, q_end - q_st)]
        context_attention_mask[i, : min(context_max_len, p_end- p_st)] = sequence_output.new_ones((1, context_max_len))[0,
                                                                   : min(context_max_len, p_end - p_st)]
    return context_sequence_output, query_sequence_output, context_attention_mask, query_attention_mask

class MLP(nn.Module):
    def __init__(self,config,input_size,output_size):
        super(MLP,self).__init__()
        self.layer1=nn.Linear(input_size,config.hidden_size)
        self.layer2=nn.Linear(config.hidden_size,output_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
    
    def forward(self,x):
        x = self.dropout(x)
        x = self.layer1(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

class CoAttention(nn.Module):
    def __init__(self, config):
        super(CoAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, context_states, query_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        mixed_query_layer = self.query(query_states)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        attention_mask=extended_attention_mask

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(context_states)
            mixed_value_layer = self.value(context_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        outputs = context_layer
        return outputs

class RobertaClassfier(RobertaPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.classifier1=MLP(config,config.hidden_size,1)
        self.loss_fct = BCELoss()
        # self.classifier1=nn.Linear(config.hidden_size,2)

        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.roberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        # (batch_size,2)
        logits = self.classifier1(pooled_output)
        logits = logits.squeeze(-1)
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            loss = self.loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RobertaDUMASpanClassifier(RobertaPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.att = CoAttention(config)

        # self.classifier = (config)
        # self.loss_fct = MulticlassHingeLoss()
        # self.loss_fct= MultiLabelSoftMarginLoss()
        # self.loss_fct = MultiMarginLoss()
        self.loss_fct=CrossEntropyLoss()
        self.classifier=ClassificationHead(config)
        # self.classifier_2 = nn.Linear(2 * config.hidden_size, config.num_labels)
        self.post_init()
    
    def _convert_labels_to_onehot(self,label:torch.LongTensor,num_label):
        batch_size=label.size(0)
        new_label=torch.zeros(size=(batch_size,num_label),device=label.device)
        for i in range(batch_size):
            new_label[i,label[i]]=1
        return new_label

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        p_ranges:Optional[torch.LongTensor]=None,
        q_ranges:Optional[torch.LongTensor]=None,
        iter = 1,

        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.roberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output,pooled_output = outputs[0],outputs[1]

        # pq_end_pos = pq_end_pos.view(-1, pq_end_pos.size(-1))

        context_sequence_output, query_sequence_output, context_attention_mask, query_attention_mask = \
            split_context_query(sequence_output, p_ranges,q_ranges)

        for _ in range(iter):
            cq_biatt_output = self.att(context_sequence_output, query_sequence_output, context_attention_mask)
            qc_biatt_output = self.att(query_sequence_output, context_sequence_output, query_attention_mask)

            query_sequence_output=cq_biatt_output
            context_sequence_output=qc_biatt_output

        cat_output=torch.cat(
            [torch.mean(qc_biatt_output,1), torch.mean(cq_biatt_output,1) ], 
            1
        )

        pooled_output=self.dropout(cat_output)
        logits=self.classifier(pooled_output)
        # logits=self.classifier_2(pooled_output)
        
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if isinstance(self.loss_fct,MultiLabelSoftMarginLoss):
                labels=self._convert_labels_to_onehot(labels,logits.size(-1))
            # print(logits.size())
            # print(labels.size())
            loss = self.loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class BertDUMASpanClassifier(BertPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.att = CoAttention(config)

        # self.classifier = (config)
        # self.loss_fct = CrossEntropyLoss()
        self.loss_fct=MulticlassHingeLoss()
        self.classifier=ClassificationHead(config)
        # self.classifier_2 = nn.Linear(2 * config.hidden_size, config.num_labels)
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        p_ranges:Optional[torch.LongTensor]=None,
        q_ranges:Optional[torch.LongTensor]=None,
        iter = 1,

        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.bert(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output,pooled_output = outputs[0],outputs[1]

        # pq_end_pos = pq_end_pos.view(-1, pq_end_pos.size(-1))

        context_sequence_output, query_sequence_output, context_attention_mask, query_attention_mask = \
            split_context_query(sequence_output, p_ranges,q_ranges)

        for _ in range(iter):
            cq_biatt_output = self.att(context_sequence_output, query_sequence_output, context_attention_mask)
            qc_biatt_output = self.att(query_sequence_output, context_sequence_output, query_attention_mask)

            query_sequence_output=cq_biatt_output
            context_sequence_output=qc_biatt_output

        cat_output=torch.cat(
            [torch.mean(qc_biatt_output,1), torch.mean(cq_biatt_output,1) ], 
            1
        )

        pooled_output=self.dropout(cat_output)
        logits=self.classifier(pooled_output)
        # logits=self.classifier_2(pooled_output)
        
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            loss = self.loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RobertaSpanClassfier(RobertaPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        
        self.roberta = RobertaModel(config,add_pooling_layer=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.classifier1=MLP(config,2*config.hidden_size,3)
        # self.classifier1=nn.Linear(3*config.hidden_size,3)
        self.loss_fct = CrossEntropyLoss()

        self.post_init()
    
    def extract_span_rep(self, seq, ranges):
        # seq:[batch_size , max_length , hidden]
        # range:[batch_size, 2]

        # span_rep:[batch_size,hidden]
        span_reps=torch.tensor([],device=seq.device)
        batch_size=seq.size(0)
        # ranges=ranges.view(batch_size,-1)
        # print(ranges)
        for i in range(batch_size):
            st,ed=ranges[i]
            assert st<ed
            # 左闭右开
            span_rep=seq[i,st:ed]
            span_rep=torch.mean(span_rep,dim=0)
            span_rep=span_rep.unsqueeze(0)
            span_reps=torch.cat([span_reps,span_rep],dim=0)
        return span_reps

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        a_ranges: Optional[torch.LongTensor] = None,
        q_ranges: Optional[torch.LongTensor] = None,

        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.roberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output,pooled_output = outputs[0],outputs[1]


        query_rep=self.extract_span_rep(sequence_output,q_ranges)
        answer_rep=self.extract_span_rep(sequence_output,a_ranges)

        # print(pooled_output.shape)
        # print(query_rep.shape)
        # print(answer_rep.shape)

        # total_rep = torch.cat([pooled_output,query_rep,answer_rep],dim=1)

        total_rep = torch.cat([query_rep,answer_rep],dim=1)

        # (batch_size,2)
        logits = self.classifier1(total_rep)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism

            loss = self.loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

