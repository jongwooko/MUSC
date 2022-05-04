from .modeling_bert import (
    BertModel, 
    BertPreTrainedModel,
    BertForMultipleChoice,
)

from .modeling_xlm_roberta import (
    XLMRobertaModel,
    XLMRobertaForSequenceClassification,
    XLMRobertaForMultipleChoice,
    XLMRobertaForTokenClassification,
)
    
from transformers.configuration_roberta import RobertaConfig
from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_xlm_roberta import XLMRobertaTokenizer
# from transformers.tokenization_roberta import RobertaTokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        
        config.output_hidden_states = True # For extract all intermeiate hidden states

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        
        self.init_weights()
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        mix_ratio=None
    ):
        
        if mix_ratio is None:
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )
        
        else:
            embedding_output = self.bert.get_embedding_output(
                input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )
            
            bsz = embedding_output.size(0)
            rev = bsz - torch.arange(bsz) - 1
            embedding_output = mix_ratio * embedding_output + (1 - mix_ratio) * embedding_output[rev]
            
            device = attention_mask.device
            attention_mask1 = attention_mask
            attention_mask2 = attention_mask[rev]
            attention_mask = torch.where(attention_mask1 + attention_mask2 == 0, 
                                         torch.zeros(attention_mask1.size()).to(device), torch.ones(attention_mask2.size()).to(device))
            outputs = self.bert.get_bert_output(embedding_output, attention_mask=attention_mask)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + (sequence_output[:, 0],) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
    
class Projector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

class LinearPredictor(BertPreTrainedModel):
    def __init__(self, bert_config, out_dim, dropout):
        super(LinearPredictor, self).__init__(bert_config)
        self.bert = BertModel(bert_config)
        self.classifier = nn.Linear(768, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def forward(self):
        raise NotImplementedError
        
class BertForSequenceTagging(LinearPredictor):
    """
    used for both tagging and ner.
    """

    def __init__(self, bert_config, out_dim, dropout=0.1):
        super(BertForSequenceTagging, self).__init__(bert_config, out_dim, dropout)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        if_tgts=None,
        **kwargs,
    ):
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        bert_out = bert_out[0]
        bert_out = self.dropout(bert_out)
        bert_out = self.classifier(bert_out)
        logits = bert_out[if_tgts]
        return (
            logits,
            torch.argmax(bert_out, dim=-1, keepdim=False),
            bert_out,
        )
    
    def get_embedding_output(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
    ):
        return self.bert.get_embedding_output(input_ids=input_ids, token_type_ids=token_type_ids,
                                              position_ids=position_ids)
    
    def get_logits_from_embedding_output(
        self,
        embedding_output,
        attention_mask=None,
        labels=None,
        if_tgts=None,
    ):
        bert_out = self.bert.get_bert_output(
                embedding_output,
                attention_mask=attention_mask
        )
        
        bert_out = bert_out[0]
        bert_out = self.dropout(bert_out)
        bert_out = self.classifier(bert_out)
        logits = bert_out[if_tgts]
        return (
            logits,
            torch.argmax(bert_out, dim=-1, keepdim=False),
            bert_out,
        )
    
    def get_last_hidden_from_embedding_output(
        self,
        embedding_output,
        attention_mask=None,
        labels=None,
    ):
        bert_out = self.bert.get_bert_output(
            embedding_output,
            attention_mask=attention_mask
        )
        bert_out = bert_out[0]
        return bert_out
    
    def get_last_hidden(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        if_tgts=None,
    ):
        bert_out = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        
        bert_out = bert_out[0]
        return bert_out
    
    def get_logits_from_last_hidden(
        self,
        last_hidden,
        labels=None,
        if_tgts=None,
    ):
        bert_out = self.dropout(last_hidden)
        bert_out = self.classifier(bert_out)
        logits = bert_out[if_tgts]
        return (
            logits,
            torch.argmax(bert_out, dim=-1, keepdim=False),
            bert_out,
        )
    
class SupConBERT(nn.Module):
    """backbone + projection head"""
    def __init__(self, encoder, head='mlp', feat_dim=128):
        super(SupConBERT, self).__init__()
        self.encoder = encoder
        if head == 'linear':
            self.head = nn.Linear(encoder.config.hidden_size, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(encoder.config.hidden_size, encoder.config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(encoder.config.hidden_size, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, **x):
        logit, feat, rep, *_ = self.encoder(**x)
        feat = F.normalize(self.head(feat), dim=1)
        return logit, feat