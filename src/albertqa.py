import sys
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AlbertConfig, AlbertModel, BertTokenizer


def convert_one_line(text_a, text_b, tokenizer):
    inputs = tokenizer.encode_plus(text_a, text_b, return_tensors='pt')
    token_ids = inputs['input_ids'][0]
    token_types = inputs['token_type_ids'][0]
    token_masks = inputs['attention_mask'][0]
    return token_ids, token_types, token_masks


def convert_batch(query, candidates, tokenizer):
    token_ids, token_types, token_masks = [], [], []
    for c in candidates:
        token_id, token_type, token_mask = convert_one_line(
            query, c, tokenizer)
        token_ids.append(token_id)
        token_types.append(token_type)
        token_masks.append(token_mask)
    token_ids = pad_sequence(token_ids, batch_first=True)
    token_types = pad_sequence(token_types, batch_first=True, padding_value=1)
    token_masks = pad_sequence(token_masks, batch_first=True)
    return token_ids, token_types, token_masks


def model_predict(query, candidates, model, tokenizer):
    token_ids, token_types, token_masks = convert_batch(
        query, candidates, tokenizer)

    with torch.no_grad():
        v_pred = model(token_ids, token_types, token_masks)
        v_pred = torch.sigmoid(v_pred)
    return v_pred.cpu().numpy().flatten()


class PairModel(nn.Module):

    def __init__(self, pretrain_path=None, dropout=0.1, config=None):
        super(PairModel, self).__init__()
        if config is not None:
            self.bert = AlbertModel(config)
        else:
            self.bert = AlbertModel.from_pretrained(
                pretrain_path, cache_dir=None, num_labels=1)
        self.head = nn.Sequential(
            OrderedDict([
                ('dropout', nn.Dropout(dropout)),
                ('clf', nn.Linear(self.bert.config.hidden_size, 1)),
            ])
        )

    def forward(self, inputs, token_type, masks, token_type_ids=None):
        _, pooled_output = self.bert(inputs, masks, token_type)
        out = self.head(pooled_output)
        return out
