import sys
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AlbertConfig, AlbertModel, BertTokenizer


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


