import sys
from asyncio import sleep
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from inferlight import Handler, InferLight
from sanic.response import json
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AlbertConfig, AlbertModel, BertTokenizer


# define the model
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


def convert_one_line(text_a, text_b, tokenizer):
    inputs = tokenizer.encode_plus(text_a, text_b, return_tensors='pt')
    token_ids = inputs['input_ids'][0]
    token_types = inputs['token_type_ids'][0]
    token_masks = inputs['attention_mask'][0]
    return token_ids, token_types, token_masks


def convert_batch(queries, candidates, tokenizer):
    token_ids, token_types, token_masks = [], [], []
    for q, c in zip(queries, candidates):
        token_id, token_type, token_mask = convert_one_line(
            q, c, tokenizer)
        token_ids.append(token_id)
        token_types.append(token_type)
        token_masks.append(token_mask)
    token_ids = pad_sequence(token_ids, batch_first=True)
    token_types = pad_sequence(token_types, batch_first=True, padding_value=1)
    token_masks = pad_sequence(token_masks, batch_first=True)
    return token_ids, token_types, token_masks


# define Handler
class MyHandler(Handler):

    def load_model(self, config=None):
        """载入模型，返回载入后的模型组件

        Returns:
            [dict] -- [模型组件]
        """
        print("** loading model.. **")
        tokenizer = BertTokenizer.from_pretrained(
            '../albert-small/', cache_dir=None, do_lower_case=True)
        bert_config = AlbertConfig.from_pretrained('../albert-small/')
        model = PairModel(config=bert_config)
        device = torch.device('cpu')
        state = torch.load(Path('../albert-small/pytorch_model.pt'),
                           map_location=device)
        model.load_state_dict(state['model'])
        model.to(device)
        model.eval()
        self.model = model
        self.tokenizer = tokenizer

    def dispatch_http_request(self, request):
        """从request中获取数据

        :param request: Sanic的request对象
        :return: 送入batch_predict的对象
        :rtype: [type]
        """
        text_a = request.json.get('text_a')
        text_b = request.json.get('text_b')
        return (text_a, text_b)

    def batch_predict(self, batch_data):
        """批量推理

        :param batch_data: List，每个元素是dispatch_http_request的返回值
        :return: List，你的结果
        :rtype: List
        """
        tokenizer = self.tokenizer
        model = self.model
        queries, candidates = zip(*batch_data)
        token_ids, token_types, token_masks = convert_batch(
            queries, candidates, tokenizer)
        with torch.no_grad():
            v_pred = model(token_ids, token_types, token_masks)
            v_pred = torch.sigmoid(v_pred)
        return v_pred.cpu().numpy().flatten()

    def make_http_response(self, result):
        """将结果变为可返回的结果

        :param result: batch_predict返回List中的元素
        :type result: Unknown
        :return: 返回结果
        :rtype: String or Dict
        """
        # result is a float32
        # we should convert it into string
        return {"相似度": str(result)}


if __name__ == "__main__":
    handler = MyHandler()
    light = InferLight(handler)
    light.register_http_endpoint('/predict', methods=['POST'])
    light.start_service(http=True)
