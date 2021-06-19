from inferlight import LightWrapper, BaseInferLightWorker
import time
from sanic import Sanic
from sanic.response import json as json_response
import random
import logging

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import torch
from torch import nn

logging.basicConfig(level=logging.INFO)

class BertModel(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.bert = AutoModelForSequenceClassification.from_pretrained(config['model'])
        self.bert.eval()
        self.device = torch.device('cuda' if config.get('use_cuda') else 'cpu')
        self.bert.to(self.device)

    def forward(self, inputs):
        return self.bert(**inputs).logits

class MyWorker(BaseInferLightWorker):

    def load_model(self, model_args):
        self.model = BertModel(model_args)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_args['model'])
        self.device = torch.device('cuda' if model_args.get('use_cuda') else 'cpu')
        return

    def build_batch(self, requests):
        encoded_input = self.tokenizer.batch_encode_plus(requests, 
                                                         return_tensors='pt',
                                                         padding=True,
                                                         truncation=True,
                                                         max_length=512)
        return encoded_input.to(self.device)

    @torch.no_grad()
    def inference(self, batch):
        model_output = self.model.forward(batch).cpu().numpy()
        scores = softmax(model_output, axis=1)
        ret = [x.tolist() for x in scores]
        return ret
        

if __name__=='__main__':
    config = {
        'model':"nlptown/bert-base-multilingual-uncased-sentiment",
        'use_cuda':True
    }

    text = """
    A Fox one day spied a beautiful bunch of ripe grapes hanging from a vine trained along the branches of a tree. The grapes seemed ready to burst with juice, and the Fox's mouth watered as he gazed longingly at them.
    The bunch hung from a high branch, and the Fox had to jump for it. The first time he jumped he missed it by a long way. So he walked off a short distance and took a running leap at it, only to fall short once more. Again and again he tried, but in vain.
    Now he sat down and looked at the grapes in disgust.
    "What a fool I am," he said. "Here I am wearing myself out to get a bunch of sour grapes that are not worth gaping for."
    And off he walked very, very scornfully.
    """

    bert_model = BertModel(config)
    bert_model.eval()
    device = torch.device('cuda' if config.get('use_cuda') else 'cpu')
    bert_model.to(device)

    wrapped_model = LightWrapper(MyWorker, config, batch_size=16, max_delay=0.05)
    tokenizer = AutoTokenizer.from_pretrained(config['model'])
    app = Sanic('test')

    @app.get('/predict')
    async def predict(request):
        text_input = [text]
        encode_input = tokenizer.batch_encode_plus(text_input, return_tensors='pt')
        encode_input = encode_input.to(device)
        with torch.no_grad():
            model_output = bert_model.forward(encode_input).cpu().numpy()
        scores = softmax(model_output, axis=1)
        return json_response({'output':scores[0].tolist()})

    @app.get('/batch_predict')
    async def batched_predict(request):
        dummy_input = text
        response = await wrapped_model.predict(dummy_input)
        if not response.succeed():
            return json_response({'output':None, 'status':'failed'})
        return json_response({'output': response.result})

    app.run(port=8888)
