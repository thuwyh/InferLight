from datetime import datetime
from pathlib import Path

import torch
from sanic import Sanic
from sanic.log import logger
from sanic.response import json, text
from transformers import AlbertConfig, AlbertModel, BertTokenizer

from albertqa import PairModel, model_predict

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

app = Sanic("QA inference")


@app.route("/")
async def test(request):
    return text("I am alive!")


@app.route("/predict", methods=["POST"])
async def predict(request):
    query = request.json.get('query')
    candidates = request.json.get('candidates')
    since = datetime.now()
    similarity = model_predict(query, candidates, model, tokenizer)
    retval = {"result": [{"question": candidates[i],
                          "score": str(similarity[i])} for i in range(len(candidates))]}
    duration = ((datetime.now()-since).microseconds)/1000
    logger.info('Prediction finished: %.3fms' % duration)
    return json(retval)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
