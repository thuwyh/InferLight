from inferlight import LightWrapper, BaseInferLightWorker
import time
from sanic import Sanic
from sanic.response import json as json_response
import random
import logging

logging.basicConfig(level=logging.INFO)

class DummyModel:

    def __init__(self, config) -> None:
        self.config = config

    def forward(self, inputs):
        time.sleep(1)
        return inputs

class MyWorker(BaseInferLightWorker):

    def load_model(self, model_args):
        self.model = DummyModel(model_args)
        return

    def build_batch(self, requests):
        return sum(requests,[])

    def inference(self, batch):
        return self.model.forward(batch)

if __name__=='__main__':
    dummy_config = {'hello':'world'}
    dummy_model = DummyModel(dummy_config)
    wrapped_model = LightWrapper(MyWorker, dummy_config, batch_size=16, max_delay=0.1)

    app = Sanic('test')

    @app.get('/predict')
    async def predict(request):
        dummy_input = [random.random()]
        output = dummy_model.forward(dummy_input)
        assert output[0]==dummy_input[0]
        return json_response({'output':output[0]})

    @app.get('/batch_predict')
    async def batched_predict(request):
        dummy_input = [random.random()]
        result = await wrapped_model.predict(dummy_input)
        assert result==dummy_input[0], result
        return json_response({'output': result})

    app.run(port=8888)
