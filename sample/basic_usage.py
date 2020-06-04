from asyncio import sleep
from sanic.response import json
from inferlight import InferLight
from handler import MyHandler

handler = MyHandler()
light = InferLight(handler)

@light.http_data_dispatch_fn
def dispatch_data(request):
    text_a = request.json.get('text_a')
    text_b = request.json.get('text_b')
    return (text_a, text_b)
    
light.register_http_endpoint('/predict', methods=['POST'])

if __name__ == "__main__":
    light.start_service(http=True)


