from sanic import Sanic
from sanic.log import logger
from sanic.response import json, text
from asyncio import sleep

app = Sanic("QA inference")


@app.listener('before_server_stop')
async def notify_server_stopping(app, loop):
    print('Server shutting down!')


@app.route("/")
async def test(request):
    return text("I am alive!")


async def predict(request,
                  data_dispatch_fn=None,
                  make_response_fn=None,
                  light=None):
    if data_dispatch_fn is None:
        raise ValueError("data_dispatch_fn should be implemented!")
    if make_response_fn is None:
        raise ValueError("make_response_fn should be implemented!")
    task_data = data_dispatch_fn(request)
    task_id = light.put_task(task_data)
    while True:
        result = light.get_result(task_id)
        if result is not None:
            break
        await sleep(0.005)
    response = make_response_fn(result)
    if isinstance(response, dict):
        return json(response)
    elif isinstance(response, str):
        return text(response)
    else:
        raise ValueError("response should be a dict or string!")
