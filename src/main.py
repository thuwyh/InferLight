from asyncio import sleep
from datetime import datetime
from multiprocessing import Manager, Process
from pathlib import Path
from uuid import uuid4

import torch
from sanic import Sanic
from sanic.log import logger
from sanic.response import json, text

from handler import batch_predict, load_model
from worker import fake_predict, worker_function

manager = Manager()
task_queue = manager.Queue()
result_dict = manager.dict()

app = Sanic("QA inference")
p = Process(target=worker_function, args=(
    task_queue, result_dict, load_model, batch_predict,), daemon=True)


@app.listener('before_server_stop')
async def notify_server_stopping(app, loop):
    print('Server shutting down!')


@app.route("/")
async def test(request):
    return text("I am alive!")


@app.route("/predict", methods=["POST"])
async def predict(request):
    text_a = request.json.get('text_a')
    text_b = request.json.get('text_b')

    uuid = str(uuid4())
    task_data = {
        "id": uuid,
        "data": (text_a, text_b)
    }
    task_queue.put(task_data, False)
    while True:
        if uuid in result_dict:
            result = result_dict[uuid]
            result_dict.pop(uuid)
            break
        await sleep(0.005)
    return json({"result": str(result)})


if __name__ == "__main__":
    p.start()
    app.run(host="0.0.0.0", port=8000)
