from sanic import Sanic
from sanic.log import logger
from sanic.response import json, text


app = Sanic("QA inference")

@app.listener('before_server_stop')
async def notify_server_stopping(app, loop):
    print('Server shutting down!')


@app.route("/")
async def test(request):
    return text("I am alive!")
