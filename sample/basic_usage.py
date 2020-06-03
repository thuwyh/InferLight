from inferlight import app, InferLight
from .handler import MyHandler

handler = MyHandler()
light = InferLight(handler)

@app.route("/predict", methods=["POST"])
async def predict(request):
    text_a = request.json.get('text_a')
    text_b = request.json.get('text_b')

    task_id = light.put_task((text_a, text_b))
    while True:
        result = light.get_result()
        if result is not None:
            break
        await sleep(0.005)
    return json({"result": str(result)})

if __name__ == "__main__":
    light.start_service()


