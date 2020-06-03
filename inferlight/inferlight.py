from multiprocessing import Manager, Process
from uuid import uuid4
from inferlight.worker import worker_function
from inferlight.handler import Handler

class InferLight:

    def __init__(self, handler:Handler):
        self.manager = Manager()
        self.task_queue = self.manager.Queue()
        self.result_dict = self.manager.dict()
        self.handler = handler

    def put_task(self, data):
        task_id = str(uuid4())
        task_data = {
            "id": task_id,
            "data": data
        }
        task_queue.put(task_data, False)
        return task_id

    def get_result(self, task_id):
        if task_id in result_dict:
            result = result_dict[task_id]
            result_dict.pop(task_id)
            return result
        else:
            return None

    def start_service(self, http_frontend=None):
        p = Process(target=worker_function, args=(
            self.task_queue, self.result_dict, self.handler,), daemon=True)
        p.start()
        if http_frontend is not None:
            http_frontend.run(host="0.0.0.0", port=8000)
