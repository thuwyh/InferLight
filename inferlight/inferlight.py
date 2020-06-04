from multiprocessing import Manager, Process
from inspect import signature
from functools import partial, update_wrapper
from uuid import uuid4
from inferlight.worker import worker_function
from inferlight.handler import Handler
from inferlight.http_frontend import app, predict


class InferLight:

    def __init__(self, handler: Handler):
        self.manager = Manager()
        self.task_queue = self.manager.Queue()
        self.result_dict = self.manager.dict()
        self.handler = handler
        self.http_frontend = app
        self.http_predict_handler = None

        self.wrap_http_predict_fn(self.handler.dispatch_http_request,
                                  self.handler.make_http_response)

    def put_task(self, data):
        task_id = str(uuid4())
        task_data = {
            "id": task_id,
            "data": data
        }
        self.task_queue.put(task_data, False)
        return task_id

    def get_result(self, task_id):
        if task_id in self.result_dict:
            result = self.result_dict[task_id]
            self.result_dict.pop(task_id)
            return result
        else:
            return None
    def register_http_endpoint(self, url: str, methods=['POST']):
        """注册http预测入口

        :param url: 预测URL
        :type url: str
        :raises ValueError: 当数据处理函数未注册时抛出异常
        """
        if self.http_predict_handler is None:
            raise NotImplementedError(
                "register a http data dispatch function first!")
        self.http_frontend.add_route(
            self.http_predict_handler, url, methods=methods)

    def wrap_http_predict_fn(self, data_dispatch_fn, make_response_fn):
        """注册http数据处理函数

        :param data_dispatch_fn: 数据处理函数，包含唯一传入参数request
        :type data_dispatch_fn: Function
        """

        # def wrapper():
        args = list(signature(data_dispatch_fn).parameters.keys())

        if not args:
            raise ValueError(
                "Required parameter `request` missing "
            )

        self.http_predict_handler = partial(predict,
                                            data_dispatch_fn=data_dispatch_fn,
                                            make_response_fn=make_response_fn,
                                            light=self)
        update_wrapper(self.http_predict_handler, predict)

    def start_service(self, http=False):
        p = Process(target=worker_function, args=(
            self.task_queue, self.result_dict, self.handler,), daemon=True)
        p.start()
        if http:
            self.http_frontend.run(host="0.0.0.0", port=8000)
