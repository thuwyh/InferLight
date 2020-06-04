from abc import ABC, abstractmethod

class Handler(ABC):

    def __init__(self):
        self.load_model()

    @abstractmethod
    def load_model(self):
        """载入模型
        """
        raise NotImplementedError

    # @abstractmethod
    # def handle_http_request(self, request):
    #     raise NotImplementedError

    @abstractmethod
    def batch_predict(self, batch_data):
        """批量推理

        :param batch_data: List，每个元素是dispatch_http_request的返回值
        :return: List，你的结果
        :rtype: List
        """
        raise NotImplementedError

    @abstractmethod
    def dispatch_http_request(self, request):
        """从request中获取数据

        :param request: Sanic的request对象
        :return: 送入batch_predict的对象
        :rtype: [type]
        """
        raise NotImplementedError


    @abstractmethod
    def make_http_response(self, result):
        """将结果变为可返回的结果

        :param result: batch_predict返回List中的元素
        :type result: Unknown
        :return: 返回结果
        :rtype: String or Dict
        """
        raise NotImplementedError
