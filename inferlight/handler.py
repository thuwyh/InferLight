from abc import ABC, abstractmethod

class Handler(ABC):

    def __init__(self):
        self.load_model()

    @abstractmethod
    def load_model(self, handler):
        """载入模型
        """
        raise NotImplementedError

    # @abstractmethod
    # def handle_http_request(self, request):
    #     raise NotImplementedError

    @abstractmethod
    def batch_predict(self, batch_data):
        """批量推理

        Arguments:
            batch_data {List} -- [由task data组成的列表]

        Returns:
            [array-like] -- [每个样本一个结果]
        """
        raise NotImplementedError

    def predict(self, data):
        pass
