from abc import ABC, abstractmethod

class Handler(ABC):

    @abstractmethod
    def load_model(self, parameter_list):
        """载入模型
        """
        raise NotImplementedError

    @abstractmethod
    def batch_predict(self, batch_data):
        """批量推理

        Arguments:
            batch_data {List} -- [由task data组成的列表]

        Returns:
            [array-like] -- [每个样本一个结果]
        """
        pass

    @abstractmethod
    def predict(self, data):
        pass
