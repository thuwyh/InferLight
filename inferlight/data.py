from enum import Enum

class InferStatus(Enum):
    SUCCEED = 0
    TIMEOUT = 1

class InferResponse:

    def __init__(self, status: InferStatus, result) -> None:
        self.status = status
        self.result = result

    def succeed(self):
        return self.status==InferStatus.SUCCEED

    