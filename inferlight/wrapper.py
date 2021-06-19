import multiprocessing as mp
from queue import Empty
import uuid
from cachetools import TTLCache
import asyncio
import threading
import logging

class LightWrapper:

    def __init__(self, worker_class, model_args:dict, 
                 batch_size=16, max_delay=0.1) -> None:
        # setup logger
        self.logger = logging.getLogger('InferLight-Wrapper')
        self.logger.setLevel(logging.INFO)

        self.result_queue = mp.Queue()
        self.data_queue = mp.Queue()
        self.result_cache = TTLCache(maxsize=10000, ttl=5)

        self.mp = mp.get_context('spawn')

        # setup worker
        self.logger.info('Starting worker...')
        worker_ready_event = self.mp.Event()
        self._worker_p = self.mp.Process(worker_class.start, args=(
            self.data_queue, self.result_queue, model_args, batch_size, max_delay, worker_ready_event
        ), daemon=True)
        self._worker_p.start()
        is_ready = worker_ready_event.wait(timeout=5)
        if is_ready:
            self.logger.info('Worker started!')
        else:
            self.logger.error('Failed to start worker!')

        self.back_thread = threading.Thread(target=self.collect_result, name="thread_collect_result")
        self.back_thread.daemon = True
        self.back_thread.start()


    def _collect_result(self):
        self.logger.info('Result collecting thread started!')
        while True:
            try:
                msg = self.result_queue.get(block=True, timeout=0.1)
            except Empty:
                msg = None

            if msg is not None:
                (task_id, result) = msg
                self.result_cache[task_id] = result

    async def get_result(self, task_id):
        while task_id not in self.result_cache:
            asyncio.sleep(0.01)
        return self.result_cache.pop(task_id)

    async def predict(self, input, timeout=2):
        # generate unique task_id
        task_id = str(uuid.uuid4())

        # send input to worker process
        self.data_queue.put((task_id, input))
        try:
            result = asyncio.wait_for(self.get_result(task_id), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        return result


