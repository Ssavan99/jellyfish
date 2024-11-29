from src.server.worker.worker import Worker
import logging


class WorkerManager(object):
    def __init__(self, opts, profiled_latencies) -> None:
        self.num_gpus = opts.n_gpus
        self.profiled_latencies = profiled_latencies
        self._workers_lst = [None] * self.num_gpus
        self._start_workers(opts)

    def _start_workers(self, opts):
        for i in range(self.num_gpus):
            worker = Worker(opts, gpu_number=i,
                            profiled_latencies=self.profiled_latencies)
            worker.start()
            self._workers_lst[i] = worker
            logging.info(f"Worker started on GPU {i}")

    def _stop_workers(self):
        for i in range(self.num_gpus):
            self._workers_lst[i].close()
            self._workers_lst[i].join()

    def close(self):
        self._stop_workers()

    def update_worker_model(self, gpu_number, model_number, batch_size):
        # self._workers_lst[gpu_number].set_desired_model(
        #     model_number, batch_size)
        """
        Dynamically updates the worker's desired model and batch size,
        ensuring compatibility with the current request rate and system constraints.
        """
        # Check if the batch size is valid for the given model
        if batch_size < 1 or batch_size > len(self.profiled_latencies[model_number]):
            logging.warn(f"Invalid batch size {batch_size} for model {model_number} on GPU {gpu_number}. "
                     f"Resetting batch size to default (1).")
            batch_size = 1

        logging.info(f"Worker on GPU {gpu_number} updated to model {model_number} with batch size {batch_size}.")
        # Update the worker with the validated model and batch size
        self._workers_lst[gpu_number].set_desired_model(model_number, batch_size)
        

    def schedule(self, gpu_number, request):
        """
        Schedule a request to the worker on the specified GPU.
        """
        worker = self._workers_lst[gpu_number]
        if not worker.can_accept_request():
            logging.warning(f"Worker on GPU {gpu_number} is overloaded. Request may be delayed or dropped.")
        # Optionally, implement load redistribution here.
        logging.info(f"Scheduling request on GPU {gpu_number}. Queue size: "
                 f"{self._workers_lst[gpu_number].queue_size()}")
        worker.put_request(request)
        # self._workers_lst[gpu_number].put_request(request)

    def get_response(self, gpu_number, non_block=True):
        if non_block:
            response = self._workers_lst[gpu_number].get_response_nowait()
        else:
            raise RuntimeError("Not implemented!")
        return response

    def is_worker_overloaded(self, gpu_number):
        """
        Check if the worker on the specified GPU is overloaded.
        """
        return self._workers_lst[gpu_number].is_overloaded()

