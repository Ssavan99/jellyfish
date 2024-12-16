import logging
import src.utils as utils
import threading

_TOTAL_MODELS = 16

class ControllerClient(object):
    def __init__(self,  host, port_number, frame_rate, slo, init_bw, lat_wire, init_model_number=None) -> None:
        self.host = host
        self.port_number = port_number
        self._frame_rate = frame_rate  # Initial frame rate
        self._slo = slo
        self._next_model = init_model_number
        self._bw = init_bw
        self._bw_cur_model = -1
        self._lat_wire = lat_wire

        self._smoothing_filter = utils.ErrorBasedFilter()
        
        # For thread-safe access
        self._lock = threading.Lock()
        self._frame_s = utils.compute_frame_size(_TOTAL_MODELS)

        # Miss rate tracking
        self._missed_frames = 0
        self._total_frames = 0

        self._ema_miss_rate = 0.0  # Exponential moving average of miss rate
        self._alpha = 0.1  # Smoothing factor for EMA

    def update_next_model(self, next_model):
        with self._lock:
            self._next_model = next_model if next_model != -1 else 0

    def get_next_model(self):
        def _aggressive_backoff(bw, model):
            net_throughput = (bw / self._frame_s[model])
            if net_throughput <= self._frame_rate:
                model = 0
                logging.info(f"Throttling network (bw {bw}) send with smallest model {model}")
            return model

        def _next_best_backoff(bw, model):
            while model > 0:
                net_throughput = (bw / self._frame_s[model])
                if net_throughput > self._frame_rate:
                    break
                model = model - 1
            return model

        def _next_slo_aware_backoff(bw, model):
            _SERVER_PERCENTAGE = 0.75
            _NET_BUDGET = (1.0 - _SERVER_PERCENTAGE) * self._slo
            while model > 0:
                net_lat = utils.compute_net_lat(self._frame_s[model], bw, self._lat_wire)
                if net_lat <= _NET_BUDGET:
                    break
                model = model - 1
            return model

        _POLICY = "BEST_BACKOFF"

        with self._lock:
            model = self._next_model
            bw = self._bw

        if _POLICY == "AGGRESIVE_BACKOFF":
            return _aggressive_backoff(bw, model)
        elif _POLICY == "BEST_BACKOFF":
            return _next_best_backoff(bw, model)
        elif _POLICY == "SERVER_INDEPENDENT_BACKOFF":
            return _next_best_backoff(bw, _TOTAL_MODELS - 1)
        elif _POLICY == "FIXED_SIZE":
            frame_size = 7
            return frame_size
        elif _POLICY == "SLO_AWARE_BACKOFF":
            model = _next_best_backoff(bw, _TOTAL_MODELS - 1)
            return _next_slo_aware_backoff(bw, model)
        else:
            return model

    def update_miss_rate(self, was_frame_missed: bool):
        with self._lock:
            miss = 1 if was_frame_missed else 0
            self._ema_miss_rate = self._alpha * miss + (1 - self._alpha) * self._ema_miss_rate

            if self._ema_miss_rate > 0.01:  # If EMA miss rate > 1%
                proportional_reduction = int(self._frame_rate * 0.9)  # Reduce frame rate by 10%
                new_frame_rate = max(15, proportional_reduction)
                if new_frame_rate != self._frame_rate:
                    logging.info(f"Reducing frame rate to {new_frame_rate} FPS to maintain miss rate <= 1%")
                    self._frame_rate = new_frame_rate

    def _get_smooth_latency(self, network_time, cur_model):
        with self._lock:
            if self._bw_cur_model != cur_model:
                self._bw_cur_model = cur_model
                self._smoothing_filter.reset()

            self._smoothing_filter.update(network_time)
            return self._smoothing_filter.predict()

    def update_bw(self, metadata):
        total_time = (metadata.client_recv_ack_ts - metadata.client_send_req_ts) * 1e3
        time_on_server = (metadata.server_send_ack_ts - metadata.server_recv_req_ts) * 1e3

        network_time = (total_time - time_on_server)
        smoothed_network_time = self._get_smooth_latency(network_time, metadata.desired_model)
        metadata.network_time = smoothed_network_time

        bw = utils.compute_net_throughput(size=metadata.frame_wire_size, total_time=smoothed_network_time, wire_time=self._lat_wire)
        
        logging.debug(f"Update client bw {bw} frame id {metadata.frame_id}")
        
        with self._lock:
            self._bw = bw

    def get_bw(self):
        with self._lock:
            return self._bw

    def get_frame_rate(self):
        with self._lock:
            return self._frame_rate
