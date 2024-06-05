import logging
import signal
from collections import defaultdict
from typing import Dict, List

from faas.context import PlatformContext
from faas.system import Metrics, FaasSystem, FunctionReplica
from kubernetes import client

from faasopts.schedulers.decentralized.pressure.globalscheduler import PressureGlobalScheduler
from faasopts.utils.pressure.api import PressureScaleScheduleEvent
from faasopts.utils.pressure.service import PressureService

logger = logging.getLogger(__name__)




class PressureK8sGlobalScheduler:
    def __init__(self, scheduler_name: str, storage_local_schedulers: Dict[str, str], ctx: PlatformContext,
                 faas: FaasSystem, global_scheduler: PressureGlobalScheduler, pressure_service: PressureService,
                 metrics: Metrics, delay, max_scale):
        self.v1 = client.CoreV1Api()
        self.scheduler_name = scheduler_name
        self.ctx = ctx
        self.metrics = metrics
        self.delay = delay
        self.max_scale = max_scale
        self.storage_local_schedulers = storage_local_schedulers
        self.faas = faas
        self.global_scheduler = global_scheduler
        self.pressure_service = pressure_service
        self.zones = [zone for zone in storage_local_schedulers.keys()]
        self.running = True
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def __str__(self):
        return f"Scheduler: {self.scheduler_name}"

    def signal_handler(self, signal, frame):
        logger.info('Signal received!')
        self.running = False

    def start_schedule(self):
        logger.info("Start scheduling %s" % self.scheduler_name)
        while self.running:
            pressure_values = self.pressure_service.wait_for_all_pressures(self.zones)
            # we only want pressure values that have not been resolved yet
            pressure_values = pressure_values[pressure_values['solved'] == 0]
            scale_schedule_events = self.global_scheduler.find_clusters_for_autoscaler_decisions(pressure_values)
            scale_ups: Dict[str, List[FunctionReplica]] = defaultdict(list)
            scale_downs: Dict[str, List[FunctionReplica]] = defaultdict(list)

            for scale in scale_schedule_events:
                if scale.delete:
                    scale_downs[scale.fn].extend(scale.replicas)
                else:
                    scale_ups[scale.fn].extend(scale.replicas)
            faas: FaasSystem = self.faas
            for fn, replicas in scale_ups.items():
                faas.scale_up(fn, replicas)
            for fn, replicas in scale_downs.items():
                faas.scale_down(fn, replicas)


        logger.info("End scheduling %s" % self.scheduler_name)

    def handle_delete(self, scale: PressureScaleScheduleEvent):
        self.faas.scale_down(scale.fn, scale.replicas)

    def handle_create(self, scale: PressureScaleScheduleEvent):
        self.faas.scale_up(scale.fn, scale.replicas)
