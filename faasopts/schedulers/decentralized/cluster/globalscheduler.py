import logging
import re
from collections import defaultdict
from typing import Dict, List, Tuple

from faas.context import PlatformContext
from faas.system import Metrics, FunctionReplica, FaasSystem
from faas.system.scheduling.decentralized import GlobalScheduler, BaseGlobalSchedulerConfiguration
from faas.util.constant import zone_label

from faasopts.utils.infrastructure.filter import get_filtered_nodes_in_zone, get_filtered_nodes

logger = logging.getLogger(__name__)


def extract_zone_out_of_name(name: str):
    pattern = r'(zone)-([a-z])'
    match = re.search(pattern, name)
    if match:
        return match.group(0)
    return None


class ClusterGlobalScheduler(GlobalScheduler):
    def __init__(self, base_config: BaseGlobalSchedulerConfiguration, storage_local_schedulers: Dict[str, str],
                 ctx: PlatformContext,
                 metrics: Metrics, max_scale, faas: FaasSystem):
        super().__init__(base_config)
        self.ctx = ctx
        self.metrics = metrics
        self.max_scale = max_scale
        self.storage_local_schedulers = storage_local_schedulers
        self.zones = ctx.zone_service.get_zones()
        self.faas = faas


    def __str__(self):
        return f"GlobalScheduler: {self.scheduler_name}"

    def find_cluster(self, replica: FunctionReplica):
        origin_cluster = replica.labels[zone_label]
        nodes_in_cluster_available = get_filtered_nodes_in_zone(self.ctx, replica, origin_cluster)
        if len(nodes_in_cluster_available) == 0:
            nodes_available, pod_per_node = get_filtered_nodes(self.ctx, replica)
            logger.info(pod_per_node)

            pod_per_zone = defaultdict(int)
            if len(nodes_available) > 0:
                cpu_dict = defaultdict(list)
                for node, has_enough in nodes_available:
                    zone = extract_zone_out_of_name(node)
                    pod_per_zone[zone] = pod_per_zone[zone] + pod_per_node[node]
                    logger.info(f'{node} - {zone} - {pod_per_node[node]} - {pod_per_zone[zone]}')
                    if not has_enough:
                        continue
                    cpu_dict[zone].append((node, pod_per_node[node]))

                zone_distances: List[Tuple[str, float]] = self.create_network_distances(origin_cluster)
                zone_distances = sorted(zone_distances, key=lambda x: x[1])
                for zone, distance in zone_distances:
                    cpus = cpu_dict[zone]
                    if len(cpus) > 0:
                        logger.info(f'Found scheduler: {zone} - {distance}')
                        found_scheduler = self.storage_local_schedulers[zone]
                        return found_scheduler, zone
                return '', ''

            else:
                logger.error(f'No node found for replica {replica}')
                return '', ''
        else:
            found_scheduler = self.storage_local_schedulers[origin_cluster]
            logger.info("found scheduler: {}".format(found_scheduler))
            return found_scheduler, origin_cluster

    def create_network_distances(self, origin_cluster: str) -> List[Tuple[str, float]]:
        distances = []
        origin_node = self.ctx.node_service.find_nodes_in_zone(origin_cluster)[0]
        for zone in self.zones:
            nodes = self.ctx.node_service.find_nodes_in_zone(zone)
            node = nodes[0]
            latency = self.ctx.network_service.get_latency(node.name, origin_node.name)
            distances.append((zone, latency))
        return distances
