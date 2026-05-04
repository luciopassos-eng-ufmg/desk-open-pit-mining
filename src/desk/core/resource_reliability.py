from dataclasses import dataclass
from typing import Callable, Optional
import random
import simpy


@dataclass
class ResourceReliabilityConfig:
    """
    Configuração opcional de confiabilidade para um recurso.

    Pode usar:
    - mttf + mttr
    ou
    - funções customizadas time_to_failure_fn e repair_time_fn
    """
    mttf: Optional[float] = None
    mttr: Optional[float] = None
    time_to_failure_fn: Optional[Callable[[], float]] = None
    repair_time_fn: Optional[Callable[[], float]] = None
    enabled: bool = True
    preempt_priority: int = -10
    start_failed: bool = False
    name: Optional[str] = None

    def sample_time_to_failure(self) -> float:
        if self.time_to_failure_fn is not None:
            return max(0.0, self.time_to_failure_fn())
        if self.mttf is None or self.mttf <= 0:
            raise ValueError("ResourceReliabilityConfig requires mttf > 0 or time_to_failure_fn")
        return random.expovariate(1 / self.mttf)

    def sample_repair_time(self) -> float:
        if self.repair_time_fn is not None:
            return max(0.0, self.repair_time_fn())
        if self.mttr is None or self.mttr <= 0:
            raise ValueError("ResourceReliabilityConfig requires mttr > 0 or repair_time_fn")
        return random.expovariate(1 / self.mttr)


class ResourceReliabilityManager:
    """
    Gerencia os processos de falha/reparo dos recursos configurados no modelo.
    """

    def __init__(self, model):
        self.model = model

    def start_all(self):
        """
        Inicia um processo SimPy por recurso confiável configurado.
        """
        for resource_name, cfg in self.model.resource_reliability.items():
            if not cfg.enabled:
                continue

            if resource_name not in self.model.resources:
                raise ValueError(f"Reliability configured for unknown resource '{resource_name}'")

            resource = self.model.resources[resource_name]
            self.model.env.process(
                self._resource_breakdown_process(resource_name, resource, cfg)
            )

    def _resource_breakdown_process(self, resource_name, resource, cfg: ResourceReliabilityConfig):
        """
        Processo genérico de falha e reparo para qualquer recurso.
        """

        # estado extra do recurso
        resource.is_down = cfg.start_failed
        resource.failure_count = 0
        resource.total_downtime = 0.0
        resource.last_failure_time = None
        resource.last_repair_time = None

        # se quiser começar falhado
        if cfg.start_failed:
            down_start = self.model.env.now
            req = self._request_resource_for_repair(resource, cfg)
            yield req

            repair_time = cfg.sample_repair_time()
            yield self.model.env.timeout(repair_time)

            self._release_resource(resource, req)
            resource.is_down = False
            resource.failure_count += 1
            resource.total_downtime += self.model.env.now - down_start
            resource.last_repair_time = self.model.env.now

        while cfg.enabled:
            ttf = cfg.sample_time_to_failure()
            yield self.model.env.timeout(ttf)

            down_start = self.model.env.now
            resource.is_down = True
            resource.last_failure_time = self.model.env.now

            req = self._request_resource_for_repair(resource, cfg)
            yield req

            self._trace_preempt(resource_name, cfg)

            repair_time = cfg.sample_repair_time()
            yield self.model.env.timeout(repair_time)

            self._release_resource(resource, req)

            resource.is_down = False
            resource.failure_count += 1
            resource.total_downtime += self.model.env.now - down_start
            resource.last_repair_time = self.model.env.now

            self._trace_repair_end(resource_name, repair_time)

    def _request_resource_for_repair(self, resource, cfg):
        if isinstance(resource, simpy.PreemptiveResource):
            return resource.request(priority=cfg.preempt_priority, preempt=True)
        elif isinstance(resource, simpy.PriorityResource):
            return resource.request(priority=cfg.preempt_priority)
        else:
            return resource.request()

    def _release_resource(self, resource, req):
        try:
            resource.release(req)
        except Exception:
            pass

    def _trace_preempt(self, resource_name: str, cfg: ResourceReliabilityConfig):
        tracer = getattr(self.model, "event_tracer", None)
        if tracer:
            tracer.trace(
                "preempt",
                entity_id=f"{resource_name}",
                resource_name=resource_name,
                details=f"failure started"
            )

    def _trace_repair_end(self, resource_name: str, repair_time: float):
        tracer = getattr(self.model, "event_tracer", None)
        if tracer:
            tracer.trace(
                "service_end",
                entity_id=f"{resource_name}",
                resource_name=resource_name,
                details=f"failure ended, repair_time={repair_time:.2f}"
            )