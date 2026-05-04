# core/desk_resource.py
from typing import Set, Optional
import simpy

class InterlockableResourceMixin:
    def __init__(self, env, capacity: int, name: str, max_queue=None):
        super().__init__(env, capacity=capacity)
        self.name = name
        self.max_queue = float("inf") if max_queue is None else max_queue
        self.is_buffer_max = False
        self.interlock_targets: Set[object] = set()

    @property
    def display_state(self) -> str:
        if getattr(self, "is_down", False):
            return "manutencao"
        if self.is_buffer_max:
            return "buffer_max"
        return "operando"

    def interlock_to(self, target_block):
        if not getattr(target_block, "supports_interlock", False):
            raise ValueError(
                f"Bloco '{getattr(target_block, 'name', str(target_block))}' "
                f"não suporta intertravamento. "
                f"Apenas processos com recurso podem ser alvo de interlock_to()."
            )
        self.interlock_targets.add(target_block)
        return target_block

    def refresh_buffer_state(self):
        old = self.is_buffer_max
        self.is_buffer_max = len(self.queue) >= self.max_queue
        if self.is_buffer_max != old:
            self._notify_interlock_targets()

    def _notify_interlock_targets(self):
        reason = f"buffer_max:{self.name}"
        for target in self.interlock_targets:
            if self.is_buffer_max:
                target.interlock(reason=reason, source_resource=self)
            else:
                target.clear_interlock(reason=reason, source_resource=self)


class DeskResource(InterlockableResourceMixin, simpy.Resource):
    pass


class DeskPriorityResource(InterlockableResourceMixin, simpy.PriorityResource):
    pass


class DeskPreemptiveResource(InterlockableResourceMixin, simpy.PreemptiveResource):
    pass