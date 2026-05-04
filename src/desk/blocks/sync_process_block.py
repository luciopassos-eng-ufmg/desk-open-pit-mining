from collections import deque
from typing import Dict, Callable, Optional, List
import simpy

from desk.core.base_block import BaseBlock
from desk.core.entity import Entity, EventLogger


class SyncProcessBlock(BaseBlock):
    def __init__(
        self,
        name: str,
        env: simpy.Environment,
        required_entity_types: Dict[str, int],
        delay_time: Callable[[], float],
        resource: Optional[simpy.Resource] = None,
        resource_units: int = 1,
        primary_entity_type: Optional[str] = None,
        event_logger: EventLogger = None
    ):
        super().__init__(name, env, event_logger)

        self.group_attributes_to_assign = {}

        self.required_entity_types = required_entity_types
        self.delay_time = delay_time
        self.resource = resource
        self.resource_units = resource_units
        self.primary_entity_type = primary_entity_type

        self.waiting = {
            entity_type: deque()
            for entity_type in required_entity_types
        }

        self.resource_name = None
        self.entities_processed = 0
        self.total_delay_time = 0.0
        self.total_queue_time = 0.0
        self.sync_counter = 0

    def assign_group_attributes(self, **attributes):
        """
        Configure attributes to be assigned once per synchronized group.

        Each value can be:
        - a fixed value
        - a callable returning a value
        """
        self.group_attributes_to_assign = attributes

    def _apply_group_attributes(self, all_entities):
        """
        Apply group-level attributes once and copy the same value to
        every entity in the synchronized batch.
        """
        assigned_values = {}

        for attr_name, attr_value in self.group_attributes_to_assign.items():
            if callable(attr_value):
                value = attr_value()
            else:
                value = attr_value

            assigned_values[attr_name] = value

            for entity in all_entities:
                entity.add_attribute(attr_name, value)

        return assigned_values

    def set_resource_name(self, name: str):
        self.resource_name = name

    def _get_entity_type(self, entity):
        return entity.get_attribute("entity_type", entity.id.split("_")[0])

    def _can_start_batch(self) -> bool:
        for entity_type, qty in self.required_entity_types.items():
            if len(self.waiting[entity_type]) < qty:
                return False
        return True

    def _pop_batch(self) -> Dict[str, List]:
        batch = {}
        for entity_type, qty in self.required_entity_types.items():
            batch[entity_type] = [
                self.waiting[entity_type].popleft()
                for _ in range(qty)
            ]
        return batch

    def _flatten_batch(self, batch: Dict[str, List]) -> List:
        all_entities = []
        for entities in batch.values():
            all_entities.extend(entities)
        return all_entities

    def process_entity(self, entity):
        entity_type = self._get_entity_type(entity)

        if entity_type not in self.waiting:
            raise ValueError(
                f"Entity type '{entity_type}' not configured for block '{self.name}'"
            )

        entity.route_history.append(self.name)
        self.waiting[entity_type].append(entity)

        while self._can_start_batch():
            batch = self._pop_batch()
            self.env.process(self._run_batch(batch))

        yield self.env.timeout(0)

    @property
    def supports_interlock(self) -> bool:
        return self.resource is not None

    def _run_batch(self, batch: Dict[str, List]):
        all_entities = self._flatten_batch(batch)

        if self.primary_entity_type:
            primary_candidates = batch.get(self.primary_entity_type, [])
            if not primary_candidates:
                raise ValueError(
                    f"Primary entity type '{self.primary_entity_type}' "
                    f"not found in batch for block '{self.name}'"
                )
            primary = primary_candidates[0]
        else:
            primary = all_entities[0]

        self.sync_counter += 1
        group_id = f"{self.name}_SYNC_{self.sync_counter}"
        
        matched_by_type = {
            entity_type: [e.id for e in entities]
            for entity_type, entities in batch.items()
        }
        matched_ids = [e.id for e in all_entities]

        for entity in all_entities:
            entity.add_attribute("sync_group_id", group_id)
            entity.add_attribute("sync_block", self.name)
            entity.add_attribute("matched_ids", matched_ids)
            entity.add_attribute(f"{self.name}_matched_entities", matched_by_type)

        self._apply_group_attributes(all_entities)
        
        # LOOP DE RETRY para suportar preempção
        while True:
            wait_event = self._wait_if_interlocked()
            if wait_event is not None:
                self._trace(
                    "interlock",
                    primary,
                    self.resource_name,
                    details=f"sync_group={group_id}, waiting for interlock release"
                )
                yield wait_event

            requests = []
            acquired = []
            queue_start = self.env.now

            try:
                if self.resource is not None:
                    for _ in range(self.resource_units):
                        if isinstance(self.resource, simpy.PreemptiveResource):
                            req = self.resource.request(priority=0, preempt=False)
                        elif isinstance(self.resource, simpy.PriorityResource):
                            req = self.resource.request(priority=0)
                        else:
                            req = self.resource.request()
                        requests.append(req)

                    yield simpy.AllOf(self.env, requests)
                    acquired = list(requests)

                    queue_time = self.env.now - queue_start
                    self.total_queue_time += queue_time

                    for entity in all_entities:
                        entity.add_attribute(f"{self.name}_queue_time", queue_time)
                else:
                    for entity in all_entities:
                        entity.add_attribute(f"{self.name}_queue_time", 0.0)

                self.log_start(primary, self.resource_name)
                self._trace(
                    "service_start",
                    primary,
                    self.resource_name,
                    details=f"sync_group={group_id}, entities={matched_ids}"
                )

                delay = max(0.0, self.delay_time())
                yield self.env.timeout(delay)

                for entity in all_entities:
                    entity.add_attribute(f"{self.name}_service_time", delay)
                    entity.add_attribute(f"{self.name}_processed", True)

                for entity in all_entities:
                    self._apply_attributes(entity)

                self.entities_processed += 1
                self.total_delay_time += delay

                self.log_complete(primary, self.resource_name)
                self._trace(
                    "service_end",
                    primary,
                    self.resource_name,
                    details=f"sync_group={group_id}, entities={matched_ids}, delay={delay:.2f}"
                )

                break  # sucesso: sai do retry loop

            except simpy.Interrupt as interrupt:
                self._trace(
                    "interrupt",
                    primary,
                    self.resource_name,
                    details=f"sync_group={group_id}, cause={interrupt.cause}"
                )

                # libera qualquer request já adquirido
                for req in acquired:
                    try:
                        self.resource.release(req)
                    except Exception:
                        pass

                # opcional: pequeno yield para reentrar limpo
                yield self.env.timeout(0)

            finally:
                # garante liberação se saiu sem interrupção
                if self.resource is not None:
                    for req in acquired:
                        try:
                            self.resource.release(req)
                        except Exception:
                            pass

        self.env.process(self.send_to_next(all_entities))