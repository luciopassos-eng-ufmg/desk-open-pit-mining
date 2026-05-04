# =====================================================================
# FILE: core/BaseBlock.py
# =====================================================================
from abc import ABC, abstractmethod
from typing import Any, Optional
import simpy
from desk.core.entity import Entity, EventLogger

# =====================================================================
# FILE: core/BaseBlock.py
# =====================================================================
class BaseBlock(ABC):
    """Abstract base class for all blocks."""
    
    def __init__(self, name: str, env: simpy.Environment, event_logger: EventLogger = None):
        self.name = name
        self.env = env
        self.next_block: Optional['BaseBlock'] = None
        self.statistics = {}
        self.event_logger = event_logger
        self.attributes_to_assign = {}  # Generic attribute assignment
        self.attributes_to_modify = {}  # Dynamic attribute modifications
        self.activity_priority = None  # Activity-specific priority
        self.tracer = getattr(env.model, 'event_tracer', None)  # Get tracer from model
        self.is_interlocked = False
        self.interlock_reasons = set()
        self._interlock_released_event = env.event()
        self._interlock_released_event.succeed()

    @property
    def supports_interlock(self) -> bool:
        return False

    @property
    def block_state(self) -> str:
        return "intertravado" if self.is_interlocked else "operando"
    
    def get_display_label(self) -> str:
        if self.is_interlocked:
            return f"{self.name}\n(intertravado)"
        return self.name

    def interlock(self, reason=None, source_resource=None):
        reason = reason or "manual"
        self.interlock_reasons.add(reason)

        if not self.is_interlocked:
            self.is_interlocked = True
            self._interlock_released_event = self.env.event()

        tracer = getattr(self.env.model, "event_tracer", None) if hasattr(self.env, "model") else None
        if tracer:
            tracer.trace(
                "interlock",
                entity_id=self.name,
                resource_name=getattr(source_resource, "name", None),
                details=f"block interlocked, reason={reason}"
            )

    def clear_interlock(self, reason=None, source_resource=None):
        reason = reason or "manual"
        self.interlock_reasons.discard(reason)

        if self.is_interlocked and not self.interlock_reasons:
            self.is_interlocked = False
            if not self._interlock_released_event.triggered:
                self._interlock_released_event.succeed()

        tracer = getattr(self.env.model, "event_tracer", None) if hasattr(self.env, "model") else None
        if tracer:
            tracer.trace(
                "interlock_clear",
                entity_id=self.name,
                resource_name=getattr(source_resource, "name", None),
                details=f"block released, reason={reason}"
            )

    def _wait_if_interlocked(self):
        if self.is_interlocked:
            return self._interlock_released_event
        return None

    def _trace(self, event_type: str, entity: Entity, resource_name: Optional[str] = None, 
               details: str = ""):
        """Helper method to trace events if verbose mode is enabled."""
        if self.tracer:
            self.tracer.trace(event_type, entity.id, resource_name, details)

    def assign_attributes(self, **attributes):
        """
        Configure attributes to assign to entities passing through this block.
        
        Args:
            **attributes: Key-value pairs where values can be:
                - Fixed values (int, float, str)
                - Callable functions that return values
        
        Example:
            block.assign_attributes(
                cost=100,
                revenue=lambda: random.uniform(200, 300),
                category="outpatient"
            )
        """
        self.attributes_to_assign = attributes

    def modify_attributes(self, **modifications):
        """
        Configure dynamic attribute modifications for entities.
        
        Args:
            **modifications: Key-value pairs where:
                - Key: attribute name to modify
                - Value: function that takes current value and returns new value
        
        Example:
            # Decrement sede by 1
            beber.modify_attributes(sede=lambda current: current - 1)
            
            # Increase cost by 10%
            activity.modify_attributes(cost=lambda current: current * 1.1)
            
            # Conditional modification
            activity.modify_attributes(
                priority=lambda current: max(0, current - 1)
            )
        """
        self.attributes_to_modify = modifications
    
    def set_activity_priority(self, priority: int):
        """
        Set the priority level for this activity.
        
        Args:
            priority: Integer priority (lower = higher priority, 0 = highest)
        
        Example:
            servir.set_activity_priority(0)  # Highest priority
            lavar.set_activity_priority(1)   # Lower priority
        """
        self.activity_priority = priority 

    
    def _apply_attributes(self, entity: Entity):
        """Apply configured attributes to entity."""

        assigned_attrs = []  # Track assigned attributes

        for attr_name, attr_value in self.attributes_to_assign.items():
            if callable(attr_value):
                value = attr_value()
            else:
                value = attr_value
            
            entity.add_attribute(attr_name, value)
            # print(f"[DEBUG ATTRIBUTE 1]: {attr_name}: {value}")
            entity.add_attribute(f"{self.name}_{attr_name}", value)            
            # print(f"[DEBUG ATTRIBUTE 2]: {self.name}_{attr_name}: {value}")

            # Record what was assigned
            assigned_attrs.append((attr_name, value))

            # print(f"[DEBUG] {attr_name}: {value}")

        return assigned_attrs  # Return list of (name, value) tuples

    def _modify_attributes(self, entity: Entity):
        """
        Apply dynamic attribute modifications to entity.
        
        NEW: Modifies existing attributes based on configured functions.
        """
        modified_attrs = []  # Track modifications

        for attr_name, modification_func in self.attributes_to_modify.items():
            
            # Get current value (with default of 0)
            current_value = entity.get_attribute(attr_name, 0)
            
            # Apply modification function
            new_value = modification_func(current_value)

            # Debug print
            # print(f"[DEBUG] {attr_name}: old={current_value} -> new={new_value}")
            
            # Update attribute
            entity.add_attribute(attr_name, new_value)

            # Record what was modified (old -> new)
            modified_attrs.append((attr_name, current_value, new_value))
        
        return modified_attrs  # Return list of (name, old_value, new_value) tuples

        
    def connect_to(self, next_block: 'BaseBlock'):
        """Connect this block to the next block in the flow."""
        self.next_block = next_block
        
    @abstractmethod
    def process_entity(self, entity: Entity):
        """Process an entity through this block. Must be implemented by subclasses."""
        pass

    def log_start(self, entity: Entity, resource_name: str = None):
        """Log activity start."""
        if self.event_logger:
            self.event_logger.log_event(
                case_id=entity.id,
                activity=self.name,
                timestamp=self.env.now,
                lifecycle='start',
                resource=resource_name,
                priority=entity.priority,
                activity_priority=self.activity_priority  # Log activity priority
            )
    
    def log_complete(self, entity: Entity, resource_name: str = None):
        """Log activity completion."""
        if self.event_logger:
            self.event_logger.log_event(
                case_id=entity.id,
                activity=self.name,
                timestamp=self.env.now,
                lifecycle='complete',
                resource=resource_name,
                priority=entity.priority,
                activity_priority=self.activity_priority  # Log activity priority
            )
    '''    
    def send_to_next(self, entity: Entity):
        """Send entity to the next connected block."""
        if self.next_block:
            yield from self.next_block.process_entity(entity)
        else:
            # Entity exits the system
            yield self.env.timeout(0)
    '''
    def send_to_next(self, entity_or_entities):
        if self.next_block is None:
            return
        # Caso seja lista de entidades
        if isinstance(entity_or_entities, (list, tuple)):
            for entity in entity_or_entities:
                self.env.process(self.next_block.process_entity(entity))
            yield self.env.timeout(0)
        # Caso seja uma única entidade
        else:
            yield from self.next_block.process_entity(entity_or_entities)
            
    def update_statistics(self, key: str, value: Any):
        """Update block statistics."""
        self.statistics[key] = value