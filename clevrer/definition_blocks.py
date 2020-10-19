from nscl.datasets.definition import DatasetDefinitionBase
from jacinle.logging import get_logger

logger = get_logger(__file__)


class BlockDefinition(DatasetDefinitionBase):
    operation_signatures = [
        # Part 1: CLEVRER dataset.
        ('scene', [], [], 'object_set'),
        ('objects', [], [], 'object_set'),
        ('events', [], [], 'event_set'),
        ('unseen_events', [], [], 'event_set'),
        ('filter', ['concept'], ['object_set'], 'object_set'),
        ('intersect', [], ['object_set', 'object_set'], 'object_set'),
        ('union', [], ['object_set', 'object_set'], 'object_set'),
        
        ('filter_order', ['concept'], ['object_set'], 'object_set'),
        ('negate', [], ['bool'], 'bool'),
        ('belong_to', [], ['object_set', 'object_set'], 'bool'),
        ('filter_status', ['concept'], ['object_set', 'time_set'], 'object_set'),
        ('filter_temporal', ['concept'], ['object_set', 'time_set'], 'object_set'),
        ('filter_spatial', ['concept'], ['object_set'], 'object_set'),

        ('query', ['attribute'], ['object'], 'word'),
        ('exist', [], ['object_set'], 'bool'),
        ('count', [], ['object_set'], 'integer'),
    ]

    attribute_concepts = {
        'color':['red', 'yellow', 'green', 'blue'],
        'material': ['rubber'],
        'shape': ['block']
    }

    spatial_concepts = {
        'order': ['top', 'middle', 'bottom'],
    }

    temporal_concepts ={
        'status': ['falling', 'stationary'],
            }
    
    time_concepts ={
        'time2': ['start', 'end'],
        }
