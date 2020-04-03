from nscl.datasets.definition import DatasetDefinitionBase
from jacinle.logging import get_logger

logger = get_logger(__file__)


class CLEVRERDefinition(DatasetDefinitionBase):
    operation_signatures = [
        # Part 1: CLEVRER dataset.
        ('scene', [], [], 'object_set'),
        ('filter', ['concept'], ['object_set'], 'object_set'),
        ('relate', ['relational_concept'], ['object'], 'object_set'),
        ('relate_attribute_equal', ['attribute'], ['object'], 'object_set'),
        ('intersect', [], ['object_set', 'object_set'], 'object_set'),
        ('union', [], ['object_set', 'object_set'], 'object_set'),
        
        ('filter_order', ['concept'], ['object_set'], 'object_set'),
        ('negate', [], ['bool'], 'bool'),
        ('belong_to', [], ['object_set', 'object_set'], 'bool'),
        ('filter_temporal', ['concept'], ['object_set', 'time_set'], 'object_set'),

        ('query', ['attribute'], ['object'], 'word'),
        ('exist', [], ['object_set'], 'bool'),
        ('count', [], ['object_set'], 'integer'),
        ('get_frame', [], ['object_set', 'time_set'], 'object_set'),
        ('filter_in', [], ['object'], 'time_set'),
        ('filter_out', [], ['object'], 'time_set'),
        ('filter_before', [], ['time_set'], ['object_set, time_set']),
        ('filter_after', [], ['time_set'], ['object_set, time_set']),
        ('end', [], ['object_set'], 'time_set'),
        ('start', [], ['object_set'], 'time_set'),
        ('filter_collision', [], ['time_set'], ['object_set, time_set']),
        ('get_col_partner', [], ['object', 'col_set'], ['object_set']),
    ]

    attribute_concepts = {
        'color': ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow'],
        'material': ['rubber', 'metal'],
        'shape': ['cube', 'sphere', 'cylinder']
    }

    relational_concepts = {
        #'spatial_relation': ['left', 'right', 'front', 'behind'],
        'order': ['first', 'second', 'last'],
        'events': ['collision']
    }

    temporal_concepts ={
        'events': ['moving', 'after', 'stationary', 'before'],
        'order': ['first', 'second', 'last'],
        'scene': ['in', 'out' ]
            }
    time_concepts ={
        'time': ['start', 'end'],
        }

    synonyms = {
        "thing": ["thing", "object"],
        "sphere": ["sphere", "ball", "spheres", "balls"],
        "cube": ["cube", "block", "cubes", "blocks"],
        "cylinder": ["cylinder", "cylinders"],
        "large": ["large", "big"],
        "small": ["small", "tiny"],
        "metal": ["metallic", "metal", "shiny"],
        "rubber": ["rubber", "matte"],
    }

    word2lemma = {
        v: k for k, vs in synonyms.items() for v in vs
    }

    EBD_CONCEPT_GROUPS = '<CONCEPTS>'
    EBD_RELATIONAL_CONCEPT_GROUPS = '<REL_CONCEPTS>'
    EBD_ATTRIBUTE_GROUPS = '<ATTRIBUTES>'

    extra_embeddings = [EBD_CONCEPT_GROUPS, EBD_RELATIONAL_CONCEPT_GROUPS, EBD_ATTRIBUTE_GROUPS]

    @staticmethod
    def _is_object_annotation_available(scene):
        assert len(scene['objects']) > 0
        if 'mask' in scene['objects'][0]:
            return True
        return False

    def annotate_scene(self, scene):
        feed_dict = dict()

        if not self._is_object_annotation_available(scene):
            return feed_dict

        for attr_name, concepts in self.attribute_concepts.items():
            concepts2id = {v: i for i, v in enumerate(concepts)}
            values = list()
            for obj in scene['objects']:
                assert attr_name in obj
                values.append(concepts2id[obj[attr_name]])
            values = np.array(values, dtype='int64')
            feed_dict['attribute_' + attr_name] = values
            lhs, rhs = np.meshgrid(values, values)
            feed_dict['attribute_relation_' + attr_name] = (lhs == rhs).astype('float32').reshape(-1)

        nr_objects = len(scene['objects'])
        for attr_name, concepts in self.relational_concepts.items():
            concept_values = []
            for concept in concepts:
                values = np.zeros((nr_objects, nr_objects), dtype='float32')
                assert concept in scene['relationships']
                this_relation = scene['relationships'][concept]
                assert len(this_relation) == nr_objects
                for i, this_row in enumerate(this_relation):
                    for j in this_row:
                        values[i, j] = 1
                concept_values.append(values)
            concept_values = np.stack(concept_values, -1)
            feed_dict['relation_' + attr_name] = concept_values.reshape(-1, concept_values.shape[-1])

        return feed_dict

    def annotate_question_metainfo(self, metainfo):
        if 'template_filename' in metainfo:
            return dict(template=metainfo['template_filename'], template_index=metainfo['question_family_index'])
        return dict()

    def annotate_question(self, metainfo):
        return dict()

    def program_to_nsclseq(self, program, question=None):
        return clevr_to_nsclseq(program)

    def canonize_answer(self, answer, question_type):
        if answer in ('yes', 'no'):
            answer = (answer == 'yes')
        elif isinstance(answer, six.string_types) and answer.isdigit():
            answer = int(answer)
            assert 0 <= answer <= 10
        return answer

    def update_collate_guide(self, collate_guide):
        # Scene annotations.
        for attr_name in self.attribute_concepts:
            collate_guide['attribute_' + attr_name] = 'concat'
            collate_guide['attribute_relation_' + attr_name] = 'concat'
        for attr_name in self.relational_concepts:
            collate_guide['relation_' + attr_name] = 'concat'

        # From ExtractConceptsAndAttributes and SearchCandidatePrograms.
        for param_type in self.parameter_types:
            collate_guide['question_' + param_type + 's'] = 'skip'
        collate_guide['program_parserv1_groundtruth_qstree'] = 'skip'
        collate_guide['program_parserv1_candidates_qstree'] = 'skip'

