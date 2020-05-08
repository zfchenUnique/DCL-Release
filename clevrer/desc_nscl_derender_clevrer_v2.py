#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Derendering model for the Neuro-Symbolic Concept Learner.

Unlike the model in NS-VQA, the model receives only ground-truth programs and needs to execute the program
to get the supervision for the VSE modules. This model tests the implementation of the differentiable
(or the so-called quasi-symbolic) reasoning process.

"""

from jacinle.utils.container import GView
from nscl.models.utils import canonize_monitors, update_from_loss_module
from clevrer.models.reasoning_v2 import ReasoningV2ModelForCLEVRER, make_reasoning_v2_configs
from clevrer.utils import predict_future_feature

configs = make_reasoning_v2_configs()
configs.model.vse_known_belong = False
configs.train.scene_add_supervision = False
configs.train.qa_add_supervision = True
import pdb

class Model(ReasoningV2ModelForCLEVRER):
    def __init__(self, args):
        configs.rel_box_flag = args.rel_box_flag 
        configs.dynamic_ftr_flag = args.dynamic_ftr_flag 
        configs.train.scene_add_supervision = args.scene_add_supervision 
        self.args = args
        super().__init__(configs, args)

    def build_temporal_prediction_model(self, args, desc_pred):
        model_pred = desc_pred.PropagationNetwork(args, residual=True, use_gpu=True)
        self._model_pred = model_pred 

    def forward(self, feed_dict_list):
        if self.training:
            loss, monitors, outputs = self.forward_default(feed_dict_list)
            return loss, monitors, None
        else:
            outputs1 = self.forward_default(feed_dict_list)
            return outputs1 

    def forward_default(self, feed_dict_list):

        if isinstance(feed_dict_list, dict):
            feed_dict_list = [feed_dict_list]

        video_num = len(feed_dict_list)
        f_sng_list = []
        f_sng_future_list = []
        for vid, feed_dict in enumerate(feed_dict_list):
            f_scene = self.resnet(feed_dict['img'])
            f_sng = self.scene_graph(f_scene, feed_dict)
            f_sng_list.append(f_sng)
    
            if len(feed_dict['predictions']) >0 and self.args.version=='v2':
                f_scene_future = self.resnet(feed_dict['img_future']) 
                f_sng_future = self.scene_graph(f_scene_future, feed_dict, mode=1)
                f_sng_future_list.append(f_sng_future)
            elif self.args.version=='v3' and feed_dict['load_predict_flag']:
                f_sng_future = predict_future_feature(self, feed_dict, f_sng, self.args)
                f_sng_future_list.append(f_sng_future)
            else:
                f_sng_future_list.append(None)

        programs = []
        _ignore_list = []
        for idx, feed_dict in enumerate(feed_dict_list):
            tmp_ignore_list = []
            tmp_prog = []
            feed_dict['answer'] = [] 
            feed_dict['question_type'] = []
            feed_dict['question_type_new'] = []
            questions_info = feed_dict['meta_ann']['questions']
            for q_id, ques in enumerate(questions_info):

                if 'program_cl' in ques.keys():
                    tmp_prog.append(ques['program_cl'])
                if 'answer' in ques.keys():
                    feed_dict['answer'].append(ques['answer'])
                    feed_dict['question_type'].append(ques['program_cl'][-1]['op'])
                else:
                    tmp_answer_list = []
                    for choice_info in ques['choices']:
                        if choice_info['answer'] == 'wrong':
                            tmp_answer_list.append(False)
                        elif choice_info['answer'] == 'correct':
                            tmp_answer_list.append(True)
                        else:
                            pdb.set_trace()
                    feed_dict['answer'].append(tmp_answer_list)
                    feed_dict['question_type'].append(ques['program_cl'][-1]['op'])
                feed_dict['question_type_new'].append(ques['question_type'])
            programs.append(tmp_prog)
            _ignore_list.append(tmp_ignore_list)
        programs_list, buffers_list, answers_list = self.reasoning(f_sng_list, programs, \
                fd=feed_dict_list, future_features_list=f_sng_future_list, nscl_model=self, ignore_list = _ignore_list)
        monitors_list = [] 
        output_list = []
        for idx, buffers  in enumerate(buffers_list): 
            monitors, outputs = {}, {}
            outputs['buffers'] = buffers 
            outputs['answer'] = answers_list[idx] 
            feed_dict = feed_dict_list[idx]
            f_sng = [f_sng_list[idx]]
            answers = answers_list[idx]
            
            update_from_loss_module(monitors, outputs, self.scene_loss(
                feed_dict, f_sng,
                self.reasoning.embedding_attribute, self.reasoning.embedding_relation,
                self.reasoning.embedding_temporal 
            ))
            update_from_loss_module(monitors, outputs, self.qa_loss(feed_dict, answers))
            monitors_list.append(monitors)
            output_list.append(outputs)

        loss = 0
        loss_scene = 0
        if self.training:
            for monitors in monitors_list:
                qa_loss_list = [qa_loss[0] for qa_loss in monitors['loss/qa']] 
                qa_loss = sum(qa_loss_list)/(len(qa_loss_list)+0.000001)
                loss += qa_loss
                if self.args.scene_add_supervision:
                    loss_scene = self.args.scene_supervision_weight * monitors['loss/scene']
            return loss, monitors, outputs
        else:
            outputs['monitors'] = monitors_list 
            outputs['buffers'] = buffers_list 
            outputs['answer'] = buffers_list 
            #pdb.set_trace()
            return outputs


def make_model(args):
    return Model(args)
