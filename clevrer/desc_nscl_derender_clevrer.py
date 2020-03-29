#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Derendering model for the Neuro-Symbolic Concept Learner.

Unlike the model in NS-VQA, the model receives only ground-truth programs and needs to execute the program
to get the supervision for the VSE modules. This model tests the implementation of the differentiable
(or the so-called quasi-symbolic) reasoning process.

Note that, in order to train this model, one must use the curriculum learning.
"""

from jacinle.utils.container import GView
from nscl.models.utils import canonize_monitors, update_from_loss_module
from clevrer.models.reasoning_v1 import ReasoningV1ModelForCLEVRER, make_reasoning_v1_configs

configs = make_reasoning_v1_configs()
configs.model.vse_known_belong = False
configs.train.scene_add_supervision = False
configs.train.qa_add_supervision = True
import pdb

class Model(ReasoningV1ModelForCLEVRER):
    def __init__(self, vocab, args):
        configs.rel_box_flag = args.rel_box_flag 
        super().__init__(vocab, configs)

    def forward(self, feed_dict_list):
        #feed_dict = GView(feed_dict)

        video_num = len(feed_dict_list)
        f_sng_list = []
        for vid, feed_dict in enumerate(feed_dict_list):
            f_scene = self.resnet(feed_dict['img'])
            f_sng = self.scene_graph(f_scene, feed_dict)
            f_sng_list.append(f_sng)

        programs = []
        for idx, feed_dict in enumerate(feed_dict_list):
            tmp_prog = []
            feed_dict['answer'] = [] 
            feed_dict['question_type'] = []
            for ques in feed_dict['meta_ann']['questions']:
                if 'answer' not in ques.keys():
                    continue 
                if 'program_cl' in ques.keys():
                    tmp_prog.append(ques['program_cl'])
                feed_dict['answer'].append(ques['answer'])
                feed_dict['question_type'].append(ques['program_cl'][-1]['op'])
            programs.append(tmp_prog)
        programs_list, buffers_list, answers_list = self.reasoning(f_sng_list, programs, fd=feed_dict_list)
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
                self.reasoning.embedding_attribute, self.reasoning.embedding_relation
            ))
            update_from_loss_module(monitors, outputs, self.qa_loss(feed_dict, answers))
            canonize_monitors(monitors)
            monitors_list.append(monitors)
            output_list.append(outputs)

        loss = 0 
        if self.training:
            for monitors in monitors_list:
                loss += monitors['loss/qa']
                if configs.train.scene_add_supervision:
                    loss = loss + monitors['loss/scene']
            return loss/len(monitors_list), monitors, outputs
        else:
            outputs['monitors'] = monitors_list 
            outputs['buffers'] = buffers_list 
            outputs['answer'] = buffers_list 
            return outputs


def make_model(args, vocab):
    return Model(vocab, args)
