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
from clevrer.utils import predict_future_feature, predict_future_feature_v2, predict_normal_feature_v2, predict_normal_feature_v3, predict_normal_feature_v4, predict_normal_feature_v5, predict_future_feature_v5    

configs = make_reasoning_v2_configs()
configs.model.vse_known_belong = False
configs.train.scene_add_supervision = False
configs.train.qa_add_supervision = True
import pdb
import torch
import copy

class Model(ReasoningV2ModelForCLEVRER):
    def __init__(self, args):
        configs.rel_box_flag = args.rel_box_flag 
        configs.dynamic_ftr_flag = args.dynamic_ftr_flag 
        configs.train.scene_add_supervision = args.scene_add_supervision 
        self.args = args
        super().__init__(configs, args)

    def make_relation_embedding_for_unseen_events(self, args): 
        if args.version=='v2_1':
            # deep copy collision event embedding for counterafact and future embedding this version 
            tax = self.reasoning.embedding_relation  
            embedding_relation_counterfact = copy.deepcopy(tax)
            embedding_relation_future = copy.deepcopy(tax)
            setattr(self.reasoning, 'embedding_relation_counterfact', embedding_relation_counterfact)
            setattr(self.reasoning, 'embedding_relation_future', embedding_relation_future)
        #pdb.set_trace()


    def build_temporal_prediction_model(self, args, desc_pred, desc_spatial_pred=None):
        if args.version=='v3':
            model_pred = desc_pred.PropagationNetwork(args, residual=True, use_gpu=True)
            self._model_pred = model_pred 
        elif args.version=='v4':
            state_dim_bp = args.state_dim
            args.state_dim = 256
            model_pred = desc_pred.PropagationNetwork(args, residual=True, use_gpu=True)
            self._model_pred = model_pred 
            
            relation_dim_bp = args.relation_dim
            box_only_flag_bp = args.box_only_flag

            args.relation_dim = 3
            args.state_dim = 4
            args.box_only_flag = 1 

            model_spatial_pred = desc_spatial_pred.PropagationNetwork(args, residual=True, use_gpu=True)
            self._model_spatial_pred = model_spatial_pred 
            
            args.state_dim = state_dim_bp 
            args.box_only_flag = box_only_flag_bp 
            args.relation_dim = relation_dim_bp

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

        #pdb.set_trace()

        video_num = len(feed_dict_list)
        f_sng_list = []
        f_sng_future_list = []
        for vid, feed_dict in enumerate(feed_dict_list):
            f_scene = self.resnet(feed_dict['img'])
            f_sng = self.scene_graph(f_scene, feed_dict)
            f_sng_list.append(f_sng)
    
            if len(feed_dict['predictions']) >0 and (self.args.version=='v2' or self.args.version=='v2_1'):
                f_scene_future = self.resnet(feed_dict['img_future']) 
                f_sng_future = self.scene_graph(f_scene_future, feed_dict, mode=1)
                f_sng_future_list.append(f_sng_future)
            #elif self.args.version=='v3' and feed_dict['load_predict_flag'] and self.args.regu_only_flag!=1:
            elif self.args.version=='v3' and self.args.regu_only_flag!=1:
                f_sng_future = predict_future_feature_v2(self, feed_dict, f_sng, self.args)
                f_sng_future_list.append(f_sng_future)
            #elif self.args.regu_only_flag==1 and not self.training and self.args.visualize_flag:
            elif self.args.version=='v4' and self.args.regu_only_flag!=1:
                f_sng_future = predict_future_feature_v5(self, feed_dict, f_sng, self.args)
                f_sng_future_list.append(f_sng_future)

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
        
        if self.args.regu_only_flag !=1:
            programs_list, buffers_list, answers_list = self.reasoning(f_sng_list, programs, \
                    fd=feed_dict_list, future_features_list=f_sng_future_list, nscl_model=self, ignore_list = _ignore_list)
        else:
            buffers_list = []
            answers_list = []

        if self.args.regu_flag ==1:
            output_ftr_list = []
            for vid in range(len(feed_dict_list)):
                if self.training:
                    if self.args.version=='v3':
                        output_pred_ftr = predict_normal_feature_v4(self, feed_dict_list[vid], f_sng_list[vid], self.args)
                    elif self.args.version=='v4':
                        output_pred_ftr = predict_normal_feature_v5(self, feed_dict_list[vid], f_sng_list[vid], self.args)
                    #output_pred_ftr = predict_normal_feature_v3(self, feed_dict_list[vid], f_sng_list[vid], self.args)
                    #output_pred_ftr = predict_normal_feature_v2(self, feed_dict_list[vid], f_sng_list[vid], self.args)
                else:
                    #output_pred_ftr = predict_normal_feature_v2(self, feed_dict_list[vid], f_sng_list[vid], self.args)
                    if self.args.version=='v3':
                        output_pred_ftr = predict_normal_feature_v4(self, feed_dict_list[vid], f_sng_list[vid], self.args)
                    else:
                        output_pred_ftr = predict_normal_feature_v5(self, feed_dict_list[vid], f_sng_list[vid], self.args)
                output_ftr_list.append(output_pred_ftr)
        else:
            output_ftr_list = None

        monitors_list = [] 
        output_list = []
        ftr_decoder = self._decoder if self.args.reconstruct_flag==1 else None
        if self.args.regu_only_flag!=1:
            for idx, buffers  in enumerate(buffers_list):
                monitors, outputs = {}, {}
                outputs['buffers'] = buffers 
                outputs['answer'] = answers_list[idx] 
                feed_dict = feed_dict_list[idx]
                f_sng = [f_sng_list[idx]]
                answers = answers_list[idx]
               
                if output_ftr_list is not None:
                    output_ftr = output_ftr_list[idx]
                else:
                    output_ftr = None

                update_from_loss_module(monitors, outputs, self.scene_loss(
                    feed_dict, f_sng,
                    self.reasoning.embedding_attribute, self.reasoning.embedding_relation,
                    self.reasoning.embedding_temporal,
                    pred_ftr_list = output_ftr,
                    decoder = ftr_decoder 
                ))
                update_from_loss_module(monitors, outputs, self.qa_loss(feed_dict, answers, 
                    result_save_path=self.args.expression_result_path))
                monitors_list.append(monitors)
                output_list.append(outputs)

        elif self.args.regu_only_flag==1:
            for idx, output_ftr in enumerate(output_ftr_list):
                monitors = {}
                feed_dict = feed_dict_list[idx]
                f_sng = [f_sng_list[idx]]
                self.scene_loss.compute_regu_loss_v2(output_ftr, f_sng, feed_dict, monitors, self.reasoning.embedding_relation)
                monitors_list.append(monitors)
                if self.args.reconstruct_flag:
                    monitors = self.scene_loss.compute_reconstruction_loss(output_ftr, f_sng, feed_dict, monitors, decoder=ftr_decoder)
                monitors_list.append(monitors)

        loss = 0
        loss_scene = 0
        if self.training:
            for monitors in monitors_list:
                if self.args.regu_only_flag!=1:
                    qa_loss_list = [qa_loss[0] for qa_loss in monitors['loss/qa']] 
                    qa_loss = sum(qa_loss_list)/(len(qa_loss_list)+0.000001)
                    loss += qa_loss
                    if self.args.scene_add_supervision:
                        loss_scene = self.args.scene_supervision_weight * monitors['loss/scene']
                        loss +=loss_scene
                    if self.args.regu_flag:
                        loss_regu = self.args.regu_weight * monitors['loss/regu']
                        loss +=loss_regu
                elif self.args.regu_only_flag==1:
                    loss_regu = self.args.regu_weight * monitors['loss/regu']
                    loss +=loss_regu
                    outputs = {}
                if self.args.add_kl_regu_flag:
                    loss_kl = self.args.kl_weight * monitors['loss/kl']
                    loss +=loss_kl
                if self.args.reconstruct_flag:
                    loss_decode = self.args.reconstruct_weight  * monitors['loss/decode']
                    loss +=loss_decode 
                #pdb.set_trace()
            if torch.isnan(loss):
                pdb.set_trace()
            return loss, monitors, outputs
        else:
            outputs = {}
            outputs['monitors'] = monitors_list 
            outputs['buffers'] = buffers_list 
            outputs['answer'] = answers_list  
            return outputs


def make_model(args):
    return Model(args)
