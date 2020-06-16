#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : losses.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/04/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

import os
import torch
import torch.nn.functional as F

from jacinle.utils.enum import JacEnum
from nscl.nn.losses import MultitaskLossBase
from nscl.datasets.definition import gdef
from clevrer.models.quasi_symbolic_v2 import fuse_box_ftr, fuse_box_overlap, do_apply_self_mask_3d, Gaussin_smooth    
import pdb
import torch.nn as nn
import copy
DEBUG_SCENE_LOSS = int(os.getenv('DEBUG_SCENE_LOSS', '0'))


__all__ = ['SceneParsingLoss', 'QALoss', 'ParserV1Loss']


class SceneParsingLoss(MultitaskLossBase):
    def __init__(self, used_concepts, add_supervision=False, args=None):
        super().__init__()
        self.used_concepts = used_concepts
        self.add_supervision = add_supervision
        self.args = args

    def compute_regu_loss_v2(self, pred_ftr_list, f_sng, feed_dict, monitors):
        ftr_loss = 0.0
        loss_list = []
        mse_loss = nn.MSELoss()
        assert len(pred_ftr_list[4]) == pred_ftr_list[0].shape[1] - self.args.n_his -1

        # masking  out objects that didn't show up in the first n_his frames
        valid_object_id_stack = pred_ftr_list[4]
        for ftr_id, tmp_ftr in enumerate(pred_ftr_list):
            obj_num = f_sng[0][3].shape[0]
            list_num = 128
            box_dim = 4
            frm_num = pred_ftr_list[0].shape[1]
            gt_list = [f_sng[0][3].view(obj_num, -1, box_dim)[:, feed_dict['tube_info']['frm_list'][idx]] for idx in range(frm_num) ]
            tmp_gt = torch.stack(gt_list, dim=1).view(obj_num, -1, box_dim)    
            invalid_mask = tmp_gt.sum(dim=2)==-2
            
            if tmp_ftr is None or isinstance(tmp_ftr, list):
                continue 
            if len(tmp_ftr.shape)==3:
                # tmp_ftr: (obj_num, frm_num , ftr_dim)
                frm_num = tmp_ftr.shape[1]
                tmp_gt = f_sng[0][ftr_id][:,:frm_num]
                for obj_id in range(invalid_mask.shape[0]):
                    for frm_id in range(invalid_mask.shape[1]):
                        if invalid_mask[obj_id, frm_id]:
                            tmp_ftr[obj_id, frm_id] = 0.0
                            tmp_gt[obj_id, frm_id] = 0.0
                
                for frm_idx, valid_obj_list in enumerate(valid_object_id_stack):
                    frm_id = self.args.n_his + 1 + frm_idx
                    for obj_id in range(obj_num):
                        if obj_id not in valid_obj_list:
                            tmp_ftr[obj_id, frm_id] = 0.0
                            tmp_gt[obj_id, frm_id] = 0.0


            elif len(tmp_ftr.shape)==4:
                frm_num = tmp_ftr.shape[2]
                tmp_gt = f_sng[0][ftr_id][:, :, :frm_num]
                for obj_id in range(invalid_mask.shape[0]):
                    for frm_id in range(invalid_mask.shape[1]):
                        if invalid_mask[obj_id, frm_id]:
                            tmp_ftr[obj_id, :, frm_id] = 0.0
                            tmp_ftr[:, obj_id, frm_id] = 0.0
                            tmp_gt[obj_id, :, frm_id] = 0.0
                            tmp_gt[:, obj_id, frm_id] = 0.0

                # tmp_ftr: (obj_num, obj_num, frm_num , ftr_dim)
                for frm_idx, valid_obj_list in enumerate(valid_object_id_stack):
                    frm_id = self.args.n_his + 1 + frm_idx
                    for obj_id in range(obj_num):
                        if obj_id not in valid_obj_list:
                            tmp_ftr[obj_id, :, frm_id] = 0.0
                            tmp_ftr[:, obj_id, frm_id] = 0.0
                            tmp_gt[obj_id, :, frm_id] = 0.0
                            tmp_gt[:, obj_id, frm_id] = 0.0


            elif len(tmp_ftr.shape)==2:
                frm_num = tmp_ftr.shape[1] // box_dim 
                gt_list = [f_sng[0][3].view(obj_num, -1, box_dim)[:, feed_dict['tube_info']['frm_list'][idx]] for idx in range(frm_num) ]
                tmp_gt = torch.stack(gt_list, dim=1).view(obj_num, -1, box_dim)    
                tmp_ftr = tmp_ftr.view(obj_num, -1, box_dim)    
                
                for obj_id in range(invalid_mask.shape[0]):
                    for frm_id in range(invalid_mask.shape[1]):
                        if invalid_mask[obj_id, frm_id]:
                            tmp_ftr[obj_id, frm_id, 2:] = 0.0
                            tmp_ftr[obj_id, frm_id, :2] = -1.0
                            tmp_gt[obj_id, frm_id, 2:] = 0.0
                            tmp_gt[obj_id, frm_id, :2] = -1.0
                    
                # tmp_ftr: (obj_num, frm_num , box_dim)
                for frm_idx, valid_obj_list in enumerate(valid_object_id_stack):
                    frm_id = self.args.n_his + 1 + frm_idx
                    for obj_id in range(obj_num):
                        if obj_id not in valid_obj_list:
                            tmp_ftr[obj_id, frm_id, 2:] = 0.0
                            tmp_ftr[obj_id, frm_id, :2] = -1.0
                            tmp_gt[obj_id, frm_id, 2:] = 0.0
                            tmp_gt[obj_id, frm_id, :2] = -1.0

            tmp_loss = mse_loss(tmp_ftr, tmp_gt)
            loss_list.append(tmp_loss)
            if ftr_id==0:
                monitors['loss/regu/obj_ftr'] = tmp_loss
            elif ftr_id ==2:
                monitors['loss/regu/rel_ftr'] = tmp_loss
            elif ftr_id ==3:
                monitors['loss/regu/obj_box'] = tmp_loss
        # adding rela loss
        if self.args.rela_dist_loss_flag==1:
            # 0:pred_frm
            rel_spa_pred = torch.stack(pred_ftr_list[-2], dim=0).squeeze(4).squeeze(3)
            # n_his+1:pred_frm
            rel_spa_gt = torch.stack(pred_ftr_list[-1], dim=0)
            assert  rel_spa_pred.shape[0]==rel_spa_gt.shape[0]+self.args.n_his +1
            pred_frm_num = rel_spa_pred.shape[0]
            x_step = self.args.n_his + 1
            tmp_loss = mse_loss(rel_spa_pred[x_step:pred_frm_num-1], rel_spa_gt[1:])
            #pdb.set_trace()
            monitors['loss/regu/rel_spa'] = tmp_loss
            loss_list.append(tmp_loss)

        ftr_loss = sum(loss_list)
        monitors['loss/regu'] = ftr_loss



    def compute_regu_loss(self, pred_ftr_list, f_sng, feed_dict, monitors):
        ftr_loss = 0.0
        loss_list = []
        mse_loss = nn.MSELoss()
        for ftr_id, tmp_ftr in enumerate(pred_ftr_list):
            obj_num = f_sng[0][3].shape[0]
            list_num = 128
            box_dim = 4
            frm_num = pred_ftr_list[0].shape[1]
            gt_list = [f_sng[0][3].view(obj_num, -1, box_dim)[:, feed_dict['tube_info']['frm_list'][idx]] for idx in range(frm_num) ]
            tmp_gt = torch.stack(gt_list, dim=1).view(obj_num, -1, box_dim)    
            invalid_mask = tmp_gt.sum(dim=2)==-2 
            if tmp_ftr is not None:
                if len(tmp_ftr.shape)==3:
                    frm_num = tmp_ftr.shape[1]
                    tmp_gt = f_sng[0][ftr_id][:,:frm_num]
                    for obj_id in range(invalid_mask.shape[0]):
                        for frm_id in range(invalid_mask.shape[1]):
                            if invalid_mask[obj_id, frm_id]:
                                tmp_ftr[obj_id, frm_id] = 0.0
                elif len(tmp_ftr.shape)==4:
                    frm_num = tmp_ftr.shape[2]
                    tmp_gt = f_sng[0][ftr_id][:, :, :frm_num]
                    for obj_id in range(invalid_mask.shape[0]):
                        for frm_id in range(invalid_mask.shape[1]):
                            if invalid_mask[obj_id, frm_id]:
                                tmp_ftr[obj_id, :, frm_id] = 0.0
                                tmp_ftr[:, obj_id, frm_id] = 0.0
                elif len(tmp_ftr.shape)==2:
                    frm_num = tmp_ftr.shape[1] // box_dim 
                    gt_list = [f_sng[0][3].view(obj_num, -1, box_dim)[:, feed_dict['tube_info']['frm_list'][idx]] for idx in range(frm_num) ]
                    tmp_gt = torch.stack(gt_list, dim=1).view(obj_num, -1, box_dim)    
                    tmp_ftr = tmp_ftr.view(obj_num, -1, box_dim)    
                    for obj_id in range(invalid_mask.shape[0]):
                        for frm_id in range(invalid_mask.shape[1]):
                            if invalid_mask[obj_id, frm_id]:
                                tmp_ftr[obj_id, frm_id, 2:] = 0.0
                                tmp_ftr[obj_id, frm_id, :2] = -1.0
                
                tmp_loss = mse_loss(tmp_ftr, tmp_gt)
                loss_list.append(tmp_loss)
                if ftr_id==0:
                    monitors['loss/regu/obj_ftr'] = tmp_loss
                elif ftr_id ==2:
                    monitors['loss/regu/rel_ftr'] = tmp_loss
                elif ftr_id ==3:
                    monitors['loss/regu/obj_box'] = tmp_loss
        ftr_loss = sum(loss_list)
        monitors['loss/regu'] = ftr_loss
        return monitors 

    def forward(self, feed_dict, f_sng, attribute_embedding, relation_embedding, temporal_embedding, buffer=None, pred_ftr_list=None):
        outputs, monitors = dict(), dict()

        if pred_ftr_list is not None:
            monitors = compute_regu_loss(pred_ftr_list, f_sng, feed_dict, monitors)

        objects = [f[1] for f in f_sng]
        all_f = torch.cat(objects)
        
        obj_box = [f[3] for f in f_sng]
        if self.args.apply_gaussian_smooth_flag:
            obj_box = [ Gaussin_smooth(f[3]) for f in f_sng]
        
        all_f_box = torch.cat(obj_box)

        for attribute, concepts in self.used_concepts['attribute'].items():
            if 'attribute_' + attribute not in feed_dict:
                continue

            all_scores = []
            for v in concepts:
                this_score = attribute_embedding.similarity(all_f, v)
                all_scores.append(this_score)

            all_scores = torch.stack(all_scores, dim=-1)
            #pdb.set_trace()
            all_labels = feed_dict['attribute_' + attribute]

            if all_labels.dim() == all_scores.dim() - 1:
                acc_key = 'acc/scene/attribute/' + attribute
                monitors[acc_key] = (
                    ((all_scores > 0).float().sum(dim=-1) == 1) *
                    (all_scores.argmax(-1) == all_labels.long())
                ).float().mean()

                if self.training and self.add_supervision:
                    this_loss = self._sigmoid_xent_loss(all_scores, all_labels.long())
                    if DEBUG_SCENE_LOSS and torch.isnan(this_loss).any():
                        print('NAN! in object_loss. Starting the debugger')
                        from IPython import embed; embed()
                    for loss_key in ['loss/scene/attribute/' + attribute, 'loss/scene']:
                        monitors[loss_key] = monitors.get(loss_key, 0) + this_loss
            else:
                acc_key = 'acc/scene/attribute/' + attribute
                monitors[acc_key] = (
                    (all_scores > 0).long() == all_labels.long()
                ).float().mean()

                if self.training and self.add_supervision:
                    this_loss = self._bce_loss(all_scores, all_labels.float())
                    if DEBUG_SCENE_LOSS and torch.isnan(this_loss).any():
                        print('NAN! in object_loss. Starting the debugger')
                        from IPython import embed; embed()
                    for loss_key in ['loss/scene/attribute/' + attribute, 'loss/scene']:
                        monitors[loss_key] = monitors.get(loss_key, 0) + this_loss


        for relation, concepts in self.used_concepts['relation'].items():
            for concept in concepts:
                if 'relation_' + concept not in feed_dict:
                    continue
                cross_scores = []
                for f in f_sng:
                    obj_num, ftr_dim = f[3].shape
                    box_dim = 4
                    smp_coll_frm_num = self.args.smp_coll_frm_num 
                    time_step = int(ftr_dim/box_dim) 
                    offset = time_step%smp_coll_frm_num 
                    seg_frm_num = int((time_step-offset)/smp_coll_frm_num) 
                    half_seg_frm_num = int(seg_frm_num/2)

                    frm_list = []
                    ftr = f[3].view(obj_num, time_step, box_dim)[:, :time_step-offset, :box_dim]
                    ftr = ftr.view(obj_num, smp_coll_frm_num, seg_frm_num*box_dim)
                    # N*N*smp_coll_frm_num*(seg_frm_num*box_dim*4)
                    rel_box_ftr = fuse_box_ftr(ftr)
                    # concatentate

                    if self.args.colli_ftr_type ==1:
                        vis_ftr_num = f[2].shape[2]
                        col_ftr_dim = f[2].shape[3]
                        off_set = smp_coll_frm_num % vis_ftr_num 
                        exp_dim = int(smp_coll_frm_num / vis_ftr_num )
                        exp_dim = max(1, exp_dim)
                        coll_ftr = torch.zeros(obj_num, obj_num, smp_coll_frm_num, col_ftr_dim, \
                                dtype=rel_box_ftr.dtype, device=rel_box_ftr.device)
                        coll_ftr_exp = f[2].unsqueeze(3).expand(obj_num, obj_num, vis_ftr_num, exp_dim, col_ftr_dim).contiguous()
                        coll_ftr_exp_view = coll_ftr_exp.view(obj_num, obj_num, vis_ftr_num*exp_dim, col_ftr_dim)
                        min_frm_num = min(vis_ftr_num*exp_dim, smp_coll_frm_num)
                        coll_ftr[:, :, :min_frm_num] = coll_ftr_exp_view[:,:, :min_frm_num] 
                        if vis_ftr_num*exp_dim<smp_coll_frm_num:
                            coll_ftr[:, :, vis_ftr_num*exp_dim:] = coll_ftr_exp_view[:,:, -1, :].unsqueeze(2) 
                        rel_ftr_norm = torch.cat([coll_ftr, rel_box_ftr], dim=-1)

                    elif not self.args.box_only_for_collision_flag:
                        col_ftr_dim = f[2].shape[2]
                        coll_ftr_exp = f[2].unsqueeze(2).expand(obj_num, obj_num, smp_coll_frm_num, col_ftr_dim)
                        rel_ftr_norm = torch.cat([coll_ftr_exp, rel_box_ftr], dim=-1)
                    else:
                        rel_ftr_norm =  rel_box_ftr 
                    if self.args.box_iou_for_collision_flag:
                        # N*N*time_step 
                        box_iou_ftr  = fuse_box_overlap(ftr.view(obj_num, -1))
                        box_iou_ftr_view = box_iou_ftr.view(obj_num, obj_num, smp_coll_frm_num, seg_frm_num)
                        rel_ftr_norm = torch.cat([rel_ftr_norm, box_iou_ftr_view], dim=-1)


                    coll_mat = relation_embedding.similarity_collision(rel_ftr_norm, concept)
                    coll_mat = 0.5 * (coll_mat + coll_mat.transpose(1, 0))
                    coll_mat = do_apply_self_mask_3d(coll_mat)
                    coll_mat_max, coll_mat_idx =  torch.max(coll_mat, dim=2)
                    cross_scores.append(coll_mat_max.view(-1))
                cross_scores = torch.cat(cross_scores)
                cross_labels = feed_dict['relation_' + concept].view(-1)

                acc_key = 'acc/scene/relation/' + concept
                monitors[acc_key] = ((cross_scores > 0).long() == cross_labels.long()).float().mean()
                acc_key_pos = 'acc/scene/relation/' + concept +'_pos'
                acc_key_neg = 'acc/scene/relation/' + concept +'_neg'
                acc_mat = ((cross_scores > self.args.colli_threshold).long() == cross_labels.long()).float()
                pos_acc = (acc_mat * cross_labels.float()).sum() / (cross_labels.float().sum()+ 0.000001)
                neg_acc = (acc_mat * (1- cross_labels.float())).sum() / ((1-cross_labels.float()).sum()+0.000001)
                monitors[acc_key_pos] = pos_acc 
                monitors[acc_key_neg] = neg_acc
                if self.training and self.add_supervision:
                    label_len = cross_labels.shape[0]
                    pos_num = cross_labels.sum().float()
                    neg_num = label_len - pos_num 
                    label_weight = [pos_num*1.0/label_len, neg_num*1.0/label_len]
                    this_loss = self._bce_loss(cross_scores, cross_labels.float(), label_weight)
                    if DEBUG_SCENE_LOSS and torch.isnan(this_loss).any():
                        print('NAN! in object_same_loss. Starting the debugger')
                        from IPython import embed; embed()
                    for loss_key in ['loss/scene/relation/' + concept, 'loss/scene']:
                        monitors[loss_key] = monitors.get(loss_key, 0) + this_loss

        for attribute, concepts in self.used_concepts['temporal'].items():
            
            if attribute != 'scene':
                continue
            for v in concepts:
                if 'temporal_' + v not in feed_dict:
                    continue
                this_score = temporal_embedding.similarity(all_f_box, v)

                if v =='in':
                    cross_labels = feed_dict['temporal_' + v]>0
                elif v =='out':
                    cross_labels = feed_dict['temporal_' + v]<128

                acc_key = 'acc/scene/temporal/' + v
                monitors[acc_key] = ((this_score > 0).long() == cross_labels.long()).float().mean()

                if self.training and self.add_supervision:
            
                    label_len = cross_labels.shape[0]
                    pos_num = cross_labels.sum().float()
                    neg_num = label_len - pos_num 
                    label_weight = [pos_num*1.0/label_len, neg_num*1.0/label_len]
                    this_loss = self._bce_loss(this_score, cross_labels.float(), label_weight)
                    if DEBUG_SCENE_LOSS and torch.isnan(this_loss).any():
                        print('NAN! in object_same_loss. Starting the debugger')
                        from IPython import embed; embed()
                    for loss_key in ['loss/scene/temporal/' + v, 'loss/scene']:
                        monitors[loss_key] = monitors.get(loss_key, 0) + this_loss
        return monitors, outputs


class QALoss(MultitaskLossBase):
    def __init__(self, add_supervision):
        super().__init__()
        self.add_supervision = add_supervision

    def forward(self, feed_dict, answers, question_index=None, loss_weights=None, accuracy_weights=None):
        """
        Args:
            feed_dict (dict): input feed dict.
            answers (list): answer derived from the reasoning module.
            question_index (list[int]): question index of the i-th answer.
            loss_weights (list[float]):
            accuracy_weights (list[float]):

        """

        monitors = {}
        outputs = {'answer': []}
            
        question_type_list = ['descriptive', 'explanatory', 'counterfactual', 'predictive']
        question_type_per_question_list = ['descriptive', 'explanatory', 'counterfactual', 'predictive']
        for query_type in question_type_list:
            monitors.setdefault('acc/qa/' + query_type, [])
            monitors.setdefault('loss/qa/' + query_type, [])

        for query_type in question_type_per_question_list:
            monitors.setdefault('acc/qa/' + query_type+'_per_ques', [])

        if 'answer' not in feed_dict or 'question_type' not in feed_dict:
            return monitors, outputs

        for i, tmp_answer in enumerate(answers):
            if tmp_answer is None:
                continue 
            query_type, a = tmp_answer 
            j = i if question_index is None else question_index[i]
            loss_w = loss_weights[i] if loss_weights is not None else 1
            acc_w = accuracy_weights[i] if accuracy_weights is not None else 1

            gt = feed_dict['answer'][j]
            response_query_type = gdef.qtype2atype_dict[query_type]

            question_type = feed_dict['question_type'][j]
            response_question_type = gdef.qtype2atype_dict[question_type]

            if response_question_type != response_query_type:
                key = 'acc/qa/' + query_type
                monitors.setdefault(key, []).append((0, acc_w))
                monitors.setdefault('acc/qa', []).append((0, acc_w))

                if self.training and self.add_supervision:
                    l = torch.tensor(10, dtype=torch.float, device=a[0].device if isinstance(a, tuple) else a.device)
                    monitors.setdefault('loss/qa/' + query_type, []).append((l, loss_w))
                    monitors.setdefault('loss/qa', []).append((l, loss_w))
                continue

            if response_query_type == 'word':
                a, word2idx = a
                argmax = a.argmax(dim=-1).item()
                idx2word = {v: k for k, v in word2idx.items()}
                outputs['answer'].append(idx2word[argmax])
                gt = word2idx[gt]
                loss = self._xent_loss
            elif response_query_type == 'bool':
                if isinstance(gt, list):
                    tmp_answer_list = []
                    for idx in range(len(gt)):
                        argmax = int((a[idx] > 0).item())
                        gt[idx] = int(gt[idx])
                        tmp_answer_list.append(argmax)
                    loss = self._bce_loss
                    outputs['answer'].append(tmp_answer_list)
                else:
                    argmax = int((a > 0).item())
                    outputs['answer'].append(argmax)
                    gt = int(gt)
                    loss = self._bce_loss
            elif response_query_type == 'integer':
                try:
                    argmax = int(round(a.item()))
                except ValueError:
                    argmax = 0
                outputs['answer'].append(argmax)
                gt = int(gt)
                loss = self._mse_loss
            else:
                raise ValueError('Unknown query type: {}.'.format(response_query_type))


            key = 'acc/qa/' + query_type
            question_type_new = feed_dict['question_type_new'][j]
            new_key = 'acc/qa/' + question_type_new            
           

            if isinstance(gt, list):
                for idx in range(len(gt)):
                    monitors.setdefault(key, []).append((int(gt[idx] == tmp_answer_list[idx]), acc_w))
                    monitors.setdefault('acc/qa', []).append((int(gt[idx] == tmp_answer_list[idx]), acc_w))
                    monitors.setdefault(new_key, []).append((int(gt[idx] == tmp_answer_list[idx]), acc_w))
                monitors.setdefault(new_key+'_per_ques', []).append((int(gt == tmp_answer_list), acc_w))
                #pdb.set_trace()
            else:
                monitors.setdefault(key, []).append((int(gt == argmax), acc_w))
                monitors.setdefault('acc/qa', []).append((int(gt == argmax), acc_w))
                monitors.setdefault(new_key, []).append((int(gt == argmax), acc_w))


            if self.training and self.add_supervision:
                if isinstance(gt, list):
                    for idx in range(len(gt)):
                        l = loss(a[idx], gt[idx])
                        monitors.setdefault('loss/qa/' + query_type, []).append((l, loss_w))
                        monitors.setdefault('loss/qa', []).append((l, loss_w))
                        monitors.setdefault('loss/qa/' + question_type_new, []).append((l, loss_w))
                else:
                    l = loss(a, gt)
                    monitors.setdefault('loss/qa/' + query_type, []).append((l, loss_w))
                    monitors.setdefault('loss/qa', []).append((l, loss_w))
                    monitors.setdefault('loss/qa/' + question_type_new, []).append((l, loss_w))
        return monitors, outputs

    def _gen_normalized_weights(self, weights, n):
        if weights is None:
            return [1 for _ in range(n)]
        sum_weights = sum(weights)
        return [w / sum_weights * n]


class ParserV1RewardShape(JacEnum):
    LOSS = 'loss'
    ACCURACY = 'accuracy'


class ParserV1Loss(MultitaskLossBase):
    def __init__(self, reward_shape='loss'):
        super().__init__()
        self.reward_shape = ParserV1RewardShape.from_string(reward_shape)

    def forward(self, feed_dict, programs_pd, accuracy, loss):
        batch_size = len(programs_pd)
        policy_loss = 0
        for i in range(len(feed_dict['question_raw'])):
            log_likelihood = [p['log_likelihood'] for p in programs_pd if i == p['scene_id']]
            if len(log_likelihood) == 0:
                continue
            log_likelihood = torch.stack(log_likelihood, dim=0)
            discounted_log_likelihood = [p['discounted_log_likelihood'] for p in programs_pd if i == p['scene_id']]
            discounted_log_likelihood = torch.stack(discounted_log_likelihood, dim=0)

            if self.reward_shape is ParserV1RewardShape.LOSS:
                # reward = -loss
                rewards = 10 - torch.stack([loss[j] for j, p in enumerate(programs_pd) if i == p['scene_id']], dim=0)
                likelihood = F.softmax(log_likelihood, dim=-1)
            elif self.reward_shape is ParserV1RewardShape.ACCURACY:
                rewards = torch.tensor([accuracy[j] for j, p in enumerate(programs_pd) if i == p['scene_id']]).to(discounted_log_likelihood)
                likelihood = F.softmax(log_likelihood * rewards + -1e6 * (1 - rewards), dim=-1)

            # \Pr[p] * reward * \nabla \log \Pr[p]
            policy_loss += (-(likelihood * rewards).detach() * discounted_log_likelihood).sum()
        return {'loss/program': policy_loss}, dict()

