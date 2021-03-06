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
import numpy as np
from .models import functional
import jactorch
from .utils import compute_LS, compute_IoU_v2, compute_union_box,pickledump, pickleload  

DEBUG_SCENE_LOSS = int(os.getenv('DEBUG_SCENE_LOSS', '0'))


__all__ = ['SceneParsingLoss', 'QALoss', 'ParserV1Loss']

def get_in_out_frame(all_f_box, concept):
    event_frm = []
    obj_num, ftr_dim = all_f_box.shape
    box_dim = 4 
    box_thre = 0.0001
    min_frm = 5
    time_step = ftr_dim // box_dim  
    for tar_obj_id in range(obj_num):
        c = concept 
        tar_ftr = all_f_box[tar_obj_id].view(time_step, box_dim)
        tar_area = tar_ftr[:, 2] * tar_ftr[:, 3]
        if c=='in':
            for t_id in range(time_step):
                end_id = min(t_id + min_frm, time_step-1)
                if torch.sum(tar_area[t_id:end_id]>box_thre)>=(end_id-t_id) and torch.sum(tar_ftr[t_id:end_id,2])>0:
                    event_frm.append(t_id)
                    break 
            if t_id== time_step - 1:
                event_frm.append(0)

        elif c=='out':
            for t_id in range(time_step-1, -1, -1):
                st_id = max(t_id - min_frm, 0)
                if torch.sum(tar_area[st_id:t_id]>box_thre)>=(t_id-st_id) and torch.sum(tar_ftr[st_id:t_id])>0:
                    event_frm.append(t_id)
                    break
            if t_id == 0:
                event_frm.append(time_step - 1)
    return event_frm 

def further_prepare_for_moving_stationary(ftr_ori, time_mask, concept):
    obj_num, ftr_dim = ftr_ori.shape 
    box_dim = 4
    time_step = int(ftr_dim/box_dim)
    if time_mask is not None and time_mask.sum()<=1:
        max_idx = torch.argmax(time_mask)
        st_idx = max(int(max_idx-time_win*0.5), 0)
        ed_idx = min(int(max_idx+time_win*0.5), time_step-1)
        time_mask[st_idx:ed_idx] = 1
    #assert time_mask is not None
    if time_mask is not None:
        ftr_mask = ftr_ori.view(obj_num, time_step, box_dim) * time_mask.view(1, time_step, 1)
    else:
        ftr_mask = ftr_ori.view(obj_num, time_step, box_dim)
    ftr_diff = torch.zeros(obj_num, time_step, box_dim, dtype=ftr_ori.dtype,
            device=ftr_ori.device)
    ftr_diff[:, :time_step-1, :] = ftr_mask[:, 0:time_step-1, :] - ftr_mask[:, 1:time_step, :]
    st_idx = 0; ed_idx = time_step - 1
    if time_mask is not None:
        for idx in range(time_step):
            if time_mask[idx]>0:
                st_idx = idx -1 if (idx-1)>=0 else idx
                break 
        for idx in range(time_step-1, -1, -1):
            if time_mask[idx]>0:
                ed_idx = idx if idx>=0 else 0
                break 
    ftr_diff[:, st_idx, :] = 0
    ftr_diff[:, ed_idx, :] = 0
    ftr_diff = ftr_diff.view(obj_num, ftr_dim)
    return ftr_diff 

def compute_kl_regu_loss(mean, var):
    kl_loss = -0.5*torch.sum(1+torch.log(var)-mean.pow(2)-var)
    return kl_loss 

def get_collision_embedding(tmp_ftr, f_sng, args, relation_embedding):
    obj_num, ftr_dim = f_sng[3].shape
    box_dim = 4
    time_step = int(ftr_dim/box_dim) 
    seg_frm_num = 4 
    half_seg_frm_num = int(seg_frm_num/2)
    frm_list = []
    smp_coll_frm_num = tmp_ftr.shape[2]
    ftr = f_sng[3].view(obj_num, time_step, box_dim)[:, :smp_coll_frm_num*seg_frm_num, :]
    ftr = ftr.view(obj_num, smp_coll_frm_num, seg_frm_num*box_dim)
    # N*N*smp_coll_frm_num*(seg_frm_num*box_dim*4)
    rel_box_ftr = fuse_box_ftr(ftr)
    # concatentate
    if args.colli_ftr_type ==1:
        vis_ftr_num = smp_coll_frm_num 
        col_ftr_dim = tmp_ftr.shape[3]
        off_set = smp_coll_frm_num % vis_ftr_num 
        exp_dim = int(smp_coll_frm_num / vis_ftr_num )
        exp_dim = max(1, exp_dim)
        coll_ftr = torch.zeros(obj_num, obj_num, smp_coll_frm_num, col_ftr_dim, \
                dtype=rel_box_ftr.dtype, device=rel_box_ftr.device)
        coll_ftr_exp = tmp_ftr.unsqueeze(3).expand(obj_num, obj_num, vis_ftr_num, exp_dim, col_ftr_dim).contiguous()
        coll_ftr_exp_view = coll_ftr_exp.view(obj_num, obj_num, vis_ftr_num*exp_dim, col_ftr_dim)
        min_frm_num = min(vis_ftr_num*exp_dim, smp_coll_frm_num)
        coll_ftr[:, :, :min_frm_num] = coll_ftr_exp_view[:,:, :min_frm_num] 
        if vis_ftr_num*exp_dim<smp_coll_frm_num:
            coll_ftr[:, :, -1*off_set:] = coll_ftr_exp_view[:,:, -1, :].unsqueeze(2) 
        rel_ftr_norm = torch.cat([coll_ftr, rel_box_ftr], dim=-1)
    else:
        raise NotImplemented 

    if args.box_iou_for_collision_flag:
        # N*N*time_step 
        box_iou_ftr  = fuse_box_overlap(ftr.view(obj_num, -1))
        box_iou_ftr_view = box_iou_ftr.view(obj_num, obj_num, smp_coll_frm_num, seg_frm_num)
        rel_ftr_norm = torch.cat([rel_ftr_norm, box_iou_ftr_view], dim=-1)

    mappings = relation_embedding.get_all_attributes()
    # shape: [batch, attributes, channel] or [attributes, channel]
    query_mapped = torch.stack([m(rel_ftr_norm) for m in mappings], dim=-2)
    query_mapped = query_mapped / query_mapped.norm(2, dim=-1, keepdim=True)
    return query_mapped 

class SceneParsingLoss(MultitaskLossBase):
    def __init__(self, used_concepts, add_supervision=False, args=None):
        super().__init__()
        self.used_concepts = used_concepts
        self.add_supervision = add_supervision
        self.args = args

    def forward(self, feed_dict, f_sng, attribute_embedding, relation_embedding, temporal_embedding, buffer=None, pred_ftr_list=None, decoder=None, result_save_path=''):
        outputs, monitors = dict(), dict()

        valid_obj_id_list = []
        for obj_id in range(len(feed_dict['tube_info'])-2):
            valid_flag = len(feed_dict['tube_info'][obj_id]['boxes'])>0
            valid_obj_id_list.append(valid_flag)
        obj_num = len(feed_dict['tube_info']) -2 

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

            all_scores = all_scores[valid_obj_id_list]
            all_labels = all_labels[valid_obj_id_list] 

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
                cross_indexes = []
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
                    cross_indexes.append(coll_mat_idx.view(-1))
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
                
                colli_label_frms = feed_dict['relation_'+concept+'_frame'].view(-1)
                colli_pred_idx = torch.cat(cross_indexes) 
                n_obj_2 = cross_labels.shape[0]
                frm_diff_list = []
                for n_idx in range(n_obj_2):
                    if cross_scores[n_idx]>0 and cross_labels[n_idx]>0:
                        pred_idx = colli_pred_idx[n_idx]
                        frm_gt = colli_label_frms[n_idx] 
                        pred_frm = feed_dict['tube_info']['frm_list'][pred_idx]
                        frm_diff = abs(pred_frm - frm_gt)
                        frm_diff_list.append(frm_diff)
                acc_key = 'acc/scene/relation/frmDiff/' + concept
                if len(frm_diff_list)>0:
                    monitors[acc_key] = (sum(frm_diff_list) / len(frm_diff_list)).float()

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
            
            #if attribute != 'scene':
            #    continue
            if attribute !='event2' and attribute !='status':
                continue 
            for v in concepts:
                if 'temporal_' + v not in feed_dict:
                    continue
                if v =='in':
                    cross_labels = feed_dict['temporal_' + v]>0
                    this_score = temporal_embedding.similarity(all_f_box, v)
                elif v =='out':
                    cross_labels = feed_dict['temporal_' + v]<128
                    this_score = temporal_embedding.similarity(all_f_box, v)
                elif v=='moving' or v=='falling':
                    cross_labels = feed_dict['temporal_' + v]>0
                    if self.args.diff_for_moving_stationary_flag:
                        all_f_box_mv = further_prepare_for_moving_stationary(all_f_box, time_mask=None, concept=v)
                    else:
                        all_f_box_mv = all_f_box 
                    obj_num = all_f_box_mv.shape[0] 
                    valid_seq_mask = torch.zeros(obj_num, 128, 1).to(all_f_box_mv.device)
                    time_step = valid_seq_mask.shape[1]
                    box_dim = 4
                    valid_len = feed_dict['valid_seq_mask'].shape[1]
                    valid_seq_mask[:, :valid_len, 0] = torch.from_numpy(feed_dict['valid_seq_mask']).float()
                    all_f_box_mv = all_f_box_mv.view(obj_num, time_step, box_dim) * valid_seq_mask - (1-valid_seq_mask)

                    all_f_box_mv = all_f_box_mv.view(obj_num, -1)
                    this_score = temporal_embedding.similarity(all_f_box_mv, v)
                elif v=='stationary':
                    cross_labels = feed_dict['temporal_' + v]>0
                    if self.args.diff_for_moving_stationary_flag:
                        all_f_box_mv = further_prepare_for_moving_stationary(all_f_box, time_mask=None, concept=v)
                    else:
                        all_f_box_mv = all_f_box 
                    obj_num = all_f_box_mv.shape[0] 
                    valid_seq_mask = torch.zeros(obj_num, 128, 1).to(all_f_box_mv.device)
                    time_step = valid_seq_mask.shape[1]
                    box_dim = 4
                    valid_len = feed_dict['valid_seq_mask'].shape[1]
                    valid_seq_mask[:, :valid_len, 0] = torch.from_numpy(feed_dict['valid_seq_mask']).float()
                    all_f_box_mv = all_f_box_mv.view(obj_num, time_step, box_dim) * valid_seq_mask - (1-valid_seq_mask)
                    all_f_box_mv = all_f_box_mv.view(obj_num, -1)
                    this_score = temporal_embedding.similarity(all_f_box_mv, v)
                this_score = this_score[valid_obj_id_list]
                cross_labels = cross_labels[valid_obj_id_list]
                
                acc_key_pos = 'acc/scene/temporal/' + v +'_pos'
                acc_key_neg = 'acc/scene/temporal/' + v +'_neg'
                cross_scores = this_score 
                acc_mat = ((cross_scores > self.args.obj_threshold).long() == cross_labels.long()).float()
                if cross_labels.float().sum()>0:
                    pos_acc = (acc_mat * cross_labels.float()).sum() / (cross_labels.float().sum()+ 0.000001)
                    monitors[acc_key_pos] = pos_acc 
                if (1-cross_labels.float()).sum()>0:
                    neg_acc = (acc_mat * (1- cross_labels.float())).sum() / ((1-cross_labels.float()).sum()+0.000001)
                    monitors[acc_key_neg] = neg_acc
                acc_key = 'acc/scene/temporal/' + v
                monitors[acc_key] = ((this_score > 0).long() == cross_labels.long()).float().mean()
                #pdb.set_trace()
                if v=='in' or v=='out':
                    tar_frm_list = get_in_out_frame(all_f_box, v)
                    frm_diff_list = []
                    for obj_id in range(all_f_box.shape[0]):
                        if this_score[obj_id]>0 and cross_labels[obj_id]>0:
                            frm_diff = abs(tar_frm_list[obj_id] - feed_dict['temporal_'+v][obj_id])
                            frm_diff_list.append(frm_diff)
                    acc_key = 'acc/scene/temporal/frmDiff/' + v
                    if len(frm_diff_list)>0:
                        monitors[acc_key] =  (sum(frm_diff_list) / len(frm_diff_list)).float()
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
    def __init__(self, add_supervision, args):
        super().__init__()
        self.add_supervision = add_supervision
        self.args = args

    def forward(self, feed_dict, answers, question_index=None, loss_weights=None, accuracy_weights=None, ground_thre=0.5, result_save_path=''):
        """
        Args:
            feed_dict (dict): input feed dict.
            answers (list): answer derived from the reasoning module.
            question_index (list[int]): question index of the i-th answer.
            loss_weights (list[float]):
            accuracy_weights (list[float]):
            ground_thre (float): threshold for video grounding 

        """

        monitors = {}
        outputs = {'answer': []}
            
        question_type_list = ['descriptive', 'explanatory', 'counterfactual', 'predictive', 'expression', 'retrieval']
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

            if len(feed_dict['answer'])>0:
                gt = feed_dict['answer'][j]
            else:
                gt = None
            response_query_type = gdef.qtype2atype_dict[query_type]

            question_type = feed_dict['question_type'][j]
            response_question_type = gdef.qtype2atype_dict[question_type]
            question_type_new = feed_dict['question_type_new'][j]
            question_sub_type = feed_dict['meta_ann']['questions'][j]['question_subtype']

            if response_question_type != response_query_type and (question_type_new!='retrieval' or a!='error'):
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
                if gt is not None:
                    gt = word2idx[gt]
                loss = self._xent_loss
            elif response_query_type == 'bool':
                if isinstance(a, list):
                    tmp_answer_list = []
                    for idx in range(len(a)):
                        argmax = int((a[idx] > 0).item())
                        if gt is not None:
                            gt[idx] = int(gt[idx])
                        tmp_answer_list.append(argmax)
                    loss = self._bce_loss
                    outputs['answer'].append(tmp_answer_list)
                else:
                    argmax = int((a > self.args.obj_threshold).item())
                    outputs['answer'].append(argmax)
                    if gt is not None:
                        gt = int(gt)
                    loss = self._bce_loss
            elif response_query_type == 'integer':
                try:
                    argmax = int(round(a.item()))
                except ValueError:
                    argmax = 0
                outputs['answer'].append(argmax)
                if gt is not None:
                    gt = int(gt)
                loss = self._mse_loss

            elif question_type_new=='expression' and question_sub_type.startswith('object'):
                if isinstance(a, tuple):
                    a = a[0]
                prp_idx = torch.argmax(a) 
                prp_tube = feed_dict['meta_ann']['tubePrp'][prp_idx]
                gt_tube = feed_dict['meta_ann']['tubeGt'][gt]
                overlap = compute_LS(prp_tube, gt_tube)
           
            elif question_type_new=='retrieval' and question_sub_type.startswith('object'):
                if isinstance(a, str) and a=='error':
                    prp_score = -1
                else:
                    prp_score = torch.max(a)
                correct_flag = 0
                if i in feed_dict['meta_ann']['pos_id_list'] and prp_score>0:
                    correct_flag =1
                elif i not in feed_dict['meta_ann']['pos_id_list'] and prp_score<0:
                    correct_flag =1
                pos_sample = 0
                if i in feed_dict['meta_ann']['pos_id_list']:
                    pos_sample =1

            elif question_type_new=='retrieval' and \
                    (question_sub_type.startswith('event_in') or question_sub_type.startswith('event_out')):
                if isinstance(a, str) and a=='error':
                    prp_score = -1
                else:
                    prp_score = torch.max(a[0])
                correct_flag = 0
                if i in feed_dict['meta_ann']['pos_id_list'] and prp_score>0:
                    correct_flag =1
                elif i not in feed_dict['meta_ann']['pos_id_list'] and prp_score<0:
                    correct_flag =1
                pos_sample = 0
                if i in feed_dict['meta_ann']['pos_id_list']:
                    pos_sample =1
            elif question_type_new=='retrieval' and \
                    question_sub_type.startswith('event_collision'):
                if isinstance(a, str) and a=='error':
                    prp_score = -1
                else:
                    prp_score = torch.max(a[0])
                correct_flag = 0
                if i in feed_dict['meta_ann']['pos_id_list'] and prp_score>0:
                    correct_flag =1
                elif i not in feed_dict['meta_ann']['pos_id_list'] and prp_score<0:
                    correct_flag =1
                pos_sample = 0
                if i in feed_dict['meta_ann']['pos_id_list']:
                    pos_sample =1

            elif question_type_new=='expression' and \
                    (question_sub_type.startswith('event_in') or question_sub_type.startswith('event_out')):
                prp_idx = int(torch.argmax(a[0]))
                prp_frm_id = int(a[2][prp_idx])
                prp_frm_len = len(feed_dict['meta_ann']['tubePrp'][prp_idx])
                if prp_frm_id>=prp_frm_len:
                    prp_frm_id = prp_frm_len - 1
                prp_box = feed_dict['meta_ann']['tubePrp'][prp_idx][prp_frm_id]
                gt_idx = gt['object']
                gt_frm_id = gt['frame']
                gt_frm_len = len(feed_dict['meta_ann']['tubeGt'][gt_idx])
                if gt_frm_id>=gt_frm_len:
                    gt_frm_id = gt_frm_len - 1
                gt_box = feed_dict['meta_ann']['tubeGt'][gt_idx][gt_frm_id]
                overlap = compute_IoU_v2(prp_box, gt_box)
                frm_dist = abs(gt_frm_id-prp_frm_id)
            
            elif question_type_new=='expression' and question_sub_type.startswith('event_collision'):
                #pdb.set_trace()
                flatten_idx = torch.argmax(a[0])
                obj_num = int(a[0].shape[0])
                obj_idx1, obj_idx2 = flatten_idx //obj_num, flatten_idx%obj_num 
                prp_frm_id = a[1][obj_idx1, obj_idx2]
                test_frm_list = a[2]
                img_frm_idx = test_frm_list[prp_frm_id]
                prp_box1 = feed_dict['meta_ann']['tubePrp'][obj_idx1][img_frm_idx]
                prp_box2 = feed_dict['meta_ann']['tubePrp'][obj_idx2][img_frm_idx]
                prp_union_box = compute_union_box(prp_box1, prp_box2) 

                gt_idx1, gt_idx2 = gt['object']
                gt_frm_id = gt['frame']
                gt_box1 = feed_dict['meta_ann']['tubeGt'][gt_idx1][gt_frm_id]
                gt_box2 = feed_dict['meta_ann']['tubeGt'][gt_idx2][gt_frm_id]
                gt_union_box = compute_union_box(gt_box1, gt_box2) 
                
                overlap = compute_IoU_v2(prp_union_box, gt_union_box)
                frm_dist = abs(gt_frm_id-img_frm_idx)

            else:
                raise ValueError('Unknown query type: {}.'.format(response_query_type))

            key = 'acc/qa/' + query_type
            new_key = 'acc/qa/' + question_type_new            

            if gt is not None and isinstance(gt, list) and question_type_new!='retrieval':
                for idx in range(len(a)):
                    monitors.setdefault(key, []).append((int(gt[idx] == tmp_answer_list[idx]), acc_w))
                    monitors.setdefault('acc/qa', []).append((int(gt[idx] == tmp_answer_list[idx]), acc_w))
                    monitors.setdefault(new_key, []).append((int(gt[idx] == tmp_answer_list[idx]), acc_w))
                monitors.setdefault(new_key+'_per_ques', []).append((int(gt == tmp_answer_list), acc_w))
            elif question_type_new=='descriptive' or question_type_new=='explanatory':
                if gt is not None:
                    monitors.setdefault(key, []).append((int(gt == argmax), acc_w))
                    monitors.setdefault('acc/qa', []).append((int(gt == argmax), acc_w))
                    monitors.setdefault(new_key, []).append((int(gt == argmax), acc_w))
            
            elif question_type_new=='expression' and question_sub_type.startswith('object'):
                new_key_v2 = 'acc/mIoU/' + question_sub_type             
                new_key_v3 = 'acc/mIoU/' + question_type_new             
                monitors.setdefault(key, []).append((int(overlap>=ground_thre), acc_w))
                monitors.setdefault('acc/qa', []).append((int(overlap>=ground_thre), acc_w))
                monitors.setdefault(new_key, []).append((int(overlap>=ground_thre), acc_w))
                monitors.setdefault(new_key_v2, []).append((overlap, acc_w))
                monitors.setdefault(new_key_v3, []).append((overlap, acc_w))
            elif question_type_new=='expression' and question_sub_type.startswith('event'):
                new_key_v2 = 'acc/mIoU/' + question_sub_type           
                new_key_v3 = 'acc/frmDist/' + question_sub_type            
                monitors.setdefault(new_key_v2, []).append((overlap, acc_w))
                monitors.setdefault(new_key_v3, []).append((frm_dist, acc_w))
            
            elif question_type_new=='retrieval' and question_sub_type.startswith('object'):
                new_key1 = 'acc/video'            
                new_key2 = 'acc/text'           
                new_key3 = 'acc/video/' + question_sub_type             
                new_key4 = 'acc/text/' + question_sub_type            
                monitors.setdefault(new_key1, []).append((correct_flag, acc_w))
                monitors.setdefault(new_key2, []).append((correct_flag, acc_w))
                monitors.setdefault(new_key3, []).append((correct_flag, acc_w))
                monitors.setdefault(new_key4, []).append((correct_flag, acc_w))
                monitors.setdefault(key, []).append((correct_flag, acc_w))

                if pos_sample==1:
                    monitors.setdefault(new_key1+'/pos', []).append((correct_flag, acc_w))
                    monitors.setdefault(new_key2+'/pos', []).append((correct_flag, acc_w))
                else:
                    monitors.setdefault(new_key1+'/neg', []).append((correct_flag, acc_w))
                    monitors.setdefault(new_key2+'/neg', []).append((correct_flag, acc_w))

            elif question_type_new=='retrieval' and question_sub_type.startswith('event'):
                new_key1 = 'acc/video'            
                new_key2 = 'acc/text'           
                new_key3 = 'acc/video/' + question_sub_type             
                new_key4 = 'acc/text/' + question_sub_type            
                monitors.setdefault(new_key1, []).append((correct_flag, acc_w))
                monitors.setdefault(new_key2, []).append((correct_flag, acc_w))
                monitors.setdefault(new_key3, []).append((correct_flag, acc_w))
                monitors.setdefault(new_key4, []).append((correct_flag, acc_w))
                monitors.setdefault(key, []).append((correct_flag, acc_w))

                if pos_sample==1:
                    monitors.setdefault(new_key1+'/pos', []).append((correct_flag, acc_w))
                    monitors.setdefault(new_key2+'/pos', []).append((correct_flag, acc_w))
                else:
                    monitors.setdefault(new_key1+'/neg', []).append((correct_flag, acc_w))
                    monitors.setdefault(new_key2+'/neg', []).append((correct_flag, acc_w))


            if self.training and self.add_supervision:
                if isinstance(gt, list):
                    for idx in range(len(gt)):
                        l = loss(a[idx], gt[idx])
                        monitors.setdefault('loss/qa/' + query_type, []).append((l, loss_w))
                        monitors.setdefault('loss/qa', []).append((l, loss_w))
                        monitors.setdefault('loss/qa/' + question_type_new, []).append((l, loss_w))
                elif gt is not None:
                    l = loss(a, gt)
                    monitors.setdefault('loss/qa/' + query_type, []).append((l, loss_w))
                    monitors.setdefault('loss/qa', []).append((l, loss_w))
                    monitors.setdefault('loss/qa/' + question_type_new, []).append((l, loss_w))
       
        if result_save_path!='':
            #pdb.set_trace()
            if not os.path.isdir(result_save_path):
                os.makedirs(result_save_path)
            full_path = os.path.join(result_save_path, 
                    str(feed_dict['meta_ann']['scene_index'])+'.pk')
            out_dict = {'answer': answers,
                    'gt': feed_dict['answer']}
            pickledump(full_path, out_dict)

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

