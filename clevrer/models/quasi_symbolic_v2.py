#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : quasi_symbolic.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/02/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.


"""
Quasi-Symbolic Reasoning.
"""

import six

import torch
import torch.nn as nn

import jactorch.nn.functional as jacf

from jacinle.logging import get_logger
from jacinle.utils.enum import JacEnum
from nscl.datasets.common.program_executor import ParameterResolutionMode
from nscl.datasets.definition import gdef
from . import concept_embedding, concept_embedding_ls
from . import quasi_symbolic_debug
import pdb
import jactorch.nn as jacnn
import torch.nn.functional as F
import copy
from scipy import signal 
import numpy as np

logger = get_logger(__file__)

__all__ = ['ConceptQuantizationContext', 'ProgramExecutorContext', 'DifferentiableReasoning', 'set_apply_self_mask']


_apply_self_mask = {'relate': True, 'relate_ae': True}
_fixed_start_end = True
time_win = 10
_symmetric_collision_flag=True
EPS = 1e-10
_collision_thre = 0.0

def compute_IoU(bbox1_xyhw, bbox2_xyhw):
    bbox1_area = bbox1_xyhw[:, 2] * bbox1_xyhw[:, 3]
    bbox2_area = bbox2_xyhw[:, 2] * bbox2_xyhw[:, 3]
    
    bbox1_x1 = bbox1_xyhw[:,0] - bbox1_xyhw[:, 2]*0.5 
    bbox1_x2 = bbox1_xyhw[:,0] + bbox1_xyhw[:, 2]*0.5 
    bbox1_y1 = bbox1_xyhw[:,1] - bbox1_xyhw[:, 3]*0.5 
    bbox1_y2 = bbox1_xyhw[:,1] + bbox1_xyhw[:, 3]*0.5 

    bbox2_x1 = bbox2_xyhw[:,0] - bbox2_xyhw[:, 2]*0.5 
    bbox2_x2 = bbox2_xyhw[:,0] + bbox2_xyhw[:, 2]*0.5 
    bbox2_y1 = bbox2_xyhw[:,1] - bbox2_xyhw[:, 3]*0.5 
    bbox2_y2 = bbox2_xyhw[:,1] + bbox2_xyhw[:, 3]*0.5 

    w = torch.clamp(torch.min(bbox1_x2, bbox2_x2) - torch.max(bbox1_x1, bbox2_x1), min=0)
    h = torch.clamp(torch.min(bbox1_y2, bbox2_y2) - torch.max(bbox1_y1, bbox2_y1), min=0)
    
    inter = w * h
    ovr = inter / (bbox1_area + bbox2_area - inter+EPS)
    return ovr

def fuse_box_overlap(box_ftr):
    obj_num, ftr_dim = box_ftr.shape
    box_dim = 4
    time_step = int(ftr_dim / 4)
    rel_ftr_box = torch.zeros(obj_num, obj_num, time_step, \
            dtype=box_ftr.dtype, device=box_ftr.device)
    for obj_id1 in range(obj_num):
        for obj_id2 in range(obj_id1+1, obj_num):
            rel_ftr_box[obj_id1, obj_id2] = compute_IoU(box_ftr[obj_id1].view(time_step, box_dim),\
                    box_ftr[obj_id2].view(time_step, box_dim))
            rel_ftr_box[obj_id2, obj_id1] = rel_ftr_box[obj_id1, obj_id2]
        #pdb.set_trace()
    return rel_ftr_box 

def Gaussin_smooth(x):
    '''
    x: N * timestep * 4
    '''
    # Create gaussian kernels
    win_size = 5
    std = 1
    box_dim = 4

    x_mask = x>0
    x_mask_neg = 1 - x_mask.float() 

    x = x*x_mask.float()

    obj_num, ftr_dim = x.shape
    time_step = int(ftr_dim / box_dim)
    x_trans = x.view(obj_num, time_step, box_dim).permute(0, 2, 1)

    pad_size = int((win_size-1)/2) 
    filter_param = signal.gaussian(win_size, std)
    filter_param = filter_param/np.sum(filter_param)
    kernel = torch.tensor(filter_param, dtype=x.dtype, device=x.device)
    
    pad_fun = nn.ReplicationPad1d(pad_size)
    x_trans_pad = pad_fun(x_trans) 

    # Apply smoothing
    x_smooth_trans = F.conv1d(x_trans_pad.contiguous().view(-1, 1, time_step+pad_size*2), kernel.unsqueeze(0).unsqueeze(0), padding=0)
    x_smooth_trans = x_smooth_trans.view(obj_num, box_dim, time_step) 
    x_smooth = x_smooth_trans.permute(0, 2, 1)
    x_smooth = x_smooth.contiguous().view(obj_num, ftr_dim)
    # remask 
    x_smooth  = x_smooth * x_mask.float()
    x_smooth += x_mask_neg.float()*(-1)
    #pdb.set_trace()
    return x_smooth.squeeze()

def fuse_box_ftr(box_ftr):
    """
    input: N*seg*ftr
    output: N*N*seg*ftr*4
    """
    obj_num, seg_num, ftr_dim = box_ftr.shape
    rel_ftr_box = torch.zeros(obj_num, obj_num, seg_num, ftr_dim*4, \
            dtype=box_ftr.dtype, device=box_ftr.device)
    for obj_id1 in range(obj_num):
        for obj_id2 in range(obj_num):
            # seg_num * ftr_dim
            tmp_ftr_minus = box_ftr[obj_id1] + box_ftr[obj_id2]
            tmp_ftr_mul = box_ftr[obj_id1] * box_ftr[obj_id2]
            tmp_ftr = torch.cat([tmp_ftr_minus, tmp_ftr_mul, box_ftr[obj_id1] , box_ftr[obj_id2]], dim=1)
            rel_ftr_box[obj_id1, obj_id2] = tmp_ftr 
    return rel_ftr_box 

def set_apply_self_mask(key, value):
    logger.warning('Set {}.apply_self_mask[{}] to {}.'.format(set_apply_self_mask.__module__, key, value))
    assert key in _apply_self_mask, key
    _apply_self_mask[key] = value


def do_apply_self_mask(m):
    self_mask = torch.eye(m.size(-1), dtype=m.dtype, device=m.device)
    return m * (1 - self_mask) + (-10) * self_mask

def do_apply_self_mask_3d(m):
    obj_num = m.size(0)
    self_mask = torch.eye(obj_num, dtype=m.dtype, device=m.device)
    frm_num = m.shape[2]
    self_mask_exp = self_mask.unsqueeze(-1).expand(obj_num, obj_num, frm_num)
    return m * (1 - self_mask_exp) + (-10) * self_mask_exp

class InferenceQuantizationMethod(JacEnum):
    NONE = 0
    STANDARD = 1
    EVERYTHING = 2
    


_test_quantize = InferenceQuantizationMethod.STANDARD
#_test_quantize = InferenceQuantizationMethod.NONE
#_test_quantize = InferenceQuantizationMethod.EVERYTHING 


def set_test_quantize(mode):
    global _test_quantize
    _test_quantize = InferenceQuantizationMethod.from_string(mode)



class ConceptQuantizationContext(nn.Module):
    def __init__(self, attribute_taxnomy, relation_taxnomy, training=False, quasi=False):
        """
        Args:
            attribute_taxnomy: attribute-level concept embeddings.
            relation_taxnomy: relation-level concept embeddings.
            training (bool): training mode or not.
            quasi(bool): if False, quantize the results as 0/1.

        """

        super().__init__()

        self.attribute_taxnomy = attribute_taxnomy
        self.relation_taxnomy = relation_taxnomy
        self.quasi = quasi

        super().train(training)

    def forward(self, f_sng):
        batch_size = len(f_sng)
        output_list = [dict() for i in range(batch_size)]

        for i in range(batch_size):
            f = f_sng[i][1]
            nr_objects = f.size(0)

            output_list[i]['filter'] = dict()
            for concept in self.attribute_taxnomy.all_concepts:
                scores = self.attribute_taxnomy.similarity(f, concept)
                if self.quasi:
                    output_list[i]['filter'][concept] = scores.detach().cpu().numpy()
                else:
                    output_list[i]['filter'][concept] = (scores > 0).nonzero().squeeze(-1).cpu().tolist()

            output_list[i]['relate_ae'] = dict()
            for attr in self.attribute_taxnomy.all_attributes:
                cross_scores = self.attribute_taxnomy.cross_similarity(f, attr)
                if _apply_self_mask['relate_ae']:
                    cross_scores = do_apply_self_mask(cross_scores)
                if self.quasi:
                    output_list[i]['relate_ae'][attr] = cross_scores.detach().cpu().numpy()
                else:
                    cross_scores = cross_scores > 0
                    output_list[i]['relate_ae'][attr] = cross_scores.nonzero().cpu().tolist()

            output_list[i]['query'] = dict()
            for attr in self.attribute_taxnomy.all_attributes:
                scores, word2idx = self.attribute_taxnomy.query_attribute(f, attr)
                idx2word = {v: k for k, v in word2idx.items()}
                if self.quasi:
                    output_list[i]['query'][attr] = scores.detach().cpu().numpy(), idx2word
                else:
                    argmax = scores.argmax(-1)
                    output_list[i]['query'][attr] = [idx2word[v] for v in argmax.cpu().tolist()]

            f = f_sng[i][2]

            output_list[i]['relate'] = dict()
            for concept in self.relation_taxnomy.all_concepts:
                scores = self.relation_taxnomy.similarity(f, concept)
                if self.quasi:
                    output_list[i]['relate'][concept] = scores.detach().cpu().numpy()
                else:
                    output_list[i]['relate'][concept] = (scores > 0).nonzero().cpu().tolist()

            output_list[i]['nr_objects'] = nr_objects

        return output_list


class ProgramExecutorContext(nn.Module):
    def __init__(self, attribute_taxnomy, relation_taxnomy, temporal_taxnomy, time_taxnomy, features, parameter_resolution, training=True, args=None, future_features=None, seg_frm_num=None):
        super().__init__()
        self.args = args 
        self.features = features
        self.parameter_resolution = ParameterResolutionMode.from_string(parameter_resolution)

        # None, attributes, relations
        self.taxnomy = [None, attribute_taxnomy, relation_taxnomy, temporal_taxnomy, time_taxnomy]
        self._concept_groups_masks = [None, None, None, None, None]
        self._time_buffer_masks = None

        self._attribute_groups_masks = None
        self._attribute_query_masks = None
        self._attribute_query_ls_masks = None
        self._attribute_query_ls_mc_masks = None
        self.train(training)
        self._events_buffer = [None, None, None] # collision, in and out 
        self.time_step = int(self.features[3].shape[1]/4)
        self.valid_seq_mask  = None
        self._unseen_event_buffer = None # for collision in the future
        self._future_features = future_features
        self._seg_frm_num = seg_frm_num 

    def filter_ancestor(self, event_list):
        obj_id_list =[]
        obj_weight_list =[]
        target_frm_id = None
        objset_weight = None
        if len(event_list)==4:
            obj1_idx = torch.argmax(event_list[0])
            obj2_idx = torch.argmax(event_list[2])
            coll_idx = event_list[1][1][obj1_idx, obj2_idx]
            target_frm_id = self._events_buffer[0][1][coll_idx] 
            objset_weight = torch.max(event_list[0], event_list[2])
            obj_id_list = [obj1_idx, obj2_idx]           
        elif len(event_list)==2: 
            obj1_idx = torch.argmax(event_list[1][0])
            objset_weight = event_list[1][0]
            target_frm_id = event_list[1][1][obj1_idx]
            obj_id_list = [obj1_idx]           
        else:
            raise NotImplementedError('Unsupported input of length: {}.'.format(len(event_list)))
        all_causes = []
        self._search_causes(objset_weight, target_frm_id, all_causes, obj_id_list)
        # merge confidence
        obj_num = len(objset_weight)
        colli_mask = torch.zeros(obj_num, obj_num, device=objset_weight.device)-10
        in_mask = torch.zeros(objset_weight.shape, device=objset_weight.device)-10
        out_mask = torch.zeros(objset_weight.shape, device=objset_weight.device)-10
                
        for tmp_cause in all_causes:
            colli_mask = torch.max(colli_mask, tmp_cause[0]) 
            in_mask = torch.max(in_mask, tmp_cause[1]) 
            out_mask = torch.max(out_mask, tmp_cause[2]) 

        return colli_mask, in_mask, out_mask 
    
    def _search_causes(self, objset_weight, target_frm_id, all_causes, explored_list):

        if target_frm_id>self._events_buffer[0][1][0]:

            frm_mask_list = [] 
            # filtering causal collisions
            for smp_id, frm_id in enumerate(self._events_buffer[0][1]): 
                if frm_id<target_frm_id:
                    frm_mask_list.append(1)
                else:
                    frm_mask_list.append(0)
            frm_weight = torch.tensor(frm_mask_list, dtype= objset_weight.dtype, device = objset_weight.device)
            frm_weight_2 = 10 * (1 - frm_weight)
            #colli_3d_mask = self._events_buffer[0][0]*frm_weight.unsqueeze(0).unsqueeze(0)
            colli_3d_mask = self._events_buffer[0][0] - frm_weight_2.unsqueeze(0).unsqueeze(0)
            colli_mask, colli_t_idx = torch.max(colli_3d_mask, dim=2)
            obj_weight_mask = torch.max(objset_weight.unsqueeze(-1), objset_weight.unsqueeze(-2))
            colli_mask3 = torch.min(colli_mask, obj_weight_mask)
            # filtering in/out collisions
            in_mask = torch.min(self._events_buffer[1][0], objset_weight)
            out_mask = torch.min(self._events_buffer[2][0], objset_weight)
            # masking out events after time
            obj_num = len(in_mask)
            for obj_id in range(obj_num):
                in_frm = self._events_buffer[1][1][obj_id]
                out_frm = self._events_buffer[2][1][obj_id]
                if in_frm>target_frm_id:
                    in_mask[obj_id] = -10
                if out_frm>target_frm_id:
                    out_mask[obj_id] = -10

            all_causes.append([colli_mask3, in_mask, out_mask])
            # filter other objects in the graphs 
            obj_idx_mat = (colli_mask3>_collision_thre).nonzero()
            event_len = obj_idx_mat.shape[0] 

            for idx in range(event_len):
                #pdb.set_trace() 
                obj_id1 = obj_idx_mat[idx, 0]
                obj_id2 = obj_idx_mat[idx, 1]
                target_frm_id = colli_t_idx[obj_id1, obj_id2]
                if obj_id1 not in explored_list:
                    new_obj_weight =  torch.zeros(objset_weight.shape, device=objset_weight.device)-10
                    new_obj_weight[obj_id1] = 10
                    explored_list.append(obj_id1)
                    self._search_causes(new_obj_weight, target_frm_id, all_causes, explored_list)
                if obj_id2 not in explored_list:
                    new_obj_weight =  torch.zeros(objset_weight.shape, device=objset_weight.device)-10
                    new_obj_weight[obj_id2] = 10
                    explored_list.append(obj_id2)
                    self._search_causes(new_obj_weight, target_frm_id, all_causes, explored_list)

    def init_unseen_events(self):
        if self._unseen_event_buffer is None:
            obj_num, obj_num2, pred_frm_num, ftr_dim = self._future_features[2].shape
            box_dim = self._future_features[3].shape[1]//pred_frm_num
            if self.args.colli_ftr_type ==1:
                # B*B*T*D
                coll_ftr = self._future_features[2]
                # bilinear sampling for target box feature
                # B*T*d1
                box_ftr = self._future_features[3].clone().view(obj_num, 1, pred_frm_num, box_dim)
                # B*B*(T*sample_frames)*d1
                # TODO: making it constant with the seen video
                box_ftr_exp = F.interpolate(box_ftr, size=[pred_frm_num*self._seg_frm_num, box_dim], mode='bilinear') 
                ftr = box_ftr_exp.view(obj_num, pred_frm_num, -1)
                rel_box_ftr = fuse_box_ftr(ftr)
                rel_ftr_norm = torch.cat([coll_ftr, rel_box_ftr], dim=-1)
                if self.args.box_iou_for_collision_flag:
                    # N*N*(T*sample_frames)
                    box_iou_ftr  = fuse_box_overlap(ftr.view(obj_num, -1))
                    box_iou_ftr_view = box_iou_ftr.view(obj_num, obj_num, pred_frm_num, self._seg_frm_num)
                    rel_ftr_norm = torch.cat([rel_ftr_norm, box_iou_ftr_view], dim=-1)
            else:
                raise NotImplementedError 

            k = 2
            masks = list()
            for cg in ['collision']:
                if isinstance(cg, six.string_types):
                    cg = [cg]
                mask = None
                for c in cg:
                    new_mask = self.taxnomy[2].similarity_collision(rel_ftr_norm, c)
                    mask = torch.min(mask, new_mask) if mask is not None else new_mask
                    if _symmetric_collision_flag:
                        mask = 0.5*(mask + mask.transpose(1, 0))
                mask = do_apply_self_mask_3d(mask)
                masks.append(mask)
            event_colli_set = torch.stack(masks, dim=0)
            event_colli_score, frm_idx = event_colli_set[0].max(dim=2)
            self._unseen_event_buffer = [event_colli_score , frm_idx]
            return event_colli_score 
        else:
            return self._unseen_event_buffer[0] 

    def init_events(self):
        if self._events_buffer[0] is None:
            obj_num = self.features[1].shape[0]
            input_objset = 10 + torch.zeros(obj_num, dtype=torch.float, device=self.features[1].device)
            event_in_objset_pro, frm_list_in = self.init_in_out_rule(input_objset, 'in') 
            event_out_objset_pro, frm_list_out = self.init_in_out_rule(input_objset, 'out') 
            event_collision_prp, frm_list_colli = self.init_collision(self.args.smp_coll_frm_num) 
            self._events_buffer[0] = [event_collision_prp, frm_list_colli]
            self._events_buffer[1] = [event_in_objset_pro, frm_list_in]
            self._events_buffer[2] = [event_out_objset_pro, frm_list_out]
        return self._events_buffer 

    def init_collision(self, smp_coll_frm_num):
        obj_num, ftr_dim = self.features[3].shape
        box_dim = 4
        time_step = int(ftr_dim/box_dim) 
        offset = time_step%smp_coll_frm_num 
        seg_frm_num = int((time_step-offset)/smp_coll_frm_num) 
        half_seg_frm_num = int(seg_frm_num/2)
        frm_list = []
        ftr = self.features[3].view(obj_num, time_step, box_dim)[:, :time_step-offset, :box_dim]
        ftr = ftr.view(obj_num, smp_coll_frm_num, seg_frm_num*box_dim)
        # N*N*smp_coll_frm_num*(seg_frm_num*box_dim*4)
        rel_box_ftr = fuse_box_ftr(ftr)
        # concatentate
        if self.args.colli_ftr_type ==1:
            try:
                vis_ftr_num = self.features[2].shape[2]
                col_ftr_dim = self.features[2].shape[3]
                off_set = smp_coll_frm_num % vis_ftr_num 
                exp_dim = int(smp_coll_frm_num / vis_ftr_num )
                exp_dim = max(1, exp_dim)
                coll_ftr = torch.zeros(obj_num, obj_num, smp_coll_frm_num, col_ftr_dim, \
                        dtype=rel_box_ftr.dtype, device=rel_box_ftr.device)
                coll_ftr_exp = self.features[2].unsqueeze(3).expand(obj_num, obj_num, vis_ftr_num, exp_dim, col_ftr_dim).contiguous()
                coll_ftr_exp_view = coll_ftr_exp.view(obj_num, obj_num, vis_ftr_num*exp_dim, col_ftr_dim)
                min_frm_num = min(vis_ftr_num*exp_dim, smp_coll_frm_num)
                coll_ftr[:, :, :min_frm_num] = coll_ftr_exp_view[:,:, :min_frm_num] 
                if vis_ftr_num*exp_dim<smp_coll_frm_num:
                    #pass
                    coll_ftr[:, :, -1*off_set:] = coll_ftr_exp_view[:,:, -1, :].unsqueeze(2) 
                    #coll_ftr[:, :, min_frm_num:] = self.features[2][:, :, -1].unsqueeze(2)
                rel_ftr_norm = torch.cat([coll_ftr, rel_box_ftr], dim=-1)
            except:
                pdb.set_trace()

        elif not self.args.box_only_for_collision_flag:
            col_ftr_dim = self.features[2].shape[2]
            coll_ftr_exp = self.features[2].unsqueeze(2).expand(obj_num, obj_num, smp_coll_frm_num, col_ftr_dim)
            rel_ftr_norm = torch.cat([coll_ftr_exp, rel_box_ftr], dim=-1)
        else:
            rel_ftr_norm =  rel_box_ftr 
        if self.args.box_iou_for_collision_flag:
            # N*N*time_step 
            box_iou_ftr  = fuse_box_overlap(ftr.view(obj_num, -1))
            box_iou_ftr_view = box_iou_ftr.view(obj_num, obj_num, smp_coll_frm_num, seg_frm_num)
            rel_ftr_norm = torch.cat([rel_ftr_norm, box_iou_ftr_view], dim=-1)

        k = 2
        masks = list()
        for cg in ['collision']:
            if isinstance(cg, six.string_types):
                cg = [cg]
            mask = None
            for c in cg:
                new_mask = self.taxnomy[2].similarity_collision(rel_ftr_norm, c)
                mask = torch.min(mask, new_mask) if mask is not None else new_mask
                if _symmetric_collision_flag:
                    mask = 0.5*(mask + mask.transpose(1, 0))
            mask = do_apply_self_mask_3d(mask)
            masks.append(mask)
        event_colli_set = torch.stack(masks, dim=0)

        for frm_id in range(smp_coll_frm_num):
            centre_frm = frm_id * seg_frm_num + half_seg_frm_num  
            frm_list.append(centre_frm)

        return event_colli_set[0], frm_list 

    def init_in_out_rule(self, selected, concept):
        
        # update obejct state
        mask = self._get_time_concept_groups_masks([concept], 3, None)
        mask = torch.min(selected.unsqueeze(0), mask)
        # find the in/out time for the target object
        k = 4
        obj_num, ftr_dim = self.features[3].shape
        box_dim = 4
        time_step = int(ftr_dim/box_dim) 
        box_thre = 0.0001
        min_frm = 5

        event_frm = []
        for tar_obj_id in range(obj_num):
            c = concept 
            tar_ftr = self.features[3][tar_obj_id].view(time_step, box_dim)
            time_weight =  torch.zeros(time_step, dtype=tar_ftr.dtype, device=tar_ftr.device)
            tar_area = tar_ftr[:, 2] * tar_ftr[:, 3]
            if c=='in':
                for t_id in range(time_step):
                    end_id = min(t_id + min_frm, time_step-1)
                    if torch.sum(tar_area[t_id:end_id]>box_thre)>=(end_id-t_id) and torch.sum(tar_ftr[t_id:end_id,2])>0:
                        if self.args.diff_for_moving_stationary_flag:
                            event_frm.append(t_id)
                        break 
                    if t_id== time_step - 1:
                        if self.args.diff_for_moving_stationary_flag:
                            event_frm.append(0)

            elif c=='out':
                for t_id in range(time_step-1, -1, -1):
                    st_id = max(t_id - min_frm, 0)
                    if torch.sum(tar_area[st_id:t_id]>box_thre)>=(t_id-st_id) and torch.sum(tar_ftr[st_id:t_id])>0:
                        if self.args.diff_for_moving_stationary_flag:
                            event_frm.append(t_id)
                        break
                    if t_id == 0:
                        if self.args.diff_for_moving_stationary_flag:
                            event_frm.append(time_step - 1)
        return mask[0], event_frm 

    def exist(self, selected):
        if isinstance(selected, tuple):
            selected = selected[0]
        if len(selected.shape)==1:
            return selected.max(dim=-1)[0]
        elif len(selected.shape)==2:
            return 0.5*(selected+selected.transpose(1, 0)).max()

    def belong_to(self, choice_output_list, cause_event_list):
        choice_result_list = []
        for choice_output in choice_output_list:
            choice_type = choice_output[0]
            choice_mask = choice_output[1]
            if choice_type == 'collision':
                choice_result = torch.min(choice_mask, cause_event_list[0]).max()  
                choice_result_list.append(choice_result)
            elif choice_type == 'in':
                choice_result = torch.min(choice_mask, cause_event_list[1]).max()  
                choice_result_list.append(choice_result)
            elif choice_type == 'out':
                choice_result = torch.min(choice_mask, cause_event_list[2]).max()  
                choice_result_list.append(choice_result)
            elif choice_type == 'object':
                choice_result1 = torch.min(choice_mask, cause_event_list[1]).max()  
                choice_result2 = torch.min(choice_mask, cause_event_list[2]).max()  
                choice_result3 = torch.min(choice_mask.unsqueeze(-1), cause_event_list[0]).max() 
                choice_result = torch.max(torch.stack([choice_result1, choice_result2, choice_result3]))
                choice_result_list.append(choice_result)
            else:
                raise NotImplementedError 
        return choice_result_list

    def count(self, selected):
        if isinstance(selected, tuple):
            selected = selected[0]
        if len(selected.shape)==1: # for objects
            if self.training:
                return torch.sigmoid(selected).sum(dim=-1)
            else:
                if _test_quantize.value >= InferenceQuantizationMethod.STANDARD.value:
                    return (selected > 0).float().sum()
                #print('Debuging!')
                return torch.sigmoid(selected).sum(dim=-1).round()
        elif len(selected.shape)==2:  # for collision
            # mask out the diag elelments for collisions
            obj_num = selected.shape[0]
            self_mask = 1- torch.eye(obj_num, dtype=selected.dtype, device=selected.device)
            count_conf = self_mask * (selected+selected.transpose(1, 0))*0.5
            if self.training:
                return torch.sigmoid(count_conf).sum()/2
            else:
                if _test_quantize.value >= InferenceQuantizationMethod.STANDARD.value:
                    return (count_conf > 0).float().sum()/2
                #print('Debuging!')
                return (torch.sigmoid(count_conf).sum()/2).round()

    _count_margin = 0.25
    _count_tau = 0.25

    def relate(self, selected, group, concept_groups):
        if isinstance(selected, tuple):
            selected = selected[0]
        mask = self._get_concept_groups_masks(concept_groups, 2)
        mask = (mask * selected.unsqueeze(-1).unsqueeze(0)).sum(dim=-2)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0)
        return mask[group]

    def filter_collision(self, selected, group, concept_groups, ques_type='descriptive'):
        if isinstance(selected, tuple):
            time_weight = selected[1].squeeze()
            selected = selected[0]
        else:
            time_weight = None

        if ques_type=='descriptive' or ques_type=='explanatory':
            colli_frm_list = self._events_buffer[0][1]
            if time_weight is not None:
                frm_mask_list = [] 
                for smp_id, frm_id in enumerate(colli_frm_list): 
                    if time_weight[frm_id]>0:
                        frm_mask_list.append(1)
                    else:
                        frm_mask_list.append(0)
                frm_weight = torch.tensor(frm_mask_list, dtype= time_weight.dtype, device = time_weight.device)
                frm_weight_2 = -10 * (1 - frm_weight)
                #colli_3d_mask = self._events_buffer[0][0]*frm_weight.unsqueeze(0).unsqueeze(0)
                colli_3d_mask = self._events_buffer[0][0] - frm_weight_2.unsqueeze(0).unsqueeze(0)
            else:
                colli_3d_mask = self._events_buffer[0][0]
            colli_mask, colli_t_idx = torch.max(colli_3d_mask, dim=2)
        elif ques_type == 'predictive':
            colli_mask, colli_t_idx = self._unseen_event_buffer 
        elif ques_type == 'counterfactual':
            pdb.set_trace()
            raise NotImplementedError
        else:
            raise NotImplementedError
        obj_set_weight = None
        if selected is not None and (not isinstance(selected, (tuple, list))):
            selected_mask  = torch.max(selected.unsqueeze(-1), selected.unsqueeze(-2))
            colli_mask2 = torch.min(colli_mask, selected_mask)
        else:
            colli_mask2 = colli_mask 
        return colli_mask2, colli_t_idx 

    def get_col_partner(self, selected, mask):
        if isinstance(mask, tuple) :
            mask_idx = mask[1]
            mask = mask[0]
        mask = (mask * selected.unsqueeze(-1)).sum(dim=-2)
        #selected_quan = jacf.general_softmax(selected, impl='gumbel_hard', training=False)
        #mask_idx = (mask * selected_quan.unsqueeze(-1)).sum(dim=-2)
        return mask

    def query(self, selected, group, attribute_groups):
        if isinstance(selected, tuple):
            selected = selected[0]
        mask, word2idx = self._get_attribute_query_masks(attribute_groups)
        mask = (mask * selected.unsqueeze(-1).unsqueeze(0)).sum(dim=-2)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0), word2idx
        return mask[group], word2idx
        
    def unique(self, selected):
        if isinstance(selected, tuple):
            selected = selected[0]
        if self.training or _test_quantize.value < InferenceQuantizationMethod.STANDARD.value:
            return jacf.general_softmax(selected, impl='standard', training=self.training)
        # trigger the greedy_max
        return jacf.general_softmax(selected, impl='gumbel_hard', training=self.training)

    def filter(self, selected, group, concept_groups):
        if isinstance(selected, tuple):
            selected = selected[0]
        if group is None:
            return selected
        mask = self._get_concept_groups_masks(concept_groups, 1)
        mask = torch.min(selected.unsqueeze(0), mask)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0)
        return mask[group]

    def negate(self, selected):
        if isinstance(selected, list):
            new_selected = []
            for idx in range(len(selected)):
                new_selected.append(-1*selected[idx])
        else:
            new_selected = -1*selected 
        return new_selected 

    def filter_temporal(self, selected, group, concept_groups):
        if group is None:
            return selected
        if isinstance(selected, list) and len(selected)==2:
            if isinstance(selected[1], tuple):
                time_mask = selected[1][1]
            else:
                time_mask = selected[1]
            if isinstance(selected[0], list):
                selected = selected[0][0]
            else:
                selected = selected[0]
        elif isinstance(selected, list) and len(selected)==1:
            selected = selected[0]
            time_mask = None
        else:
            time_mask = None
        mask = self._get_time_concept_groups_masks(concept_groups, 3, time_mask)
        mask = torch.min(selected.unsqueeze(0), mask)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0)
        return mask[group]

    def filter_temporal_bp(self, selected, group, concept_groups):
        if group is None:
            return selected
        if isinstance(selected, tuple):
            selected = selected[0]
        mask = self._get_concept_groups_masks(concept_groups, 3)
        mask = torch.min(selected.unsqueeze(0), mask)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0)
        return mask[group]

    def filter_order(self, selected, group, concept_groups):
        if group is None:
            return selected
        if isinstance(selected, tuple) and len(selected)==3:
            event_frm = selected[2]
            selected_idx = selected[1]
            selected = selected[0]
        elif isinstance(selected, tuple) and len(selected)==2:
            selected_idx = selected[1]
            selected = selected[0]
        
        if len(selected.shape)==1:
            k = 3
            event_frm_sort, sorted_idx = torch.sort(event_frm, dim=-1)
            masks = list()
            obj_num = len(selected)
            for cg in concept_groups:
                if isinstance(cg, six.string_types):
                    cg = [cg]
                mask = None
                for c in cg:
                    if c=='first':
                        for idx in range(obj_num):
                            new_mask_idx = sorted_idx[idx]
                            if event_frm_sort[idx]!=0:
                                break 

                    elif c=='second':
                        in_idx = 0
                        for idx in range(obj_num):
                            new_mask_idx = sorted_idx[idx]
                            if event_frm_sort[idx]!=0:
                                in_idx +=1
                            if in_idx==2:
                                break 
                    
                    elif c=='last':
                        for idx in range(obj_num-1, -1, -1):
                            new_mask_idx = sorted_idx[idx]
                            if event_frm_sort[idx]!= self.time_step-1:
                                break 
                    new_mask = -10 + torch.zeros(selected.shape, device=selected.device)
                    new_mask[new_mask_idx] = 10
                    mask = torch.min(mask, new_mask) if mask is not None else new_mask
                masks.append(mask)
            masks = torch.stack(masks, dim=0)

        elif len(selected.shape)==2:
            _, sorted_idx = torch.sort(selected_idx, dim=-1)

            masks = list()
            for cg in concept_groups:
                if isinstance(cg, six.string_types):
                    cg = [cg]
                mask = None
                for c in cg:
                    if c=='first':
                        new_mask_idx = sorted_idx[:, 0]
                    elif c=='second':
                        new_mask_idx = sorted_idx[:, 1]
                    elif c=='last':
                        new_mask_idx = sorted_idx[:, -1]
                    new_mask = -10 + torch.zeros(selected.shape, device=selected.device)
                    for obj_id, rank_idx in enumerate(new_mask_idx): 
                        new_mask[obj_id, rank_idx] = 10 
                    mask = torch.min(mask, new_mask) if mask is not None else new_mask
                if _apply_self_mask['relate']:
                    mask = do_apply_self_mask(mask)
                masks.append(mask)
            masks = torch.stack(masks, dim=0)

        mask = torch.min(selected.unsqueeze(0), masks)
        return mask[group]

    def filter_before_after(self, time_weight, group, concept_groups):
        if isinstance(time_weight, tuple):
            selected = time_weight[0]
            time_weight = time_weight[1]
            time_weight = time_weight.squeeze()
        else:
            selected = None
        k = 4
        naive_weight = True
        if naive_weight:
            max_weight = torch.argmax(time_weight)
            time_step = len(time_weight)
            time_mask = torch.zeros([time_step], device = time_weight.device)
            assert len(concept_groups[group])==1
            if concept_groups[group]==['before']:
                time_mask[:max_weight] = 1.0
            elif concept_groups[group] == ['after']:
                time_mask[max_weight:] = 1.0
        else:
            time_step = len(time_weight)
            time_weight = Guaussin_smooth(time_weight)
            max_weight = torch.max(time_weight)
            norm_time_weight = (time_weight/max_weight)**100
            after_weight = torch.cumsum(norm_time_weight, dim=-1)
            after_weight = after_weight/torch.max(after_weight)
            assert len(concept_groups[group])==1
            if concept_groups[group]==['before']:
                time_mask = 1 - after_weight 
            elif concept_groups[group] == ['after']:
                time_mask = after_weight 
        # update obejct state
        mask = self._get_time_concept_groups_masks(concept_groups, 3, time_mask)
        if selected is not None:
            mask = torch.min(selected.unsqueeze(0), mask)
        # update features
        #box_dim = int(self.features[3].shape[1]/time_step)
        #time_mask_exp = time_mask.unsqueeze(1).expand(time_step, box_dim).contiguous().view(1, time_step*box_dim)
        #print('Bug!!!')
        #self.features[3] = self.features[3] * time_mask_exp 
        self._time_buffer_masks = time_mask 
        return mask[group], time_mask 


    def filter_in_out_rule(self, selected, group, concept_groups):
        if isinstance(selected, tuple):
            selected = selected[0]
        # update obejct state
        assert len(concept_groups[group])==1 
        c = concept_groups[group][0]
        if c=='in':
            c_id = 1
        elif c=='out':
            c_id = 2
        mask = torch.min(selected, self._events_buffer[c_id][0])
        max_obj_id = torch.argmax(mask)
        frm_id = self._events_buffer[c_id][1][max_obj_id] 
        time_weight =  torch.zeros(self.time_step, dtype=mask.dtype, device=mask.device)
        time_weight[frm_id] = 1
        self._time_buffer_masks = time_weight
        event_index = self._events_buffer[c_id][1]
        event_frm = torch.tensor(event_index, dtype= selected.dtype, device = selected.device)
        return mask, self._time_buffer_masks, event_frm 

    def filter_start_end(self, group, concept_groups):
        k = 4
        #if self._concept_groups_masks[k] is None:
        masks = list()
        for cg in concept_groups:
            if isinstance(cg, six.string_types):
                cg = [cg]
            mask = None
            for c in cg:
                concept = self.taxnomy[k].get_concept(c)
                new_mask = concept.softmax_normalized_embedding 
                mask = torch.min(mask, new_mask) if mask is not None else new_mask
                if _fixed_start_end:
                    mask = torch.zeros(mask.shape, dtype=mask.dtype, device=mask.device)
                    if c == 'start':
                        mask[:,:time_win] = 1
                    elif c == 'end':
                        mask[:,-time_win:] = 1
                masks.append(mask)
        self._time_buffer_masks = mask 
        self._concept_groups_masks[k] = torch.stack(masks, dim=0)
        return self._concept_groups_masks[k][group]

    def filter_time_object(self, selected, time_weight):
        obj_num = self.features[3].shape[0]
        time_step = len(time_weight.squeeze())
        ftr = self.features[3].view(obj_num, time_step, 4) * time_weight.view(1, time_step, 1)
        ftr = ftr.view(obj_num, -1)
        # enlarging the scores for object filtering
        obj_weight = torch.tanh(self.taxnomy[4].exist_object(ftr))*5
        mask = torch.min(selected, obj_weight.squeeze())
        return mask

    def _get_concept_groups_masks_bp(self, concept_groups, k):
        #if self._concept_groups_masks[k] is None:
        masks = list()
        for cg in concept_groups:
            if isinstance(cg, six.string_types):
                cg = [cg]
            mask = None
            for c in cg:
                new_mask = self.taxnomy[k].similarity(self.features[k], c)
                mask = torch.min(mask, new_mask) if mask is not None else new_mask
            if k == 2 and _apply_self_mask['relate']:
                mask = do_apply_self_mask(mask)
            masks.append(mask)
        self._concept_groups_masks[k] = torch.stack(masks, dim=0)
        return self._concept_groups_masks[k]

    def _get_concept_groups_masks(self, concept_groups, k):
        if self._concept_groups_masks[k] is None:
            masks = list()
            for cg in concept_groups:
                if isinstance(cg, six.string_types):
                    cg = [cg]
                mask = None
                for c in cg:
                    new_mask = self.taxnomy[k].similarity(self.features[k], c)
                    mask = torch.min(mask, new_mask) if mask is not None else new_mask
                if k == 2 and _apply_self_mask['relate']:
                    mask = do_apply_self_mask(mask)
                masks.append(mask)
            self._concept_groups_masks[k] = torch.stack(masks, dim=0)
        return self._concept_groups_masks[k]

    def _get_order_groups_masks(self, concept_groups, k):
        masks = list()
        for cg in concept_groups:
            if isinstance(cg, six.string_types):
                cg = [cg]
            mask = None
            for c in cg:
                new_mask = self.taxnomy[k].similarity(self.features[k], c)
                mask = torch.min(mask, new_mask) if mask is not None else new_mask
            if k == 2 and _apply_self_mask['relate']:
                mask = do_apply_self_mask(mask)
            masks.append(mask)
        self._concept_groups_masks[k] = torch.stack(masks, dim=0)
        return self._concept_groups_masks[k]

    def _get_collision_groups_masks(self, concept_groups, k, time_mask):
        assert k==2
        #if self._concept_groups_masks[k] is None:
        obj_num, ftr_dim = self.features[3].shape
        box_dim = 4
        time_step = int(ftr_dim/box_dim) 
        if time_mask is not None:
            ftr = self.features[3].view(obj_num, time_step, box_dim) * time_mask.view(1, time_step, 1)
        else:
            ftr = self.features[3].clone()
            if self._time_buffer_masks is not None:
                pdb.set_trace()
        ftr = ftr.view(obj_num, -1)

        rel_box_ftr = fuse_box_ftr(ftr)
        # concatentate
        if not self.args.box_only_for_collision_flag:
            rel_ftr_norm = torch.cat([self.features[k], rel_box_ftr], dim=-1)
        else:
            rel_ftr_norm =  rel_box_ftr 
        if self.args.box_iou_for_collision_flag:
            box_iou_ftr  = fuse_box_overlap(ftr)
            rel_ftr_norm = torch.cat([rel_ftr_norm, box_iou_ftr], dim=-1)

        masks = list()
        for cg in concept_groups:
            if isinstance(cg, six.string_types):
                cg = [cg]
            mask = None
            for c in cg:
                new_mask = self.taxnomy[k].similarity_collision(rel_ftr_norm, c)
                mask = torch.min(mask, new_mask) if mask is not None else new_mask
                if _symmetric_collision_flag:
                    mask = 0.5*(mask + mask.transpose(1, 0))
            if k == 2 and _apply_self_mask['relate']:
                mask = do_apply_self_mask(mask)
            masks.append(mask)
        self._concept_groups_masks[k] = torch.stack(masks, dim=0)
        #self.features[2] = rel_ftr_norm 
        return self._concept_groups_masks[k]

    def further_prepare_for_moving_stationary(self, ftr_ori, time_mask, concept):
        obj_num, ftr_dim = ftr_ori.shape 
        box_dim = 4
        time_step = int(ftr_dim/box_dim)
        if time_mask is None and (self._time_buffer_masks is not None):
            time_mask = self._time_buffer_masks 
        elif time_mask is not None and time_mask.sum()<=1:
            max_idx = torch.argmax(time_mask)
            st_idx = max(int(max_idx-time_win*0.5), 0)
            ed_idx = min(int(max_idx+time_win*0.5), time_step-1)
            time_mask[st_idx:ed_idx] = 1
        #assert time_mask is not None
        if time_mask is not None:
            ftr_mask = ftr_ori.view(obj_num, time_step, box_dim) * time_mask.view(1, time_step, 1)
        else:
            ftr_mask = ftr_ori.view(obj_num, time_step, box_dim)
        ftr_diff = torch.zeros(obj_num, time_step, box_dim, dtype=ftr_ori.dtype, \
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

    def _get_time_concept_groups_masks(self, concept_groups, k, time_mask):
        obj_num, ftr_dim = self.features[3].shape
        box_dim = 4
        time_step = int(ftr_dim/box_dim)
        #if self._concept_groups_masks[k] is None:
        if time_mask is not None and time_mask.sum()>1:
            ftr = self.features[3].view(obj_num, time_step, box_dim) * time_mask.view(1, time_step, 1)
            ftr = ftr.view(obj_num, -1)
        elif time_mask is not None:
            ftr = self.features[3].clone()
        else:
            if self._time_buffer_masks is None or self._time_buffer_masks.sum()<=1:
                ftr = self.features[3].clone()
                time_mask = self._time_buffer_masks 
            else:
                ftr = self.features[3].view(obj_num, time_step, box_dim) * self._time_buffer_masks.view(1, time_step, 1)
                ftr = ftr.view(obj_num, -1)
                time_mask = self._time_buffer_masks.squeeze() 
        masks = list()
        for cg in concept_groups:
            if isinstance(cg, six.string_types):
                cg = [cg]
            mask = None
            for c in cg:
                if (c == 'moving' or c == 'stationary') and self.args.diff_for_moving_stationary_flag:
                    ftr = self.further_prepare_for_moving_stationary(self.features[3], time_mask, c)
                if self.valid_seq_mask is not None:
                    ftr = ftr.view(obj_num, time_step, box_dim) * self.valid_seq_mask - (1-self.valid_seq_mask)
                    ftr = ftr.view(obj_num, -1)
                new_mask = self.taxnomy[k].similarity(ftr, c)
                mask = torch.min(mask, new_mask) if mask is not None else new_mask
            masks.append(mask)
        self._concept_groups_masks[k] = torch.stack(masks, dim=0)
        return self._concept_groups_masks[k]

    def _get_attribute_groups_masks(self, attribute_groups):
        if self._attribute_groups_masks is None:
            masks = list()
            for attribute in attribute_groups:
                mask = self.taxnomy[1].cross_similarity(self.features[1], attribute)
                if _apply_self_mask['relate_ae']:
                    mask = do_apply_self_mask(mask)
                masks.append(mask)
            self._attribute_groups_masks = torch.stack(masks, dim=0)
        return self._attribute_groups_masks

    def _get_attribute_query_masks(self, attribute_groups):
        if self._attribute_query_masks is None:
            masks, word2idx = list(), None
            for attribute in attribute_groups:
                mask, this_word2idx = self.taxnomy[1].query_attribute(self.features[1], attribute)
                masks.append(mask)
                # sanity check.
                if word2idx is not None:
                    for k in word2idx:
                        assert word2idx[k] == this_word2idx[k]
                word2idx = this_word2idx

            self._attribute_query_masks = torch.stack(masks, dim=0), word2idx
        return self._attribute_query_masks

    def _get_attribute_query_ls_masks(self, attribute_groups):
        if self._attribute_query_ls_masks is None:
            masks, word2idx = list(), None
            for attribute in attribute_groups:
                mask, this_word2idx = self.taxnomy[1].query_attribute(self.features[1], attribute)
                masks.append(mask)
                word2idx = this_word2idx

            self._attribute_query_ls_masks = torch.stack(masks, dim=0), word2idx
        return self._attribute_query_ls_masks

    def _get_attribute_query_ls_mc_masks(self, attribute_groups, concepts):
        if self._attribute_query_ls_mc_masks is None:
            masks, word2idx = list(), None
            for attribute in attribute_groups:
                mask, this_word2idx = self.taxnomy[1].query_attribute_mc(self.features[1], attribute, concepts)
                masks.append(mask)
                word2idx = this_word2idx

            self._attribute_query_ls_mc_masks = torch.stack(masks, dim=0), word2idx
        return self._attribute_query_ls_mc_masks


class DifferentiableReasoning(nn.Module):
    def __init__(self, used_concepts, input_dims, hidden_dims, parameter_resolution='deterministic', vse_attribute_agnostic=False, args=None, seg_frm_num=-1):
        super().__init__()
        self.used_concepts = used_concepts
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.parameter_resolution = parameter_resolution
        self.args= args 
        self._seg_frm_num = seg_frm_num 

        for i, nr_vars in enumerate(['attribute', 'relation', 'temporal', 'time']):
            if nr_vars not in self.used_concepts:
                continue
            setattr(self, 'embedding_' + nr_vars, concept_embedding.ConceptEmbedding(vse_attribute_agnostic))
            tax = getattr(self, 'embedding_' + nr_vars)
            rec = self.used_concepts[nr_vars]

            for a in rec['attributes']:
                tax.init_attribute(a, self.input_dims[1 + i], self.hidden_dims[1 + i])
            for (v, b) in rec['concepts']:
                tax.init_concept(v, self.hidden_dims[1 + i], known_belong=b)
            if nr_vars=='time':
                tax.exist_object = jacnn.LinearLayer(self.input_dims[1+i], 1, activation=None)
                # TODO more complicated filter_in and out function
                tax.filter_in = jacnn.LinearLayer(self.input_dims[1+i], 128, activation=None)
                tax.filter_out = jacnn.LinearLayer(self.input_dims[1+i], 128, activation=None)

    def forward(self, batch_features, progs_list, fd=None, future_features_list=None):
        # To do divide programs into oe set and mc set
        # run program seperately 
        programs_list_oe, buffers_list_oe, result_list_oe = self.forward_oe(batch_features, progs_list, fd)
        #programs_list_mc, buffers_list_mc, result_list_mc = self.forward_mc(batch_features, progs_list, fd, future_features_list)
        programs_list_mc, buffers_list_mc, result_list_mc = self.forward_mc_dynamic(batch_features, progs_list, fd, future_features_list)
        programs_list=[]; buffers_list= []; result_list = []
        for vid in range(len(fd)):
            programs_list.append(programs_list_oe[vid] + programs_list_mc[vid])
            buffers_list.append(buffers_list_oe[vid] + buffers_list_mc[vid]) 
            result_list.append(result_list_oe[vid] + result_list_mc[vid]) 
        return programs_list, buffers_list, result_list 

    def forward_oe(self, batch_features, progs_list, fd=None):
        assert len(progs_list) == len(batch_features)
        programs_list = []
        buffers_list = []
        result_list = []
        batch_size = len(batch_features)
        
        for vid_id, vid_ftr in enumerate(batch_features):
            features = batch_features[vid_id]
            progs = progs_list[vid_id] 
            feed_dict = fd[vid_id]
            programs = []
            buffers = []
            result = []
            obj_num = len(feed_dict['tube_info']) - 2
            
            #if feed_dict['meta_ann']['scene_index']==7398:
            #    pdb.set_trace()

            ctx_features = [None]
            for f_id in range(1, 4): 
                ctx_features.append(features[f_id].clone())

            if self.args.apply_gaussian_smooth_flag:
                ctx_features[3] = Gaussin_smooth(ctx_features[3])

            ctx = ProgramExecutorContext(self.embedding_attribute, self.embedding_relation, \
                    self.embedding_temporal, self.embedding_time, ctx_features,\
                    parameter_resolution=self.parameter_resolution, training=self.training, args=self.args)

            if 'valid_seq_mask' in feed_dict.keys():
                ctx.valid_seq_mask = torch.zeros(obj_num, 128, 1).to(features[3].device)
                valid_len = feed_dict['valid_seq_mask'].shape[1]
                ctx.valid_seq_mask[:, :valid_len, 0] = torch.from_numpy(feed_dict['valid_seq_mask']).float()

            for i,  prog in enumerate(progs):

                if feed_dict['meta_ann']['questions'][i]['question_type']!='descriptive':
                    continue 
            
                ctx._concept_groups_masks = [None, None, None, None, None]
                ctx._time_buffer_masks = None
                ctx._attribute_groups_masks = None
                ctx._attribute_query_masks = None

                buffer = []

                buffers.append(buffer)
                programs.append(prog)
                

                for block_id, block in enumerate(prog):
                    op = block['op']

                    if op == 'scene' or op =='objects':
                        buffer.append(10 + torch.zeros(obj_num, dtype=torch.float, device=features[1].device))
                        continue
                    elif op == 'events':
                        buffer.append(ctx.init_events())
                        continue

                    inputs = []
                    for inp, inp_type in zip(block['inputs'], gdef.operation_signatures_dict[op][1]):
                        inp = buffer[inp]
                        if inp_type == 'object':
                            inp = ctx.unique(inp)
                        inputs.append(inp)

                    if op == 'filter':
                        buffer.append(ctx.filter(*inputs, block['concept_idx'], block['concept_values']))
                    elif op == 'filter_order':
                        buffer.append(ctx.filter_order(*inputs, block['temporal_concept_idx'], block['temporal_concept_values']))
                    elif op == 'end' or op == 'start':
                        buffer.append(ctx.filter_start_end(*inputs, block['time_concept_idx'], block['time_concept_values']))
                    elif op =='get_frame':
                        buffer.append(ctx.filter_time_object(*inputs))
                    elif op == 'filter_in' or op == 'filter_out':
                        buffer.append(ctx.filter_in_out_rule(*inputs, block['time_concept_idx'],\
                                block['time_concept_values']))
                    elif op == 'filter_before' or op == 'filter_after':
                        #print(feed_dict['meta_ann']['questions'][i]['question'])
                        buffer.append(ctx.filter_before_after(*inputs, block['time_concept_idx'], block['time_concept_values']))
                    elif op == 'filter_temporal':
                        buffer.append(ctx.filter_temporal(inputs, block['temporal_concept_idx'], block['temporal_concept_values']))
                    elif op == 'filter_collision':
                        buffer.append(ctx.filter_collision(*inputs, block['relational_concept_idx'], block['relational_concept_values']))
                    elif op == 'get_col_partner':
                        buffer.append(ctx.get_col_partner(*inputs))
                    elif op == 'exist':
                        buffer.append(ctx.exist(*inputs))
                    else:
                        assert block_id == len(prog) - 1, 'Unexpected query operation: {}. Are you using the CLEVR-convension?'.format(op)
                        if op == 'query':
                            buffer.append(ctx.query(*inputs, block['attribute_idx'], block['attribute_values']))
                        elif op == 'count':
                            buffer.append(ctx.count(*inputs))
                        elif op == 'negate':
                            buffer.append(ctx.negate(*inputs))
                        else:
                            raise NotImplementedError('Unsupported operation: {}.'.format(op))

                    if not self.training and _test_quantize.value > InferenceQuantizationMethod.STANDARD.value:
                        if block_id != len(prog) - 1:
                            if not isinstance(buffer[-1], tuple):
                                buffer[-1] = -10 + 20 * (buffer[-1] > 0).float()
                            else:
                                buffer[-1] = list(buffer[-1])
                                for out_id, out_value in enumerate(buffer[-1]):
                                    buffer[-1][out_id] = -10 + 20 * (buffer[-1][out_id] > 0).float()
                                buffer[-1] = tuple(buffer[-1])

                result.append((op, buffer[-1]))

                quasi_symbolic_debug.embed(self, i, buffer, result, feed_dict)
            
            programs_list.append(programs)
            buffers_list.append(buffers)
            result_list.append(result)
        return programs_list, buffers_list, result_list

    def forward_mc(self, batch_features, progs_list, fd=None, future_feature_list=None):
        assert len(progs_list) == len(batch_features)
        programs_list = []
        buffers_list = []
        result_list = []
        batch_size = len(batch_features)
        for vid_id, vid_ftr in enumerate(batch_features):
            features = batch_features[vid_id]
            progs = progs_list[vid_id] 
            feed_dict = fd[vid_id]
            future_features = future_feature_list[vid_id]
            programs = []
            buffers = []
            result = []
            obj_num = len(feed_dict['tube_info']) - 2


            ctx_features = [None]
            for f_id in range(1, 4): 
                ctx_features.append(features[f_id].clone())

            if self.args.apply_gaussian_smooth_flag:
                ctx_features[3] = Gaussin_smooth(ctx_features[3])

            ctx = ProgramExecutorContext(self.embedding_attribute, self.embedding_relation, \
                    self.embedding_temporal, self.embedding_time, ctx_features,\
                    parameter_resolution=self.parameter_resolution, \
                    training=self.training, args=self.args, future_features=future_features,\
                    seg_frm_num = self._seg_frm_num)
            
            if 'valid_seq_mask' in feed_dict.keys():
                ctx.valid_seq_mask = torch.zeros(obj_num, 128, 1).to(features[3].device)
                valid_len = feed_dict['valid_seq_mask'].shape[1]
                ctx.valid_seq_mask[:, :valid_len, 0] = torch.from_numpy(feed_dict['valid_seq_mask']).float()

            valid_num = 0
            for i,  prog in enumerate(progs):
                if feed_dict['meta_ann']['questions'][i]['question_type']!='explanatory' and \
                    feed_dict['meta_ann']['questions'][i]['question_type']!='predictive':
                    continue 

                buffer = []

                buffers.append(buffer)
                programs.append(prog)
                
                choice_output  = []
                choice_buffer_list = []
                for c_id, tmp_choice in enumerate(feed_dict['meta_ann']['questions'][i]['choices']):
                    choice_prog = tmp_choice['program_cl']
                    ctx._concept_groups_masks = [None, None, None, None, None]
                    ctx._time_buffer_masks = None
                    ctx._attribute_groups_masks = None
                    ctx._attribute_query_masks = None
                    #print(tmp_choice['program_cl'])
                    #print(tmp_choice['choice'])
                    tmp_event_buffer = []
                    choice_buffer = []
                    choice_type = None
                    for block_id, block in enumerate(choice_prog):
                        op = block['op']
                        
                        if op == 'scene' or op =='objects':
                            choice_buffer.append(10 + torch.zeros(obj_num, dtype=torch.float, device=features[1].device))
                            continue
                        elif op == 'events':
                            choice_buffer.append(ctx.init_events())
                            continue
                        elif op == 'unseen_events':
                            choice_buffer.append(ctx.init_unseen_events())
                            continue

                        inputs = []
                        for inp, inp_type in zip(block['inputs'], gdef.operation_signatures_dict[op][1]):
                            inp = choice_buffer[inp]
                            if inp_type == 'object':
                                inp = ctx.unique(inp)
                            inputs.append(inp)

                        if op == 'filter':
                            choice_buffer.append(ctx.filter(*inputs, block['concept_idx'], block['concept_values']))
                        elif op == 'filter_order':
                            choice_buffer.append(ctx.filter_order(*inputs, block['temporal_concept_idx'], block['temporal_concept_values']))
                        elif op == 'end' or op == 'start':
                            choice_buffer.append(ctx.filter_start_end(*inputs, block['time_concept_idx'], block['time_concept_values']))
                        elif op =='get_frame':
                            choice_buffer.append(ctx.filter_time_object(*inputs))
                        elif op == 'filter_in' or op == 'filter_out':
                            choice_buffer.append(ctx.filter_in_out_rule(*inputs, block['time_concept_idx'],\
                                    block['time_concept_values']))
                            tmp_event_buffer.append(choice_buffer[block_id][0])
                            choice_type = block['time_concept_values'][block['time_concept_idx']][0]
                        elif op == 'filter_before' or op == 'filter_after':
                            choice_buffer.append(ctx.filter_before_after(*inputs, block['time_concept_idx'], block['time_concept_values']))
                        elif op == 'filter_temporal':
                            choice_buffer.append(ctx.filter_temporal(inputs, block['temporal_concept_idx'], block['temporal_concept_values']))
                        elif op == 'filter_collision':
                            choice_buffer.append(ctx.filter_collision(*inputs, block['relational_concept_idx'], block['relational_concept_values']))
                            choice_type = block['relational_concept_values'][block['relational_concept_idx']][0]
                            tmp_event_buffer.append(choice_buffer[block_id][0])
                        elif op == 'get_col_partner':
                            choice_buffer.append(ctx.get_col_partner(*inputs))
                        elif op == 'exist':
                            choice_buffer.append(ctx.exist(*inputs))
                        else:
                            raise NotImplementedError 
                    event_buffer = None
                    if len(tmp_event_buffer) == 0:
                        event_buffer = choice_buffer[-1]
                        choice_type = 'object'
                    else:
                        for tmp_mask in tmp_event_buffer:
                            event_buffer = torch.min(event_buffer, tmp_mask) if event_buffer is not None else tmp_mask
                    choice_output.append([choice_type, event_buffer]) 
                    choice_buffer_list.append(choice_buffer)

                ctx._concept_groups_masks = [None, None, None, None, None]
                ctx._time_buffer_masks = None
                ctx._attribute_groups_masks = None
                ctx._attribute_query_masks = None
                for block_id, block in enumerate(prog):
                    op = block['op']

                    if op == 'scene' or op =='objects':
                        buffer.append(10 + torch.zeros(obj_num, dtype=torch.float, device=features[1].device))
                        continue
                    elif op == 'events':
                        buffer.append(ctx.init_events())
                        continue
                    elif op == 'unseen_events':
                        buffer.append(ctx.init_unseen_events())
                        continue

                    inputs = []
                    for inp, inp_type in zip(block['inputs'], gdef.operation_signatures_dict[op][1]):
                        inp = buffer[inp]
                        if inp_type == 'object':
                            inp = ctx.unique(inp)
                        inputs.append(inp)

                    if op == 'filter':
                        buffer.append(ctx.filter(*inputs, block['concept_idx'], block['concept_values']))
                    elif op == 'filter_order':
                        buffer.append(ctx.filter_order(*inputs, block['temporal_concept_idx'], block['temporal_concept_values']))
                    elif op == 'end' or op == 'start':
                        buffer.append(ctx.filter_start_end(*inputs, block['time_concept_idx'], block['time_concept_values']))
                    elif op =='get_frame':
                        buffer.append(ctx.filter_time_object(*inputs))
                    elif op == 'filter_in' or op == 'filter_out':
                        buffer.append(ctx.filter_in_out_rule(*inputs, block['time_concept_idx'],\
                                block['time_concept_values']))
                    elif op == 'filter_before' or op == 'filter_after':
                        buffer.append(ctx.filter_before_after(*inputs, block['time_concept_idx'], block['time_concept_values']))
                    elif op == 'filter_temporal':
                        buffer.append(ctx.filter_temporal(inputs, block['temporal_concept_idx'], block['temporal_concept_values']))
                    elif op == 'filter_collision':
                        buffer.append(ctx.filter_collision(*inputs, block['relational_concept_idx'], block['relational_concept_values']))
                    elif op == 'get_col_partner':
                        buffer.append(ctx.get_col_partner(*inputs))
                    elif op == 'belong_to':
                        #pdb.set_trace()
                        buffer.append(ctx.belong_to(choice_output, *inputs))
                    elif op == 'exist':
                        buffer.append(ctx.exist(*inputs))
                    elif op == 'filter_ancestor':
                        buffer.append(ctx.filter_ancestor(inputs))
                    else:
                        assert block_id == len(prog) - 1, 'Unexpected query operation: {}. Are you using the CLEVR-convension?'.format(op)
                        if op == 'query':
                            buffer.append(ctx.query(*inputs, block['attribute_idx'], block['attribute_values']))
                        elif op == 'count':
                            buffer.append(ctx.count(*inputs))
                        elif op == 'negate':
                            buffer.append(ctx.negate(*inputs))
                        else:
                            pdb.set_trace()
                            raise NotImplementedError('Unsupported operation: {}.'.format(op))

                    if not self.training and _test_quantize.value > InferenceQuantizationMethod.STANDARD.value:
                        if block_id != len(prog) - 1:
                            if not isinstance(buffer[-1], tuple):
                                buffer[-1] = -10 + 20 * (buffer[-1] > 0).float()
                            else:
                                buffer[-1] = list(buffer[-1])
                                for out_id, out_value in enumerate(buffer[-1]):
                                    buffer[-1][out_id] = -10 + 20 * (buffer[-1][out_id] > 0).float()
                                buffer[-1] = tuple(buffer[-1])

                result.append((op, buffer[-1]))
                quasi_symbolic_debug.embed(self, i, buffer, result, feed_dict, valid_num)
                valid_num +=1
            
            programs_list.append(programs)
            buffers_list.append(buffers)
            result_list.append(result)
        return programs_list, buffers_list, result_list 


    def forward_mc_dynamic(self, batch_features, progs_list, fd=None, future_feature_list=None):
        assert len(progs_list) == len(batch_features)
        programs_list = []
        buffers_list = []
        result_list = []
        batch_size = len(batch_features)
        for vid_id, vid_ftr in enumerate(batch_features):
            features = batch_features[vid_id]
            progs = progs_list[vid_id] 
            feed_dict = fd[vid_id]
            future_features = future_feature_list[vid_id]
            programs = []
            buffers = []
            result = []
            obj_num = len(feed_dict['tube_info']) - 2

            print(feed_dict['meta_ann']['scene_index'])
            print('Debug!')

            ctx_features = [None]
            for f_id in range(1, 4): 
                ctx_features.append(features[f_id].clone())

            if self.args.apply_gaussian_smooth_flag:
                ctx_features[3] = Gaussin_smooth(ctx_features[3])

            ctx = ProgramExecutorContext(self.embedding_attribute, self.embedding_relation, \
                    self.embedding_temporal, self.embedding_time, ctx_features,\
                    parameter_resolution=self.parameter_resolution, \
                    training=self.training, args=self.args, future_features=future_features,\
                    seg_frm_num = self._seg_frm_num)
            
            if 'valid_seq_mask' in feed_dict.keys():
                ctx.valid_seq_mask = torch.zeros(obj_num, 128, 1).to(features[3].device)
                valid_len = feed_dict['valid_seq_mask'].shape[1]
                ctx.valid_seq_mask[:, :valid_len, 0] = torch.from_numpy(feed_dict['valid_seq_mask']).float()

            valid_num = 0
            for i,  prog in enumerate(progs):
                ques_type = feed_dict['meta_ann']['questions'][i]['question_type']
                if feed_dict['meta_ann']['questions'][i]['question_type']!='explanatory' and \
                    feed_dict['meta_ann']['questions'][i]['question_type']!='predictive':
                    continue 

                buffer = []

                buffers.append(buffer)
                programs.append(prog)
               
                ctx._concept_groups_masks = [None, None, None, None, None]
                ctx._time_buffer_masks = None
                ctx._attribute_groups_masks = None
                ctx._attribute_query_masks = None
                belong_block_id = -1

                """
                parse the program before operator ``belong to'' 
                """
                for block_id, block in enumerate(prog):
                    op = block['op']

                    if op == 'scene' or op =='objects':
                        buffer.append(10 + torch.zeros(obj_num, dtype=torch.float, device=features[1].device))
                        continue
                    elif op == 'events':
                        buffer.append(ctx.init_events())
                        continue
                    elif op == 'unseen_events':
                        buffer.append(ctx.init_unseen_events())
                        continue

                    inputs = []
                    for inp, inp_type in zip(block['inputs'], gdef.operation_signatures_dict[op][1]):
                        inp = buffer[inp]
                        if inp_type == 'object':
                            inp = ctx.unique(inp)
                        inputs.append(inp)

                    if op == 'belong_to':
                        belong_block_id = block_id
                        break
                    elif op == 'filter':
                        buffer.append(ctx.filter(*inputs, block['concept_idx'], block['concept_values']))
                    elif op == 'filter_order':
                        buffer.append(ctx.filter_order(*inputs, block['temporal_concept_idx'], block['temporal_concept_values']))
                    elif op == 'end' or op == 'start':
                        buffer.append(ctx.filter_start_end(*inputs, block['time_concept_idx'], block['time_concept_values']))
                    elif op =='get_frame':
                        buffer.append(ctx.filter_time_object(*inputs))
                    elif op == 'filter_in' or op == 'filter_out':
                        buffer.append(ctx.filter_in_out_rule(*inputs, block['time_concept_idx'],\
                                block['time_concept_values']))
                    elif op == 'filter_before' or op == 'filter_after':
                        buffer.append(ctx.filter_before_after(*inputs, block['time_concept_idx'], block['time_concept_values']))
                    elif op == 'filter_temporal':
                        buffer.append(ctx.filter_temporal(inputs, block['temporal_concept_idx'], block['temporal_concept_values']))
                    elif op == 'filter_collision':
                        buffer.append(ctx.filter_collision(*inputs, block['relational_concept_idx'], block['relational_concept_values'], ques_type))
                    elif op == 'get_col_partner':
                        buffer.append(ctx.get_col_partner(*inputs))
                        
                        buffer.append(ctx.belong_to(choice_output, *inputs))
                    elif op == 'exist':
                        buffer.append(ctx.exist(*inputs))
                    elif op == 'filter_ancestor':
                        buffer.append(ctx.filter_ancestor(inputs))
                    else:
                        pdb.set_trace()
                        raise NotImplementedError('Unsupported operation: {}.'.format(op))

                    if not self.training and _test_quantize.value > InferenceQuantizationMethod.STANDARD.value:
                        if block_id != len(prog) - 1:
                            if not isinstance(buffer[-1], tuple):
                                buffer[-1] = -10 + 20 * (buffer[-1] > 0).float()
                            else:
                                buffer[-1] = list(buffer[-1])
                                for out_id, out_value in enumerate(buffer[-1]):
                                    buffer[-1][out_id] = -10 + 20 * (buffer[-1][out_id] > 0).float()
                                buffer[-1] = tuple(buffer[-1])

                """
                parse the choices for operator ``belong to'' 
                """

                choice_output  = []
                choice_buffer_list = []
                for c_id, tmp_choice in enumerate(feed_dict['meta_ann']['questions'][i]['choices']):
                    choice_prog = tmp_choice['program_cl']
                    ctx._concept_groups_masks = [None, None, None, None, None]
                    ctx._time_buffer_masks = None
                    ctx._attribute_groups_masks = None
                    ctx._attribute_query_masks = None
                    #print(tmp_choice['program_cl'])
                    #print(tmp_choice['choice'])
                    tmp_event_buffer = []
                    choice_buffer = []
                    choice_type = None
                    for block_id, block in enumerate(choice_prog):
                        op = block['op']
                        
                        if op == 'scene' or op =='objects':
                            choice_buffer.append(10 + torch.zeros(obj_num, dtype=torch.float, device=features[1].device))
                            continue
                        elif op == 'events':
                            choice_buffer.append(ctx.init_events())
                            continue

                        inputs = []
                        for inp, inp_type in zip(block['inputs'], gdef.operation_signatures_dict[op][1]):
                            inp = choice_buffer[inp]
                            if inp_type == 'object':
                                inp = ctx.unique(inp)
                            inputs.append(inp)

                        if op == 'filter':
                            choice_buffer.append(ctx.filter(*inputs, block['concept_idx'], block['concept_values']))
                        elif op == 'filter_order':
                            choice_buffer.append(ctx.filter_order(*inputs, block['temporal_concept_idx'], block['temporal_concept_values']))
                        elif op == 'end' or op == 'start':
                            choice_buffer.append(ctx.filter_start_end(*inputs, block['time_concept_idx'], block['time_concept_values']))
                        elif op =='get_frame':
                            choice_buffer.append(ctx.filter_time_object(*inputs))
                        elif op == 'filter_in' or op == 'filter_out':
                            choice_buffer.append(ctx.filter_in_out_rule(*inputs, block['time_concept_idx'],\
                                    block['time_concept_values']))
                            tmp_event_buffer.append(choice_buffer[block_id][0])
                            choice_type = block['time_concept_values'][block['time_concept_idx']][0]
                        elif op == 'filter_before' or op == 'filter_after':
                            choice_buffer.append(ctx.filter_before_after(*inputs, block['time_concept_idx'], block['time_concept_values']))
                        elif op == 'filter_temporal':
                            choice_buffer.append(ctx.filter_temporal(inputs, block['temporal_concept_idx'], block['temporal_concept_values']))
                        elif op == 'filter_collision':
                            choice_buffer.append(ctx.filter_collision(*inputs, block['relational_concept_idx'], block['relational_concept_values'], ques_type))
                            choice_type = block['relational_concept_values'][block['relational_concept_idx']][0]
                            tmp_event_buffer.append(choice_buffer[block_id][0])
                        elif op == 'get_col_partner':
                            choice_buffer.append(ctx.get_col_partner(*inputs))
                        elif op == 'exist':
                            choice_buffer.append(ctx.exist(*inputs))
                        else:
                            raise NotImplementedError 
                    event_buffer = None
                    if len(tmp_event_buffer) == 0:
                        event_buffer = choice_buffer[-1]
                        choice_type = 'object'
                    else:
                        for tmp_mask in tmp_event_buffer:
                            event_buffer = torch.min(event_buffer, tmp_mask) if event_buffer is not None else tmp_mask
                    choice_output.append([choice_type, event_buffer]) 
                    choice_buffer_list.append(choice_buffer)

                """
                parse the operators after ``belong to'' 
                """
                ctx._concept_groups_masks = [None, None, None, None, None]
                ctx._time_buffer_masks = None
                ctx._attribute_groups_masks = None
                ctx._attribute_query_masks = None
                for block_id, block in enumerate(prog):
                    if block_id <belong_block_id:
                        continue 
                    op = block['op']

                    inputs = []
                    for inp, inp_type in zip(block['inputs'], gdef.operation_signatures_dict[op][1]):
                        inp = buffer[inp]
                        if inp_type == 'object':
                            inp = ctx.unique(inp)
                        inputs.append(inp)

                    if op == 'belong_to':
                        #pdb.set_trace()
                        buffer.append(ctx.belong_to(choice_output, *inputs))
                    elif op == 'exist':
                        buffer.append(ctx.exist(*inputs))
                    elif op == 'filter_ancestor':
                        buffer.append(ctx.filter_ancestor(inputs))
                    else:
                        assert block_id == len(prog) - 1, 'Unexpected query operation: {}. Are you using the CLEVR-convension?'.format(op)
                        if op == 'query':
                            buffer.append(ctx.query(*inputs, block['attribute_idx'], block['attribute_values']))
                        elif op == 'count':
                            buffer.append(ctx.count(*inputs))
                        elif op == 'negate':
                            buffer.append(ctx.negate(*inputs))
                        else:
                            pdb.set_trace()
                            raise NotImplementedError('Unsupported operation: {}.'.format(op))

                    if not self.training and _test_quantize.value > InferenceQuantizationMethod.STANDARD.value:
                        if block_id != len(prog) - 1:
                            if not isinstance(buffer[-1], tuple):
                                buffer[-1] = -10 + 20 * (buffer[-1] > 0).float()
                            else:
                                buffer[-1] = list(buffer[-1])
                                for out_id, out_value in enumerate(buffer[-1]):
                                    buffer[-1][out_id] = -10 + 20 * (buffer[-1][out_id] > 0).float()
                                buffer[-1] = tuple(buffer[-1])

                result.append((op, buffer[-1]))
                quasi_symbolic_debug.embed(self, i, buffer, result, feed_dict, valid_num)
                valid_num +=1
            
            programs_list.append(programs)
            buffers_list.append(buffers)
            result_list.append(result)
        return programs_list, buffers_list, result_list 





