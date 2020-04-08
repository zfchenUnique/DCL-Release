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

logger = get_logger(__file__)

__all__ = ['ConceptQuantizationContext', 'ProgramExecutorContext', 'DifferentiableReasoning', 'set_apply_self_mask']


_apply_self_mask = {'relate': True, 'relate_ae': True}
_fixed_start_end = True
time_win = 10
_symmetric_collision_flag=True
EPS = 1e-10

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

def Guaussin_smooth(x):
    # Create gaussian kernels
    kernel = torch.tensor((0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006), dtype=x.dtype, device=x.device)
    # Apply smoothing
    x_smooth = F.conv1d(x.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=3)
    #pdb.set_trace()
    return x_smooth.squeeze()

def fuse_box_ftr(box_ftr):
    obj_num, ftr_dim = box_ftr.shape
    rel_ftr_box = torch.zeros(obj_num, obj_num, ftr_dim*4, \
            dtype=box_ftr.dtype, device=box_ftr.device)
    for obj_id1 in range(obj_num):
        for obj_id2 in range(obj_num):
            tmp_ftr_minus = box_ftr[obj_id1] - box_ftr[obj_id2]
            tmp_ftr_mul = box_ftr[obj_id1] * box_ftr[obj_id2]
            tmp_ftr = torch.cat([tmp_ftr_minus, tmp_ftr_mul, box_ftr[obj_id1] , box_ftr[obj_id2]], dim=0)
            rel_ftr_box[obj_id1, obj_id2] = tmp_ftr 
    return rel_ftr_box 

def set_apply_self_mask(key, value):
    logger.warning('Set {}.apply_self_mask[{}] to {}.'.format(set_apply_self_mask.__module__, key, value))
    assert key in _apply_self_mask, key
    _apply_self_mask[key] = value


def do_apply_self_mask(m):
    self_mask = torch.eye(m.size(-1), dtype=m.dtype, device=m.device)
    return m * (1 - self_mask) + (-10) * self_mask


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
    def __init__(self, attribute_taxnomy, relation_taxnomy, temporal_taxnomy, time_taxnomy, features, parameter_resolution, training=True, args=None):
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

    def exist(self, selected):
        if isinstance(selected, tuple):
            selected = selected[0]
        if len(selected.shape)==1:
            return selected.max(dim=-1)[0]
        elif len(selected.shape)==2:
            return 0.5*(selected+selected.transpose(1, 0)).max()

    def belong_to(self, selected1, selected2):
        if isinstance(selected, tuple):
            selected = selected[0]
        return (selected1 * selected2).sum(dim=-1)

    def count(self, selected):
        #pdb.set_trace()
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
                return torch.sigmoid(count_conf).sum()
            else:
                if _test_quantize.value >= InferenceQuantizationMethod.STANDARD.value:
                    return (count_conf > 0).float().sum()
                #print('Debuging!')
                return torch.sigmoid(count_conf).sum().round()

    _count_margin = 0.25
    _count_tau = 0.25

    def relate(self, selected, group, concept_groups):
        if isinstance(selected, tuple):
            selected = selected[0]
        #pdb.set_trace()
        mask = self._get_concept_groups_masks(concept_groups, 2)
        mask = (mask * selected.unsqueeze(-1).unsqueeze(0)).sum(dim=-2)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0)
        return mask[group]

    def filter_collision(self, selected, group, concept_groups):
        if isinstance(selected, tuple):
            time_weight = selected[1].squeeze()
            selected = selected[0]
        else:
            time_weight = None
        mask = self._get_collision_groups_masks(concept_groups, 2, time_weight)
        #mask = (mask * selected.unsqueeze(-1).unsqueeze(0)).sum(dim=-2)
        #if torch.is_tensor(group):
        #    return (mask * group.unsqueeze(1)).sum(dim=0)
        return mask[group]

    def get_col_partner(self, selected, mask):
        mask = (mask * selected.unsqueeze(-1)).sum(dim=-2)
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
        #pdb.set_trace()
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
        if isinstance(selected, tuple):
            selected = selected[0]
        return -1*selected 

    def filter_temporal(self, selected, group, concept_groups):
        if group is None:
            return selected
        if isinstance(selected, list) and len(selected)==2:
            if isinstance(selected[1], tuple):
                time_mask = selected[1][1]
            else:
                time_mask = selected[1]
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
        #pdb.set_trace()
        mask = self._get_concept_groups_masks(concept_groups, 3)
        mask = torch.min(selected.unsqueeze(0), mask)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0)
        return mask[group]

    def filter_order(self, selected, group, concept_groups):
        if group is None:
            return selected
        if isinstance(selected, tuple):
            #pdb.set_trace()
            selected = selected[0]
        if len(selected.shape)==1:
            k = 3
            #pdb.set_trace()
            mask = self._get_concept_groups_masks(concept_groups, k)
        elif len(selected.shape)==2:
            k = 2
            mask = self._get_order_groups_masks(concept_groups, k)
        mask = torch.min(selected.unsqueeze(0), mask)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0)
        return mask[group]

    def filter_before_after(self, time_weight, group, concept_groups):
        if isinstance(time_weight, tuple):
            selected = time_weight[0]
            time_weight = time_weight[1]
            time_weight = time_weight.squeeze()
        else:
            selected = None
        k = 4
        #pdb.set_trace()
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
        mask = self._get_time_concept_groups_masks(concept_groups, 3, None)
        mask = torch.min(selected.unsqueeze(0), mask)

        # find the in/out time for the target object
        k = 4
        obj_num, ftr_dim = self.features[3].shape
        box_dim = 4
        time_step = int(ftr_dim/box_dim) 
        box_thre = 0.0001
        min_frm = 5

        masks = []
            #time_mask = None
        for c in concept_groups[group]:
            tar_obj_id = torch.argmax(mask[group])
            tar_ftr = self.features[3][tar_obj_id].view(time_step, box_dim)
            #ftr_diff = torch.zeros(time_step, box_dim, dtype=tar_ftr.dtype, \
            #device=tar_ftr.device)
            time_weight =  torch.zeros(time_step, dtype=tar_ftr.dtype, device=tar_ftr.device)
            tar_area = tar_ftr[:, 2] * tar_ftr[:, 3]
            if c=='in':
                for t_id in range(time_step):
                    end_id = min(t_id + min_frm, time_step-1)
                    if torch.sum(tar_area[t_id:end_id]>box_thre)>=(end_id-t_id):
                        time_weight[t_id:end_id] = 1
                        break 
                    if t_id== time_step - 1:
                        time_weight[0] = 1
            elif c=='out':
                for t_id in range(time_step, -1, -1):
                    st_id = max(t_id - min_frm, 0)
                    if torch.sum(tar_area[st_id:t_id]>box_thre)>=(t_id-st_id):
                        time_weight[st_id:t_id] = 1
                        break
                    if t_id == 0:
                        time_weight[time_step-1] = 1
            masks.append(time_weight)
        #pdb.set_trace()
        self._time_buffer_masks = torch.stack(masks, dim=0)[0]
        return mask[group], self._time_buffer_masks


    def filter_in_out(self, selected, group, concept_groups):
        if isinstance(selected, tuple):
            selected = selected[0]
        k = 4
        obj_num, ftr_dim = self.features[3].shape
        box_dim = 4
        time_step = int(ftr_dim/box_dim) 
        ftr = self.features[3].view(obj_num, time_step, box_dim)
        pdb.set_trace()
        # time_step* box_dim
        tar_ftr = (selected.view(obj_num, 1, 1) * ftr).sum(dim=0) 
        ftr_diff = torch.zeros(time_step, box_dim, dtype=ftr.dtype, \
                device=ftr.device)
        ftr_diff[:time_step-1] = (tar_ftr[0:time_step-1] - tar_ftr[1:time_step])
        ftr_diff = ftr_diff.view(1, ftr_dim)
        masks = []
        for cg in concept_groups:
            if isinstance(cg, six.string_types):
                cg = [cg]
            for c in cg:
                if c=='in':
                    time_weight = self.taxnomy[k].filter_in(ftr_diff)
                elif c=='out':
                    time_weight = self.taxnomy[k].filter_out(ftr_diff)
                time_weight = F.softmax(time_weight, dim=-1)
                masks.append(time_mask)
        self._time_buffer_masks = torch.stack(masks, dim=0)
        # update obejct state
        mask = self._get_time_concept_groups_masks(concept_groups, 3, self._concept_groups_masks[k][group])
        mask = torch.min(selected.unsqueeze(0), mask)
        
        return mask[group], self._time_buffer_masks[group]

    def filter_start_end(self, group, concept_groups):
        #pdb.set_trace()
        k = 4
        if self._concept_groups_masks[k] is None:
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
                        #pdb.set_trace() 
                    masks.append(mask)
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
            #pdb.set_trace()
            if k == 2 and _apply_self_mask['relate']:
                mask = do_apply_self_mask(mask)
            masks.append(mask)
        self._concept_groups_masks[k] = torch.stack(masks, dim=0)
        return self._concept_groups_masks[k]

    def _get_collision_groups_masks(self, concept_groups, k, time_mask):
        assert k==2
        if self._concept_groups_masks[k] is None:
            obj_num, ftr_dim = self.features[3].shape
            box_dim = 4
            time_step = int(ftr_dim/box_dim) 
            if time_mask is not None:
                ftr = self.features[3].view(obj_num, time_step, box_dim) * time_mask.view(1, time_step, 1)
            else:
                ftr = self.features[3]
            ftr = ftr.view(obj_num, -1)

            rel_box_ftr = fuse_box_ftr(ftr)
            # concatentate
            if not self.args.box_only_for_collision_flag:
                rel_ftr_norm = torch.cat([self.features[k], rel_box_ftr], dim=-1)
            else:
                rel_ftr_norm =  rel_box_ftr 
            if self.args.box_iou_for_collision_flag:
                #pdb.set_trace()
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
                        #pdb.set_trace()
                if k == 2 and _apply_self_mask['relate']:
                    mask = do_apply_self_mask(mask)
                masks.append(mask)
            self._concept_groups_masks[k] = torch.stack(masks, dim=0)
            self.features[2] = rel_ftr_norm 
        return self._concept_groups_masks[k]

    def _get_time_concept_groups_masks(self, concept_groups, k, time_mask):
        obj_num, ftr_dim = self.features[3].shape
        box_dim = 4
        time_step = int(ftr_dim/box_dim)
        if time_mask is not None:
            ftr = self.features[3].view(obj_num, time_step, box_dim) * time_mask.view(1, time_step, 1)
            ftr = ftr.view(obj_num, -1)
        else:
            ftr = self.features[3]
        #if self._concept_groups_masks[k] is None:
        masks = list()
        for cg in concept_groups:
            if isinstance(cg, six.string_types):
                cg = [cg]
            mask = None
            for c in cg:
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
    def __init__(self, used_concepts, input_dims, hidden_dims, parameter_resolution='deterministic', vse_attribute_agnostic=False, args=None):
        super().__init__()
        self.used_concepts = used_concepts
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.parameter_resolution = parameter_resolution
        self.args= args 
        #pdb.set_trace()

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

    def forward(self, batch_features, progs_list, fd=None):
        assert len(progs_list) == len(batch_features)
        programs_list = []
        buffers_list = []
        result_list = []
        batch_size = len(batch_features)
        #pdb.set_trace()
        for vid_id, vid_ftr in enumerate(batch_features):
            features = batch_features[vid_id]
            progs = progs_list[vid_id] 
            feed_dict = fd[vid_id]
            programs = []
            buffers = []
            result = []
            obj_num = len(feed_dict['tube_info']) - 2
          

            for i,  prog in enumerate(progs):
                buffer = []

                buffers.append(buffer)
                programs.append(prog)
                #pdb.set_trace()
                
                ctx_features = [None]
                for f_id in range(1, 4): 
                    ctx_features.append(features[f_id].clone())

                ctx = ProgramExecutorContext(self.embedding_attribute, self.embedding_relation, \
                        self.embedding_temporal, self.embedding_time, ctx_features,\
                        parameter_resolution=self.parameter_resolution, training=self.training, args=self.args)

                for block_id, block in enumerate(prog):
                    op = block['op']

                    if op == 'scene' or op =='objects':
                        buffer.append(10 + torch.zeros(obj_num, dtype=torch.float, device=features[1].device))
                        continue

                    inputs = []
                    for inp, inp_type in zip(block['inputs'], gdef.operation_signatures_dict[op][1]):
                        inp = buffer[inp]
                        if inp_type == 'object':
                            inp = ctx.unique(inp)
                        #if inp is None:
                        #    pdb.set_trace()
                        inputs.append(inp)

                    # TODO(Jiayuan Mao @ 10/06): add support of soft concept attention.
                    #pdb.set_trace()
                    if op == 'filter':
                        buffer.append(ctx.filter(*inputs, block['concept_idx'], block['concept_values']))
                    elif op == 'filter_order':
                        #pdb.set_trace()
                        buffer.append(ctx.filter_order(*inputs, block['temporal_concept_idx'], block['temporal_concept_values']))
                    elif op == 'end' or op == 'start':
                        #pdb.set_trace()
                        # TODO filter on the temporal features
                        buffer.append(ctx.filter_start_end(*inputs, block['time_concept_idx'], block['time_concept_values']))
                    elif op =='get_frame':
                        buffer.append(ctx.filter_time_object(*inputs))
                    elif op == 'filter_in' or op == 'filter_out':
                        #print(feed_dict['meta_ann']['questions'][i]['question'])
                        #pdb.set_trace()
                        buffer.append(ctx.filter_in_out_rule(*inputs, block['time_concept_idx'],\
                                block['time_concept_values']))
                    elif op == 'filter_before' or op == 'filter_after':
                        buffer.append(ctx.filter_before_after(*inputs, block['time_concept_idx'], block['time_concept_values']))
                    elif op == 'filter_temporal':
                        #pdb.set_trace()
                        buffer.append(ctx.filter_temporal(inputs, block['temporal_concept_idx'], block['temporal_concept_values']))
                    elif op == 'filter_collision':
                        #pdb.set_trace()
                        buffer.append(ctx.filter_collision(*inputs, block['relational_concept_idx'], block['relational_concept_values']))
                    elif op == 'get_col_partner':
                        buffer.append(ctx.get_col_partner(*inputs))
                    elif op == 'belong_to':
                        buffer.append(ctx.belong_to(*inputs))
                    elif op == 'exist':
                        buffer.append(ctx.exist(*inputs))
                    else:
                        assert block_id == len(prog) - 1, 'Unexpected query operation: {}. Are you using the CLEVR-convension?'.format(op)
                        if op == 'query':
                            buffer.append(ctx.query(*inputs, block['attribute_idx'], block['attribute_values']))
                        elif op == 'count':
                            buffer.append(ctx.count(*inputs))
                        elif op == 'negate':
                            pdb.set_trace()
                            buffer.append(ctx.negate(*inputs))
                        else:
                            continue
                            pdb.set_trace()
                            #raise NotImplementedError('Unsupported operation: {}.'.format(op))

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

                #pdb.set_trace()
                quasi_symbolic_debug.embed(self, i, buffer, result, feed_dict)
            
            programs_list.append(programs)
            buffers_list.append(buffers)
            result_list.append(result)
        return programs_list, buffers_list, result_list 
