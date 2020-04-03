#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : scene_graph.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/19/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

"""
Scene Graph generation.
"""

import os

import torch
import torch.nn as nn
import jactorch
import jactorch.nn as jacnn

from . import functional

import pdb
import numpy as np

DEBUG = bool(int(os.getenv('DEBUG_SCENE_GRAPH', 0)))

__all__ = ['SceneGraph']


class SceneGraph(nn.Module):
    def __init__(self, feature_dim, output_dims, downsample_rate, args=None):
        super().__init__()
        self.pool_size = 7
        self.feature_dim = feature_dim
        self.output_dims = output_dims
        self.downsample_rate = downsample_rate
        self.args = args 

        if self.args.rel_box_flag:
            self.col_fuse = nn.Linear(128*4*4, output_dims[1]) 

        self.object_roi_pool = jacnn.PrRoIPool2D(self.pool_size, self.pool_size, 1.0 / downsample_rate)
        self.context_roi_pool = jacnn.PrRoIPool2D(self.pool_size, self.pool_size, 1.0 / downsample_rate)
        self.relation_roi_pool = jacnn.PrRoIPool2D(self.pool_size, self.pool_size, 1.0 / downsample_rate)

        if not DEBUG:
            self.context_feature_extract = nn.Conv2d(feature_dim, feature_dim, 1)
            self.relation_feature_extract = nn.Conv2d(feature_dim, feature_dim // 2 * 3, 1)

            self.object_feature_fuse = nn.Conv2d(feature_dim * 2, output_dims[1], 1)
            self.relation_feature_fuse = nn.Conv2d(feature_dim // 2 * 3 + output_dims[1] * 2, output_dims[2], 1)

            self.object_feature_fc = nn.Sequential(nn.ReLU(True), nn.Linear(output_dims[1] * self.pool_size ** 2, output_dims[1]))
            self.relation_feature_fc = nn.Sequential(nn.ReLU(True), nn.Linear(output_dims[2] * self.pool_size ** 2, output_dims[2]))

            self.reset_parameters()
        else:
            def gen_replicate(n):
                def rep(x):
                    return torch.cat([x for _ in range(n)], dim=1)
                return rep

            self.pool_size = 32
            self.object_roi_pool = jacnn.PrRoIPool2D(32, 32, 1.0 / downsample_rate)
            self.context_roi_pool = jacnn.PrRoIPool2D(32, 32, 1.0 / downsample_rate)
            self.relation_roi_pool = jacnn.PrRoIPool2D(32, 32, 1.0 / downsample_rate)
            self.context_feature_extract = gen_replicate(2)
            self.relation_feature_extract = gen_replicate(3)
            self.object_feature_fuse = jacnn.Identity()
            self.relation_feature_fuse = jacnn.Identity()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def merge_tube_obj_ftr(self, outputs, feed_dict):
        obj_num = len(feed_dict['tube_info']) -2
        obj_ftr = torch.zeros(obj_num, outputs[0][1].shape[1], dtype=outputs[0][1].dtype, \
                device=outputs[0][1].device)
        box_ftr = torch.zeros(obj_num, 128*4, dtype=outputs[0][1].dtype, \
                device=outputs[0][1].device)
        rel_ftr = torch.zeros(obj_num, obj_num, outputs[0][1].shape[1], \
                dtype=outputs[0][1].dtype, device=outputs[0][1].device)
        #obj_count =[0 for obj_id in range(obj_num)]
        #rel_count = np.zeros([obj_num, obj_num])
        for out_id, out_ftr in enumerate(outputs):
            tube_id_list = out_ftr[3]
            for ftr_id, tube_id in enumerate(tube_id_list):
                obj_ftr[tube_id] += out_ftr[1][ftr_id]

            for t_id1, tube_id1 in enumerate(tube_id_list):
                for t_id2, tube_id2 in enumerate(tube_id_list):
                    rel_ftr[tube_id1, tube_id2] += out_ftr[2][t_id1, t_id2]              

        vid_len = len(feed_dict['tube_info'][0])
        tube_list = []
        for obj_id in range(obj_num):
            box_seq = feed_dict['tube_info']['box_seq']['tubes'][obj_id]
            box_list = []
            for box_id, box in enumerate(box_seq):
                if isinstance(box, list):
                    if box==[0, 0, 1, 1]:
                        box =[0.0, 0, 0, 0]
                    box_tensor = torch.tensor(box, dtype=outputs[0][1].dtype, device=outputs[0][1].device)
                elif isinstance(box, torch.Tensor) or  isinstance(box, np.ndarray) :
                    box_tensor = torch.tensor(box, dtype=outputs[0][1].dtype, device=outputs[0][1].device)
                box_list.append(box_tensor)
            # 128 * 4
            box_seq_tensor = torch.stack(box_list, dim=0)
            tube_list.append(box_seq_tensor)
        tube_tensor = torch.stack(tube_list, dim=0).view(obj_num, -1)
        box_dim = min(128*4, tube_tensor.shape[1])
        box_ftr[:,:box_dim] = tube_tensor

        rel_ftr_norm = self._norm(rel_ftr)
        if self.args.rel_box_flag: 
            rel_ftr_box = torch.zeros(obj_num, obj_num, 128*4*4, \
                    dtype=outputs[0][1].dtype, device=outputs[0][1].device)
            for obj_id1 in range(obj_num):
                for obj_id2 in range(obj_num):
                    tmp_ftr_minus = box_ftr[obj_id1] - box_ftr[obj_id2]
                    tmp_ftr_mul = box_ftr[obj_id1] * box_ftr[obj_id2]
                    tmp_ftr = torch.cat([tmp_ftr_minus, tmp_ftr_mul, box_ftr[obj_id1] , box_ftr[obj_id2]], dim=0)
                    rel_ftr_box[obj_id1, obj_id2] = tmp_ftr 

            rel_ftr_box_v2 = self.col_fuse(rel_ftr_box) 
            rel_ftr_norm = torch.cat([rel_ftr_norm, rel_ftr_box_v2], dim=-1)

        return None, self._norm(obj_ftr), rel_ftr_norm, box_ftr  

    def forward(self, input, feed_dict):
        object_features = input
        context_features = self.context_feature_extract(input)
        relation_features = self.relation_feature_extract(input)

        outputs = list()

        def parse_boxes_for_frm(feed_dict, frm_idx):
            boxes_list = []
            tube_id_list = []
            frm_id = feed_dict['tube_info']['frm_list'][frm_idx]
            for tube_id, tube_info in feed_dict['tube_info'].items():
                if not isinstance(tube_id, int):
                    continue 
                assert len(tube_info['frm_name'])==len(tube_info['boxes'])
                if frm_id not in tube_info['frm_name']:
                    continue
                box_idx = tube_info['frm_name'].index(frm_id)
                box = tube_info['boxes'][box_idx]
                boxes_list.append(torch.tensor(box, device=feed_dict['img'].device))
                tube_id_list.append(tube_id)
            boxes_tensor = torch.stack(boxes_list, 0).cuda()
            return boxes_tensor, tube_id_list
 
        #pdb.set_trace()
        for i in range(input.size(0)):
            boxes, tube_id_list = parse_boxes_for_frm(feed_dict, i)

            with torch.no_grad():
                batch_ind = i + torch.zeros(boxes.size(0), 1, dtype=boxes.dtype, device=boxes.device)

                # generate a "full-image" bounding box
                image_h, image_w = input.size(2) * self.downsample_rate, input.size(3) * self.downsample_rate
                image_box = torch.cat([
                    torch.zeros(boxes.size(0), 1, dtype=boxes.dtype, device=boxes.device),
                    torch.zeros(boxes.size(0), 1, dtype=boxes.dtype, device=boxes.device),
                    image_w + torch.zeros(boxes.size(0), 1, dtype=boxes.dtype, device=boxes.device),
                    image_h + torch.zeros(boxes.size(0), 1, dtype=boxes.dtype, device=boxes.device)
                ], dim=-1)

                # meshgrid to obtain the subject and object bounding boxes
                sub_id, obj_id = jactorch.meshgrid(torch.arange(boxes.size(0), dtype=torch.int64, device=boxes.device), dim=0)
                sub_id, obj_id = sub_id.contiguous().view(-1), obj_id.contiguous().view(-1)
                sub_box, obj_box = jactorch.meshgrid(boxes, dim=0)
                sub_box = sub_box.contiguous().view(boxes.size(0) ** 2, 4)
                obj_box = obj_box.contiguous().view(boxes.size(0) ** 2, 4)

                # union box
                union_box = functional.generate_union_box(sub_box, obj_box)
                rel_batch_ind = i + torch.zeros(union_box.size(0), 1, dtype=boxes.dtype, device=boxes.device)

                # intersection maps
                box_context_imap = functional.generate_intersection_map(boxes, image_box, self.pool_size)
                sub_union_imap = functional.generate_intersection_map(sub_box, union_box, self.pool_size)
                obj_union_imap = functional.generate_intersection_map(obj_box, union_box, self.pool_size)

            this_context_features = self.context_roi_pool(context_features, torch.cat([batch_ind, image_box], dim=-1))
            x, y = this_context_features.chunk(2, dim=1)
            this_object_features = self.object_feature_fuse(torch.cat([
                self.object_roi_pool(object_features, torch.cat([batch_ind, boxes], dim=-1)),
                x, y * box_context_imap
            ], dim=1))

            this_relation_features = self.relation_roi_pool(relation_features, torch.cat([rel_batch_ind, union_box], dim=-1))
            x, y, z = this_relation_features.chunk(3, dim=1)
            this_relation_features = self.relation_feature_fuse(torch.cat([
                this_object_features[sub_id], this_object_features[obj_id],
                x, y * sub_union_imap, z * obj_union_imap
            ], dim=1))

            if DEBUG:
                outputs.append([
                    None,
                    this_object_features,
                    this_relation_features
                ])
            else:
                outputs.append([
                    None,
                    self._norm(self.object_feature_fc(this_object_features.view(boxes.size(0), -1))),
                    self._norm(self.relation_feature_fc(this_relation_features.view(boxes.size(0) * boxes.size(0), -1)).view(boxes.size(0), boxes.size(0), -1)),
                    tube_id_list 
                ])

        outputs_new = self.merge_tube_obj_ftr(outputs, feed_dict)
        return outputs_new

    def _norm(self, x):
        return x / (x.norm(2, dim=-1, keepdim=True)+1e-7)

