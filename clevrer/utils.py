import pickle
import json
import sys
import pycocotools.mask as mask
import copy
import pycocotools.mask as cocoMask
import numpy as np
import torch
import os
import cv2
import pdb
from collections import defaultdict
from nscl.datasets.definition import gdef

COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'yellow', 'cyan', 'purple']
MATERIALS = ['metal', 'rubber']
SHAPES = ['sphere', 'cylinder', 'cube']
ORDER  = ['first', 'second', 'last']
ALL_CONCEPTS= COLORS + MATERIALS + SHAPES + ORDER 

def _norm(x, dim=-1):
    return x / (x.norm(2, dim=dim, keepdim=True)+1e-7)

def prepare_future_prediction_input(feed_dict, f_sng, args):
    """"
    attr: obj_num, attr_dim, 1, 1 (None)
    x: obj_num, state_dim*(n_his+1)
    rel: return from prepare_relations
    label_obj: obj_num, state_dim, 1 , 1
    label_rel: obj_num * obj_num, rela_dim, 1, 1
    """""
    x_step = args.n_his +1 
    last_frm_id_list = [frm_id for frm_id in feed_dict['tube_info']['frm_list'][-args.n_his-1:]]
    obj_num, ftr_t_dim = f_sng[3].shape
    ftr_dim = f_sng[1].shape[-1]
    box_dim = 4
    t_dim = ftr_t_dim//box_dim
    spatial_seq = f_sng[3].view(obj_num, t_dim, box_dim)
    tmp_box_list = [spatial_seq[:, frm_id] for frm_id in last_frm_id_list]
    x_box = torch.stack(tmp_box_list, dim=1).contiguous().view(obj_num, args.n_his+1, box_dim)  
    x_ftr = f_sng[0][:, -x_step:] .view(obj_num, x_step, ftr_dim)
    x = torch.cat([x_box, x_ftr], dim=2).view(obj_num, x_step*(ftr_dim+box_dim), 1, 1).contiguous()


    # obj_num*obj_num, box_dim*total_step, 1, 1
    spatial_rela = extract_spatial_relations(x_box.view(obj_num, x_step, box_dim))
    ftr_rela = f_sng[2][:, :, -x_step:].view(obj_num*obj_num, x_step*ftr_dim, 1, 1) 
    rela = torch.cat([spatial_rela, ftr_rela], dim=1)
    rel = prepare_relations(obj_num)
    for idx in range(len(rel)-2):
        rel[idx] = rel[idx].to(ftr_rela.device)
    rel.append(rela)
    attr = None 
    node_r_idx, node_s_idx, Ra = rel[3], rel[4], rel[5]
    Rr_idx, Rs_idx, value = rel[0], rel[1], rel[2]

    Rr = torch.sparse.FloatTensor(
        Rr_idx, value, torch.Size([node_r_idx.shape[0], value.size(0)])).to(ftr_rela.device)
    Rs = torch.sparse.FloatTensor(
        Rs_idx, value, torch.Size([node_s_idx.shape[0], value.size(0)])).to(ftr_rela.device)

    return attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx 

def prepare_counterfact_prediction_input(feed_dict, f_sng, args):
    """"
    attr: obj_num, attr_dim, 1, 1 (None)
    x: obj_num, state_dim*(n_his+1)
    rel: return from prepare_relations
    label_obj: obj_num, state_dim, 1 , 1
    label_rel: obj_num * obj_num, rela_dim, 1, 1
    """""
    x_step = args.n_his +1 
    first_id_list = [frm_id for frm_id in feed_dict['tube_info']['frm_list'][:x_step]]
    obj_num, ftr_t_dim = f_sng[3].shape
    ftr_dim = f_sng[1].shape[-1]
    box_dim = 4
    t_dim = ftr_t_dim//box_dim
    spatial_seq = f_sng[3].view(obj_num, t_dim, box_dim)
    tmp_box_list = [spatial_seq[:, frm_id].clone() for frm_id in first_id_list]
    x_box = torch.stack(tmp_box_list, dim=1).contiguous().view(obj_num, x_step, box_dim)  
    x_ftr = f_sng[0][:, :x_step].view(obj_num, x_step, ftr_dim).clone()
    x = torch.cat([x_box, x_ftr], dim=2).view(obj_num, x_step*(ftr_dim+box_dim), 1, 1).contiguous()

    # obj_num*obj_num, box_dim*total_step, 1, 1
    spatial_rela = extract_spatial_relations(x_box.view(obj_num, x_step, box_dim))
    ftr_rela = f_sng[2][:, :, :x_step].view(obj_num*obj_num, x_step*ftr_dim, 1, 1) 
    rela = torch.cat([spatial_rela, ftr_rela], dim=1)
    rel = prepare_relations(obj_num)
    for idx in range(len(rel)-2):
        rel[idx] = rel[idx].to(ftr_rela.device)
    rel.append(rela)
    attr = None 
    node_r_idx, node_s_idx, Ra = rel[3], rel[4], rel[5]
    Rr_idx, Rs_idx, value = rel[0], rel[1], rel[2]

    Rr = torch.sparse.FloatTensor(
        Rr_idx, value, torch.Size([node_r_idx.shape[0], value.size(0)])).to(ftr_rela.device)
    Rs = torch.sparse.FloatTensor(
        Rs_idx, value, torch.Size([node_s_idx.shape[0], value.size(0)])).to(ftr_rela.device)

    return attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx 


def prepare_relations(n):
    node_r_idx = np.arange(n)
    node_s_idx = np.arange(n)

    rel = np.zeros((n**2, 2))
    rel[:, 0] = np.repeat(np.arange(n), n)
    rel[:, 1] = np.tile(np.arange(n), n)

    n_rel = rel.shape[0]
    Rr_idx = torch.LongTensor([rel[:, 0], np.arange(n_rel)])
    Rs_idx = torch.LongTensor([rel[:, 1], np.arange(n_rel)])
    value = torch.FloatTensor([1] * n_rel)

    rel = [Rr_idx, Rs_idx, value, node_r_idx, node_s_idx]
    return rel

def extract_spatial_relations(feats):
    """
    Extract spatial relations
    """
    ### prepare relation attributes
    n_objects, t_frame, box_dim = feats.shape
    feats = feats.view(n_objects, t_frame*box_dim, 1, 1)
    n_relations = n_objects * n_objects
    relation_dim =  box_dim
    state_dim = box_dim 
    Ra = torch.ones([n_relations, relation_dim *t_frame, 1, 1], device=feats.device) * -0.5

    #change to relative position
    #  relation_dim = self.args.relation_dim
    #  state_dim = self.args.state_dim
    for i in range(n_objects):
        for j in range(n_objects):
            idx = i * n_objects + j
            Ra[idx, 0::relation_dim] = feats[i, 0::state_dim] - feats[j, 0::state_dim]  # x
            Ra[idx, 1::relation_dim] = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # y
            Ra[idx, 2::relation_dim] = feats[i, 2::state_dim] - feats[j, 2::state_dim]  # h
            Ra[idx, 3::relation_dim] = feats[i, 3::state_dim] - feats[j, 3::state_dim]  # w
    return Ra

def predict_counterfact_features_v2(model, feed_dict, f_sng, args, counter_fact_id):
    data = prepare_counterfact_prediction_input(feed_dict, f_sng, args)
    #x: obj_num, state_dim*(n_his+1)
    x_step = args.n_his + 1
    attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx = data
    n_objects_ori = x.shape[0]
   
    for i in range(n_objects_ori):
        for j in range(n_objects_ori):
            idx = i * n_objects_ori + j
            if i==counter_fact_id or j==counter_fact_id:
                Ra[idx] = 0.0
    x[counter_fact_id] = 0.0

    pred_obj_list = []
    pred_rel_list = []
    for t_step in range(args.n_his+1):
        pred_obj_list.append(x[:,t_step*args.state_dim:(t_step+1)*args.state_dim])
        pred_rel_list.append(Ra[:,t_step*args.relation_dim:(t_step+1)*args.relation_dim])
    relation_dim = args.relation_dim
    state_dim = args.state_dim 
    box_dim = 4
    for p_id, frm_id  in enumerate(range(0, args.n_seen_frames, args.frame_offset)):
        x = torch.cat(pred_obj_list[p_id:p_id+x_step], dim=1) 
        Ra = torch.cat(pred_rel_list[p_id:p_id+x_step], dim=1) 

        valid_object_id_list = check_valid_object_id_list(x, args)

        if counter_fact_id in valid_object_id_list:
            counter_idx = valid_object_id_list.index(counter_fact_id)
            del valid_object_id_list[counter_idx]

        if len(valid_object_id_list) == 0:
            break 
        data_valid = prepare_valid_input(x, Ra, valid_object_id_list, args)
        attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx = data_valid 
        n_objects = x.shape[0]

        feats = x
        # update relation
        for i in range(n_objects):
            for j in range(n_objects):
                idx = i * n_objects + j
                Ra[idx, 0::relation_dim] = feats[i, 0::state_dim] - feats[j, 0::state_dim]  # x
                Ra[idx, 1::relation_dim] = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # y
                Ra[idx, 2::relation_dim] = feats[i, 2::state_dim] - feats[j, 2::state_dim]  # h
                Ra[idx, 3::relation_dim] = feats[i, 3::state_dim] - feats[j, 3::state_dim]  # w

        pred_obj_valid, pred_rel_valid = model._model_pred(
            attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx, args.pstep)
        
        pred_obj = torch.zeros(n_objects_ori, state_dim, 1, 1, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        for valid_id, ori_id in enumerate(valid_object_id_list):
            pred_obj[ori_id] = pred_obj_valid[valid_id]
            pred_obj[ori_id, box_dim:] = _norm(pred_obj_valid[valid_id, box_dim:], dim=0)
        pred_rel = torch.zeros(n_objects_ori*n_objects_ori, relation_dim, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        
        for valid_id, ori_id in enumerate(valid_object_id_list):
            for valid_id_2, ori_id_2 in enumerate(valid_object_id_list):
                valid_idx = valid_id * n_objects + valid_id_2 
                ori_idx = ori_id * n_objects + ori_id_2
                pred_rel[ori_idx] = pred_rel_valid[valid_idx]
                pred_rel[ori_id, box_dim:] = _norm(pred_rel_valid[valid_idx, box_dim:], dim=0)

        pred_obj_list.append(pred_obj)
        pred_rel_list.append(pred_rel.view(n_objects_ori*n_objects_ori, relation_dim, 1, 1)) 
    #make the output consitent with video scene graph
    pred_frm_num = len(pred_obj_list) 
    ftr_dim = f_sng[1].shape[1]
    box_dim = 4
    box_ftr = torch.stack(pred_obj_list[-pred_frm_num:], dim=1)[:, :, :box_dim].contiguous().view(n_objects_ori, pred_frm_num, box_dim) 
    #visualize_prediction(box_ftr, feed_dict, whatif_id=counter_fact_id, store_img=True, args=args)
    #pdb.set_trace()
    rel_ftr_exp = torch.stack(pred_rel_list[-pred_frm_num:], dim=1)[:, :, box_dim:].contiguous().view(n_objects_ori, n_objects_ori, pred_frm_num, ftr_dim)
    return None, None, rel_ftr_exp, box_ftr.view(n_objects_ori, -1)  

def predict_counterfact_features(model, feed_dict, f_sng, args, counter_fact_id):
    data = prepare_counterfact_prediction_input(feed_dict, f_sng, args)
    #x: obj_num, state_dim*(n_his+1)
    x_step = args.n_his + 1
    attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx = data
    

    pred_obj_list = []
    pred_rel_list = []
    for t_step in range(args.n_his+1):
        pred_obj_list.append(x[:,t_step*args.state_dim:(t_step+1)*args.state_dim])
        pred_rel_list.append(Ra[:,t_step*args.relation_dim:(t_step+1)*args.relation_dim])
    n_objects = x.shape[0]
    relation_dim = args.relation_dim
    state_dim = args.state_dim 
    for p_id, frm_id  in enumerate(range(0, args.n_seen_frames, args.frame_offset)):
        x = torch.cat(pred_obj_list[p_id:p_id+x_step], dim=1) 
        Ra = torch.cat(pred_rel_list[p_id:p_id+x_step], dim=1) 

        feats = x
        # update relation
        for i in range(n_objects):
            for j in range(n_objects):
                idx = i * n_objects + j
                Ra[idx, 0::relation_dim] = feats[i, 0::state_dim] - feats[j, 0::state_dim]  # x
                Ra[idx, 1::relation_dim] = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # y
                Ra[idx, 2::relation_dim] = feats[i, 2::state_dim] - feats[j, 2::state_dim]  # h
                Ra[idx, 3::relation_dim] = feats[i, 3::state_dim] - feats[j, 3::state_dim]  # w

        # masking out counter_fact_id 
        x[counter_fact_id] = -1.0
        for i in range(n_objects):
            for j in range(n_objects):
                idx = i * n_objects + j
                if i==counter_fact_id or j==counter_fact_id:
                    Ra[idx] = -1.0

        pred_obj, pred_rel = model._model_pred(
            attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx, args.pstep)
        pred_obj_list.append(pred_obj)
        pred_rel_list.append(pred_rel.view(n_objects*n_objects, relation_dim, 1, 1)) 
    #make the output consitent with video scene graph
    pred_frm_num = len(pred_obj_list) 
    ftr_dim = f_sng[1].shape[1]
    box_dim = 4
    box_ftr = torch.stack(pred_obj_list[-pred_frm_num:], dim=1)[:, :, :box_dim].contiguous().view(n_objects, pred_frm_num, box_dim) 
    rel_ftr_exp = torch.stack(pred_rel_list[-pred_frm_num:], dim=1)[:, :, box_dim:].contiguous().view(n_objects, n_objects, pred_frm_num, ftr_dim)
    #pdb.set_trace()
    return None, None, rel_ftr_exp, box_ftr.view(n_objects, -1)  

def predict_future_feature(model, feed_dict, f_sng, args):
    data = prepare_future_prediction_input(feed_dict, f_sng, args)
    #x: obj_num, state_dim*(n_his+1)
    x_step = args.n_his + 1
    attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx = data
    pred_obj_list = []
    pred_rel_list = []
    for t_step in range(args.n_his+1):
        pred_obj_list.append(x[:,t_step*args.state_dim:(t_step+1)*args.state_dim])
        pred_rel_list.append(Ra[:,t_step*args.relation_dim:(t_step+1)*args.relation_dim])

    n_objects = x.shape[0]
    relation_dim = args.relation_dim
    state_dim = args.state_dim 
    for p_id in range(args.pred_frm_num):
        x = torch.cat(pred_obj_list[p_id:p_id+x_step], dim=1) 
        Ra = torch.cat(pred_rel_list[p_id:p_id+x_step], dim=1) 
        feats = x
        # update relation
        for i in range(n_objects):
            for j in range(n_objects):
                idx = i * n_objects + j
                Ra[idx, 0::relation_dim] = feats[i, 0::state_dim] - feats[j, 0::state_dim]  # x
                Ra[idx, 1::relation_dim] = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # y
                Ra[idx, 2::relation_dim] = feats[i, 2::state_dim] - feats[j, 2::state_dim]  # h
                Ra[idx, 3::relation_dim] = feats[i, 3::state_dim] - feats[j, 3::state_dim]  # w

        pred_obj, pred_rel = model._model_pred(
            attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx, args.pstep)
        pred_obj_list.append(pred_obj)
        pred_rel_list.append(pred_rel.view(n_objects*n_objects, relation_dim, 1, 1)) 
    #make the output consitent with video scene graph
    pred_frm_num = args.pred_frm_num 
    ftr_dim = f_sng[1].shape[1]
    box_dim = 4
    box_ftr = torch.stack(pred_obj_list[-pred_frm_num:], dim=1)[:, :, :box_dim].contiguous().view(n_objects, pred_frm_num, box_dim) 
    rel_ftr_exp = torch.stack(pred_rel_list[-pred_frm_num:], dim=1)[:, :, box_dim:].contiguous().view(n_objects, n_objects, pred_frm_num, ftr_dim)
    return None, None, rel_ftr_exp, box_ftr.view(n_objects, -1)  

def check_valid_object_id_list(x, args):
    valid_object_id_list = []
    x_step  = args.n_his + 1
    box_dim = 4
    for obj_id in range(x.shape[0]):
        tmp_obj_feat = x[obj_id].view(x_step, -1)
        last_obj_box = tmp_obj_feat[-1, :box_dim]
        x_c, y_c, w, h = last_obj_box
        x1 = x_c - w*0.5
        y1 = y_c - h*0.5
        x2 = x_c + w*0.5
        y2 = y_c + h*0.5
        obj_valid = True
        if w <=0 or h<=0:
            obj_valid = False
        elif x2<=0 or y2<=0:
            obj_valid = False
        elif x1>=1 or y1>=1:
            obj_valid = False
        if obj_valid:
            valid_object_id_list.append(obj_id)
    return valid_object_id_list 

def prepare_valid_input(x, Ra, valid_object_id_list, args):
    x_valid_list = [x[obj_id] for obj_id in valid_object_id_list]
    x_valid = torch.stack(x_valid_list, dim=0)
    valid_obj_num = len(valid_object_id_list)

    rel = prepare_relations(valid_obj_num)
    for idx in range(len(rel)-2):
        rel[idx] = rel[idx].to(x_valid.device)

    n_objects = x.shape[0]
    ra_valid_list = []
    for i in range(n_objects):
        for j in range(n_objects):
            idx = i * n_objects + j
            if (i in valid_object_id_list) and (j in valid_object_id_list):
                ra_valid_list.append(Ra[idx])
    Ra_valid = torch.stack(ra_valid_list, dim=0)

    rel.append(Ra_valid)
    attr = None 
    node_r_idx, node_s_idx, Ra_valid = rel[3], rel[4], rel[5]
    Rr_idx, Rs_idx, value = rel[0], rel[1], rel[2]

    Rr = torch.sparse.FloatTensor(
        Rr_idx, value, torch.Size([node_r_idx.shape[0], value.size(0)])).to(x_valid.device)
    Rs = torch.sparse.FloatTensor(
        Rs_idx, value, torch.Size([node_s_idx.shape[0], value.size(0)])).to(x_valid.device)

    return attr, x_valid, Rr, Rs, Ra_valid, node_r_idx, node_s_idx 

def predict_future_feature_v2(model, feed_dict, f_sng, args):
    data = prepare_future_prediction_input(feed_dict, f_sng, args)
    #x: obj_num, state_dim*(n_his+1)
    x_step = args.n_his + 1
    attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx = data
    pred_obj_list = []
    pred_rel_list = []
    for t_step in range(args.n_his+1):
        pred_obj_list.append(x[:,t_step*args.state_dim:(t_step+1)*args.state_dim])
        pred_rel_list.append(Ra[:,t_step*args.relation_dim:(t_step+1)*args.relation_dim])

    n_objects_ori = x.shape[0]
    relation_dim = args.relation_dim
    state_dim = args.state_dim 
    box_dim = 4

    for p_id in range(args.pred_frm_num):
        x = torch.cat(pred_obj_list[p_id:p_id+x_step], dim=1) 
        Ra = torch.cat(pred_rel_list[p_id:p_id+x_step], dim=1) 
    
        # remove invalid object, object coordinates that has been out of size
        valid_object_id_list = check_valid_object_id_list(x, args) 
        if len(valid_object_id_list) == 0:
            break 
        data_valid = prepare_valid_input(x, Ra, valid_object_id_list, args)
        attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx = data_valid 
        n_objects = x.shape[0]
        feats = x
        # update relation
        for i in range(n_objects):
            for j in range(n_objects):
                idx = i * n_objects + j
                Ra[idx, 0::relation_dim] = feats[i, 0::state_dim] - feats[j, 0::state_dim]  # x
                Ra[idx, 1::relation_dim] = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # y
                Ra[idx, 2::relation_dim] = feats[i, 2::state_dim] - feats[j, 2::state_dim]  # h
                Ra[idx, 3::relation_dim] = feats[i, 3::state_dim] - feats[j, 3::state_dim]  # w

        # normalize data


        pred_obj_valid, pred_rel_valid = model._model_pred(
            attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx, args.pstep)
       
        pred_obj = torch.zeros(n_objects_ori, state_dim, 1, 1, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        for valid_id, ori_id in enumerate(valid_object_id_list):
            pred_obj[ori_id] = pred_obj_valid[valid_id]
            pred_obj[ori_id, box_dim:] = _norm(pred_obj_valid[valid_id, box_dim:], dim=0)
        pred_rel = torch.zeros(n_objects_ori*n_objects_ori, relation_dim, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        
        for valid_id, ori_id in enumerate(valid_object_id_list):
            for valid_id_2, ori_id_2 in enumerate(valid_object_id_list):
                valid_idx = valid_id * n_objects + valid_id_2 
                ori_idx = ori_id * n_objects + ori_id_2
                pred_rel[ori_idx] = pred_rel_valid[valid_idx]
                pred_rel[ori_id, box_dim:] = _norm(pred_rel_valid[valid_idx, box_dim:], dim=0)

        pred_obj_list.append(pred_obj)
        pred_rel_list.append(pred_rel.view(n_objects_ori*n_objects_ori, relation_dim, 1, 1)) 
    
    #make the output consitent with video scene graph
    pred_frm_num = len(pred_obj_list) 
    ftr_dim = f_sng[1].shape[1]
    box_dim = 4
    box_ftr = torch.stack(pred_obj_list[-pred_frm_num:], dim=1)[:, :, :box_dim].contiguous().view(n_objects_ori, pred_frm_num, box_dim) 
    rel_ftr_exp = torch.stack(pred_rel_list[-pred_frm_num:], dim=1)[:, :, box_dim:].contiguous().view(n_objects_ori, n_objects_ori, pred_frm_num, ftr_dim)
    #visualize_prediction(box_ftr, feed_dict, whatif_id=-1, store_img=True, args=args)
    #pdb.set_trace()
    return None, None, rel_ftr_exp, box_ftr.view(n_objects_ori, -1)  

def visualize_prediction(box_ftr, feed_dict, whatif_id=-1, store_img=False, args=None):

    # print('states', states.shape)
    # print('actions', actions.shape)
    # print(filename)

    # print(actions[:, 0, :])
    # print(states[:20, 0, :])
    filename = str(feed_dict['meta_ann']['scene_index'])
    videoname = 'dumps/'+ filename + '_' + str(int(whatif_id)) +'.avi'
    #videoname = filename + '.mp4'
    os.system('mkdir -p ' + filename)


    background_fn = '../temporal_reasoning-master/background.png'
    bg = cv2.imread(background_fn)
    H, W, C = bg.shape
    bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_AREA)

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(videoname, fourcc, 3, (W, H))
    
    scene_idx = feed_dict['meta_ann']['scene_index']
    sub_idx = int(scene_idx/1000)
    sub_img_folder = 'image_'+str(sub_idx).zfill(2)+'000-'+str(sub_idx+1).zfill(2)+'000'
    img_full_folder = os.path.join(args.frm_img_path, sub_img_folder) 

    if whatif_id == -1:
        n_frame = len(feed_dict['tube_info']['frm_list']) + box_ftr.shape[1]
    else:
        n_frame = box_ftr.shape[1] 
    padding_patch_list = []
    for i in range(n_frame):
        if whatif_id==-1:
            if i < len(feed_dict['tube_info']['frm_list']):
                frm_id = feed_dict['tube_info']['frm_list'][i]
                img_full_path = os.path.join(img_full_folder, 'video_'+str(scene_idx).zfill(5), str(frm_id+1)+'.png')
                img = cv2.imread(img_full_path)
                for tube_id in range(len(feed_dict['tube_info']['box_seq']['tubes'])):
                    tmp_box = feed_dict['tube_info']['box_seq']['tubes'][tube_id][frm_id]
                    x = float(tmp_box[0] - tmp_box[2]*0.5)
                    y = float(tmp_box[1] - tmp_box[3]*0.5)
                    w = float(tmp_box[2])
                    h = float(tmp_box[3])
                    img = cv2.rectangle(img, (int(x*W), int(y*H)), (int(x*W + w*W), int(y*H + h*H)), (36,255,12), 1)
                    cv2.putText(img, str(tube_id), (int(x*W), int(y*H)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    if i==len(feed_dict['tube_info']['frm_list'])-1:
                        padding_patch = img[int(h*H):int(y*H+h*H),int(x*W):int(W*x+w*W)]
                        hh, ww, c = padding_patch.shape
                        if hh*ww*c==0:
                            padding_patch  = np.zeros((24, 24, 3), dtype=np.float32)
                        padding_patch_list.append(padding_patch)
            else:
                pred_offset =  i - len(feed_dict['tube_info']['frm_list'])
                frm_id = feed_dict['tube_info'] ['frm_list'][-1] + (args.frame_offset*pred_offset+1)  
                img = copy.deepcopy(bg)
                for tube_id in range(box_ftr.shape[0]):
                    tmp_box = box_ftr[tube_id][pred_offset]
                    x = float(tmp_box[0] - tmp_box[2]*0.5)
                    y = float(tmp_box[1] - tmp_box[3]*0.5)
                    w = float(tmp_box[2])
                    h = float(tmp_box[3])
                    y2 = y +h
                    x2 = x +w
                    if w<=0 or h<=0:
                        continue
                    if x>1:
                        continue
                    if y>1:
                        continue
                    if x2 <=0:
                        continue
                    if y2 <=0:
                        continue 
                    if x<0:
                        x=0
                    if y<0:
                        y=0
                    if x2>1:
                        x2=1
                    if y2>1:
                        y2=1
                    patch_resize = cv2.resize(padding_patch_list[tube_id], (int(x2*W) - int(x*W), int(y2*H) - int(y*H)))
                    img[int(y*H):int(y2*H), int(x*W):int(x2*W)] = patch_resize
                    cv2.putText(img, str(tube_id), (int(x*W), int(y*H)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
            store_img = True

            if store_img:
                cv2.imwrite(os.path.join( 'tmp/%s_%d.png' % (filename, i)), img.astype(np.uint8))
        else:
            frm_id = feed_dict['tube_info']['frm_list'][i]
            img_full_path = os.path.join(img_full_folder, 'video_'+str(scene_idx).zfill(5), str(frm_id+1)+'.png')
            img_rgb = cv2.imread(img_full_path)
            #for tube_id in range(len(feed_dict['tube_info']['box_seq']['tubes'])):
            img = copy.deepcopy(bg)
            for tube_id in range(box_ftr.shape[0]):
                tmp_box = feed_dict['tube_info']['box_seq']['tubes'][tube_id][frm_id]
                x = float(tmp_box[0] - tmp_box[2]*0.5)
                y = float(tmp_box[1] - tmp_box[3]*0.5)
                w = float(tmp_box[2])
                h = float(tmp_box[3])
                img_patch = img_rgb[int(y*H):int(y*H + h*H) , int(x*W): int(x*W + w*W)]
                hh, ww, c = img_patch.shape
                if hh*ww*c==0:
                    img_patch  = np.zeros((24, 24, 3), dtype=np.float32)

                tmp_box = box_ftr[tube_id][i]
                x = float(tmp_box[0] - tmp_box[2]*0.5)
                y = float(tmp_box[1] - tmp_box[3]*0.5)
                w = float(tmp_box[2])
                h = float(tmp_box[3])
                y2 = y +h
                x2 = x +w
                if w<=0 or h<=0:
                    continue
                if x>1:
                    continue
                if y>1:
                    continue
                if x2 <=0:
                    continue
                if y2 <=0:
                    continue 
                if x<0:
                    x=0
                if y<0:
                    y=0
                if x2>1:
                    x2=1
                if y2>1:
                    y2=1
                patch_resize = cv2.resize(img_patch, (max(int(x2*W) - int(x*W), 1), max(int(y2*H) - int(y*H), 1)))
                img[int(y*H):int(y2*H), int(x*W):int(x2*W)] = patch_resize
                cv2.putText(img, str(tube_id), (int(x*W), int(y*H)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            if store_img:
                cv2.imwrite(os.path.join( 'tmp/%s_%d_%d.png' % (filename, i, int(whatif_id))), img.astype(np.uint8))
        out.write(img)

def collate_dict(batch):
    return batch

def remove_wrapper_for_paral_training(feed_dict_list):
    for feed_idx, feed_dict in enumerate(feed_dict_list):
        new_feed_fict = {}
        for key_name, value in feed_dict.items():
            if isinstance(value, torch.Tensor):
                new_value = value.squeeze(0)
            pdb.set_trace()
            new_feed_dict[key_name] = new_value

def default_reduce_func(k, v):
    if torch.is_tensor(v):
        return v.mean()
    return v

def custom_reduce_func(k, v):

    if isinstance(v, list):
        for idx in range(len(v)-1, -1, -1):
            if v[idx]<0:
                del v[idx]
        if len(v)>0:
            return sum(v)/len(v)
        else:
            return  -1
    else:
        invalid_mask = v<0
        if invalid_mask.float().sum()>0:
            pdb.set_trace()
            valid_mask = 1 - invalid_mask.float()
            valid_v = torch.sum(v*valid_mask)
            valid_num = valid_mask.sum()
            if valid_num>0:
                return valid_v/valid_num
            else:
                return -1

    if '_max' in k:
        return v.max()
    elif '_sum' in k:
        return v.sum()
    else:
        return default_reduce_func(k, v)

def decode_mask_to_xyxy(mask):
    bbx_xyxy = cocoMask.toBbox(mask)
    bbx_xyxy[2] =  bbx_xyxy[2] + bbx_xyxy[0]
    bbx_xyxy[3] =  bbx_xyxy[3] + bbx_xyxy[1]
    return bbx_xyxy  

def transform_conpcet_forms_for_nscl(pg_list):
    nsclseq = clevrer_to_nsclseq(pg_list)
    nsclqsseq  = nsclseq_to_nsclqsseq(nsclseq)
    return nsclqsseq 

def transform_conpcet_forms_for_nscl_v2(pg_list):
    nsclseq = clevrer_to_nsclseq_v2(pg_list)
    nsclqsseq  = nsclseq_to_nsclqsseq(nsclseq)
    return nsclqsseq 

def nsclseq_to_nsclqsseq(seq_program):
    qs_seq = copy.deepcopy(seq_program)
    cached = defaultdict(list)
    for sblock in qs_seq:
        for param_type in gdef.parameter_types:
            if param_type in sblock:
                sblock[param_type + '_idx'] = len(cached[param_type])
                sblock[param_type + '_values'] = cached[param_type]
                cached[param_type].append(sblock[param_type])
    return qs_seq

def get_clevrer_op_attribute(op):
    return op.split('_')[1]

def clevrer_to_nsclseq(clevr_program_ori):
    # remove useless program
    clevr_program = []
    for pg_idx, pg in enumerate(clevr_program_ori):
        if pg=='get_col_partner' and 0:
            if clevr_program[-1]=='unique':
                uni_op = clevr_program.pop()
                filter_op = clevr_program.pop()
                if filter_op.startswith('filter'):
                    attr = clevr_program.pop()
                    assert attr in ALL_CONCEPTS
                else:
                    print(clevr_program_ori)
                    pdb.set_trace()
            else:
                print(clevr_program_ori)
                pdb.set_trace()
        else:
            clevr_program.append(pg)


    nscl_program = [{'op': 'scene', 'inputs':[]}] 
    mapping = dict()
    exe_stack = []
    inputs_idx = 0
    col_idx = -1
    obj_num = 0
    obj_stack = None
    for block_id, block in enumerate(clevr_program):
        if block == 'scene':
            current = dict(op='scene')
        elif block=='filter_shape' or block=='filter_color' or block=='filter_material':
            concept = exe_stack.pop()
            if len(nscl_program)>0:
                last = nscl_program[-1]
            else:
                last = {'op': 'padding'}
            if last['op']=='filter_shape' or last['op']=='filter_color' or last['op']=='filter_material':
                last['concept'].append(concept)
            else:
                current = dict(op='filter', concept=[concept])
        elif block.startswith('filter_order'):
            concept = exe_stack.pop()
            current = dict(op=block, temporal_concept=[concept])
            if len(nscl_program)>0:
                last = nscl_program[-1]
                if last['op']=='filter_collision':
                    col_idx = inputs_idx +1 
        elif block.startswith('end'):
            current = dict(op=block, time_concept=['end'])
        elif block.startswith('start'):
            current = dict(op=block, time_concept=['start'])
        elif block.startswith('filter_collision'):
            current = dict(op='filter_collision', relational_concept=['collision'])
            col_idx = inputs_idx + 1
        elif block.startswith('filter_in') or block.startswith('filter_out'):
            concept = block.split('_')[-1]
            current = dict(op=block, time_concept=[concept])
        elif block.startswith('filter_after') or block == 'filter_before':
            concept = block.split('_')[-1]
            current = dict(op=block, time_concept=[concept])
        elif block == 'filter_stationary' or block == 'filter_moving':
            concept = block.split('_')[-1]
            current = dict(op='filter_temporal', temporal_concept=[concept])
        elif block.startswith('filter'):
            current = dict(op=block)
        elif block == 'unique' or block == 'events' or block == 'all_events' or block == 'null' or block == 'get_object':
            continue 
        elif block == 'get_frame':
            if not (nscl_program[-1]['op']=='start' or nscl_program[-1]['op']=='end'):
                continue 
            current = dict(op=block)
        elif block == 'objects': # fix bug on fitlering time
            if len(clevr_program)>(block_id+1): 
                next_op = clevr_program[block_id+1]
                if next_op=='filter_collision':
                    continue
            current = dict(op=block)
            obj_num +=1
            if obj_num>1:
                obj_stack = inputs_idx
        elif block == 'events':
            current = dict(op=block)
        elif block in ALL_CONCEPTS:
            exe_stack.append(block)
            continue 
        else:
            if block.startswith('query'):
                if block_id == len(clevr_program) - 1:
                    attribute = get_clevrer_op_attribute(block)
                    current = dict(op='query', attribute=attribute)
            elif block == 'exist':
                current = dict(op='exist')
            elif block == 'count':
                if block_id == len(clevr_program) - 1:
                    current = dict(op='count')
            else:
                current = dict(op=block)
                #raise ValueError('Unknown CLEVR operation: {}.'.format(op))

        if current is None:
            assert len(block['inputs']) == 1
        else:
            if block =='end' or block == 'start':
                current['inputs'] = []
            elif block =='get_frame':
                current['inputs'] = [inputs_idx - 1, inputs_idx ]
            elif block =='get_col_partner':
                current['inputs'] = [inputs_idx, col_idx]
            elif block == 'filter_stationary' or block == 'filter_moving':
                if obj_stack is not None: 
                    current['inputs'] = [obj_stack, inputs_idx]
                else:
                    current['inputs'] = [inputs_idx]
            else:
                current['inputs'] = [inputs_idx]
            inputs_idx +=1 
            nscl_program.append(current)

    return nscl_program

def sort_by_x(obj):
    return obj[1][0, 1, 0, 0]


def decode_mask_to_box(mask, crop_box_size, H, W):
    bbx_xywh = cocoMask.toBbox(mask)
    bbx_xyxy = copy.deepcopy(bbx_xywh)
    crop_box = copy.deepcopy(bbx_xywh)
    
    bbx_xyxy[2] =  bbx_xyxy[2] + bbx_xyxy[0]
    bbx_xyxy[3] =  bbx_xyxy[3] + bbx_xyxy[1]
    
    bbx_xywh[0] = bbx_xywh[0]*1.0/mask['size'][1] 
    bbx_xywh[2] = bbx_xywh[2]*1.0/mask['size'][1] 
    bbx_xywh[1] = bbx_xywh[1]*1.0/mask['size'][0] 
    bbx_xywh[3] = bbx_xywh[3]*1.0/mask['size'][0] 
    bbx_xywh[0] = bbx_xywh[0] + bbx_xywh[2]/2.0 
    bbx_xywh[1] = bbx_xywh[1] + bbx_xywh[3]/2.0 

    crop_box[1] = int((bbx_xyxy[0])*W/mask['size'][1]) # w
    crop_box[0] = int((bbx_xyxy[1])*H/mask['size'][0]) # h
    crop_box[2] = int(crop_box_size[0])
    crop_box[3] = int(crop_box_size[1])


    ret = np.ones((4, crop_box_size[0], crop_box_size[1]))
    ret[0, :, :] *= bbx_xywh[0]
    ret[1, :, :] *= bbx_xywh[1]
    ret[2, :, :] *= bbx_xywh[2]
    ret[3, :, :] *= bbx_xywh[3]
    ret = torch.FloatTensor(ret)
    return bbx_xyxy, ret, crop_box.astype(int)   


def mapping_obj_ids_to_tube_ids(objects, tubes, frm_id ):
    obj_id_to_map_id = {}
    fix_ids = []
    for obj_id, obj_info in enumerate(objects):
        bbox_xyxy, xyhw_exp, crop_box = decode_mask_to_box(objects[obj_id]['mask'], [24, 24], 100, 150)
        tube_id = get_tube_id_from_bbox(bbox_xyxy, frm_id, tubes)
        obj_id_to_map_id[obj_id] = tube_id
        if tube_id==-1:
            fix_ids.append(obj_id)

    if len(fix_ids)>0:
        fix_id = 0 # fixiong bugs invalid ids
        for t_id in range(len(tubes)):
            if t_id in obj_id_to_map_id.values():
                continue
            else:
                obj_id_to_map_id[fix_ids[fix_id]] = t_id  
                fix_id  +=1
                print('invalid tube ids!\n')
                if fix_id==len(fix_ids):
                    break 
    tube_id = len(tubes)
    for obj_id, tube_id in obj_id_to_map_id.items():
        if tube_id==-1:
            obj_id_to_map_id[obj_id] = tube_id 
            tube_id +=1
    return obj_id_to_map_id 

def check_box_in_tubes(objects, idx, tubes):

    tube_frm_boxes = [tube[idx] for tube in tubes]
    for obj_id, obj_info in enumerate(objects):
        box_xyxy = decode_box(obj_info['mask'])
        if list(box_xyxy) not in tube_frm_boxes:
            return False
    return True

def decode_box(obj_info):
    bbx_xywh = mask.toBbox(obj_info)
    bbx_xyxy = copy.deepcopy(bbx_xywh)
    bbx_xyxy[2] =  bbx_xyxy[2] + bbx_xyxy[0]
    bbx_xyxy[3] =  bbx_xyxy[3] + bbx_xyxy[1]
    return bbx_xyxy 

def set_debugger():
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(call_pdb=True)

def get_tube_id_from_bbox(bbox_xyxy, frame_id, tubes):
    for tube_id, tube_info in enumerate(tubes):
        if tube_info[frame_id]==list(bbox_xyxy):
            return tube_id
    return -1

def get_tube_id_from_bbox(bbox_xyxy, frame_id, tubes):
    for tube_id, tube_info in enumerate(tubes):
        if tube_info[frame_id]==list(bbox_xyxy):
            return tube_id
    return -1

def checking_duplicate_box_among_tubes(frm_list, tubes):
    """
    checking boxes that are using by different tubes
    """
    valid_flag=False
    for frm_idx, frm_id in enumerate(frm_list):
        for tube_id, tube_info in enumerate(tubes):
            tmp_box = tube_info[frm_id] 
            for tube_id2 in range(tube_id+1, len(tubes)):
                if tmp_box==tubes[tube_id2][frm_id]:
                    valid_flag=True
                    return valid_flag
    return valid_flag 

def check_object_inconsistent_identifier(frm_list, tubes):
    """
    checking whether boxes are lost during the track
    """
    valid_flag = False
    for tube_id, tube_info in enumerate(tubes):
        if tube_info[frm_list[0]]!=[0,0,1,1]:
            for tmp_id in range(1, len(frm_list)):
                tmp_frm = frm_list[tmp_id]
                if tube_info[tmp_frm]==[0, 0, 1, 1]:
                    valid_flag=True
                    return valid_flag 
    return valid_flag 

def jsonload(path):
    f = open(path)
    this_ans = json.load(f)
    f.close()
    return this_ans

def jsondump(path, this_dic):
    f = open(path, 'w')
    this_ans = json.dump(this_dic, f)
    f.close()

def pickleload(path):
    f = open(path, 'rb')
    this_ans = pickle.load(f)
    f.close()
    return this_ans

def pickledump(path, this_dic):
    f = open(path, 'wb')
    this_ans = pickle.dump(this_dic, f)
    f.close()

def clevrer_to_nsclseq_v2(clevr_program_ori):
    # remove useless program
    clevr_program = []
    for pg_idx, pg in enumerate(clevr_program_ori):
        clevr_program.append(pg)


    nscl_program = [{'op': 'scene', 'inputs':[]}] 
    mapping = dict()
    exe_stack = []
    inputs_idx = 0
    col_idx = -1
    obj_num = 0
    obj_stack = None
    buffer_for_ancestor = []
    for block_id, block in enumerate(clevr_program):
        if block == 'query_collision_partner':
            block = 'get_col_partner'
        if block == 'query_frame':
            block = 'get_frame'
        if block == 'filter_counterfact':
            block = 'get_counterfact'
        if block == 'query_object':
            block = 'get_object'
        if block == 'filter_start':
            block = 'start'
        if block == 'filter_end':
            block = 'end'

        if block == 'scene':
            current = dict(op='scene')
        elif block=='filter_shape' or block=='filter_color' or block=='filter_material':
            if len(exe_stack)==0:
                print('fail to parse program!')
                print(clevr_program)
                print(block_id)
                continue 
            concept = exe_stack.pop()
            if len(nscl_program)>0:
                last = nscl_program[-1]
            else:
                last = {'op': 'padding'}
            if last['op']=='filter_shape' or last['op']=='filter_color' or last['op']=='filter_material':
                last['concept'].append(concept)
            else:
                current = dict(op='filter', concept=[concept])
        elif block.startswith('filter_order'):
            concept = exe_stack.pop()
            current = dict(op=block, temporal_concept=[concept])
            if len(nscl_program)>0:
                last = nscl_program[-1]
                if last['op']=='filter_collision':
                    col_idx = inputs_idx +1 
        elif block.startswith('end'):
            current = dict(op=block, time_concept=['end'])
        elif block.startswith('start'):
            current = dict(op=block, time_concept=['start'])
        elif block.startswith('filter_collision'):
            current = dict(op='filter_collision', relational_concept=['collision'])
            buffer_for_ancestor.append(inputs_idx)
            buffer_for_ancestor.append(inputs_idx+1)
            col_idx = inputs_idx + 1
        elif block.startswith('filter_in') or block.startswith('filter_out'):
            concept = block.split('_')[-1]
            current = dict(op=block, time_concept=[concept])
            buffer_for_ancestor.append(inputs_idx)
            buffer_for_ancestor.append(inputs_idx+1)
        elif block.startswith('filter_after') or block == 'filter_before':
            concept = block.split('_')[-1]
            current = dict(op=block, time_concept=[concept])
        elif block == 'filter_stationary' or block == 'filter_moving':
            concept = block.split('_')[-1]
            current = dict(op='filter_temporal', temporal_concept=[concept])
        elif block.startswith('filter'):
            current = dict(op=block)
        elif block == 'unique'  or block == 'all_events' or block == 'null' or block == 'get_object':
            continue 
        elif block == 'get_frame':
            if not (nscl_program[-1]['op']=='start' or nscl_program[-1]['op']=='end'):
                continue 
            current = dict(op=block)
        elif block == 'objects': # fix bug on fitlering time
            if len(clevr_program)>(block_id+1): 
                next_op = clevr_program[block_id+1]
                if next_op=='filter_collision':
                    continue
            current = dict(op=block)
            obj_num +=1
            if obj_num>1:
                obj_stack = inputs_idx
        elif block in ALL_CONCEPTS:
            exe_stack.append(block)
            continue
        elif block == 'filter_ancestor':
            current = dict(op=block)
        else:
            if block.startswith('query'):
                if block_id == len(clevr_program) - 1:
                    attribute = get_clevrer_op_attribute(block)
                    current = dict(op='query', attribute=attribute)
            elif block == 'exist':
                current = dict(op='exist')
            elif block == 'count':
                if block_id == len(clevr_program) - 1:
                    current = dict(op='count')
            else:
                current = dict(op=block)

        if current is None:
            assert len(block['inputs']) == 1
        else:
            if block =='end' or block == 'start':
                current['inputs'] = []
            elif block =='get_frame':
                off_set = 0
                if len(nscl_program)>=2 and nscl_program[-2]['op']=='events':
                    off_set +=1
                current['inputs'] = [inputs_idx - 1 - off_set, inputs_idx ]
            elif block =='get_col_partner':
                current['inputs'] = [inputs_idx, col_idx]
            elif block == 'filter_stationary' or block == 'filter_moving':
                if obj_stack is not None:
                    if nscl_program[obj_stack]['op']=='events':
                        obj_stack -=1
                    current['inputs'] = [obj_stack, inputs_idx]
                else:
                    current['inputs'] = [inputs_idx]
            elif block == 'filter_ancestor':
                current['inputs'] = buffer_for_ancestor 
            else:
                current['inputs'] = [inputs_idx]
            inputs_idx +=1 
            nscl_program.append(current)

    return nscl_program





