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


def decode_mask_to_xyxy(mask):
    bbx_xyxy = cocoMask.toBbox(mask)
    bbx_xyxy[2] =  bbx_xyxy[2] + bbx_xyxy[0]
    bbx_xyxy[3] =  bbx_xyxy[3] + bbx_xyxy[1]
    return bbx_xyxy  

def transform_conpcet_forms_for_nscl(pg_list):
    nsclseq = clevrer_to_nsclseq(pg_list)
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

