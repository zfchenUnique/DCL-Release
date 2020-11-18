from clevrer.utils import pickledump, pickleload, jsonload, compute_LS, compute_IoU_v2, compute_union_box 
import pdb
import os 
from nscl.datasets.definition import gdef
import torch

def prepare_grounding_exp(grounding_info, vid):
    exp_info = grounding_info[str(vid)]
    query_info_list = []
    for exp_type, exp_list in exp_info.items():
        query_info_list +=exp_list
    for q_id, q_info in enumerate(query_info_list):
        query_info_list[q_id]['question_id'] = q_id
        query_info_list[q_id]['question_type'] = 'expression'
        query_info_list[q_id]['question_subtype'] = query_info_list[q_id]['expression_family'] 
    return query_info_list 

def evaluate_grounding():
    opt = load_options()
    monitors = {}
    grounding_info = jsonload(opt['expression_path'])
    grounding_result_path = opt['grounding_result_path']
    ground_thre = opt['ground_thre']
    frm_thre = opt['frm_thre']
    acc_w = 1
    for vid in range(opt['start_index'], opt['end_index']): 
        test_result_path = os.path.join(grounding_result_path, str(vid)+'.pk')
        gt_tube_full_path = os.path.join(opt['tube_gt_path'], 'annotation_'+str(vid).zfill(5)+'.pk')
        prp_full_path = os.path.join(opt['tube_prp_path'], 'proposal_' + str(vid).zfill(5)+'.pk') 
        test_result = pickleload(test_result_path)
        tube_gt_info = pickleload(gt_tube_full_path)['tubes']
        tube_prp_info = pickleload(prp_full_path)['tubes']
        
        answers = test_result['answer'] 
        gts = test_result['gt']
        ques_info_list = prepare_grounding_exp(grounding_info, vid)
        #pdb.set_trace()
        for i, tmp_answer in enumerate(answers):
            query_type, a = tmp_answer 
            j = i 
            gt = gts[i]
            question_type_new = 'expression' 
            question_sub_type = ques_info_list[i]['question_subtype']
            if question_type_new=='expression' and question_sub_type.startswith('object'):
                if isinstance(a, tuple):
                    a = a[0]
                prp_idx = torch.argmax(a) 
                prp_tube = tube_prp_info[prp_idx]
                gt_tube = tube_gt_info[gt]
                overlap = compute_LS(prp_tube, gt_tube)

            elif question_type_new=='expression' and \
                    (question_sub_type.startswith('event_in') or question_sub_type.startswith('event_out')):
                prp_idx = int(torch.argmax(a[0]))
                prp_frm_id = int(a[2][prp_idx])
                prp_frm_len = len(tube_prp_info[prp_idx])
                if prp_frm_id>=prp_frm_len:
                    prp_frm_id = prp_frm_len - 1
                prp_box = tube_prp_info[prp_idx][prp_frm_id]
                gt_idx = gt['object']
                gt_frm_id = gt['frame']
                gt_frm_len = len(tube_gt_info[gt_idx])
                if gt_frm_id>=gt_frm_len:
                    gt_frm_id = gt_frm_len - 1
                gt_box = tube_gt_info[gt_idx][gt_frm_id]
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
                prp_box1 = tube_prp_info[obj_idx1][img_frm_idx]
                prp_box2 = tube_prp_info[obj_idx2][img_frm_idx]
                prp_union_box = compute_union_box(prp_box1, prp_box2) 

                gt_idx1, gt_idx2 = gt['object']
                gt_frm_id = gt['frame']
                gt_box1 = tube_gt_info[gt_idx1][gt_frm_id]
                gt_box2 = tube_gt_info[gt_idx2][gt_frm_id]
                gt_union_box = compute_union_box(gt_box1, gt_box2) 
                
                overlap = compute_IoU_v2(prp_union_box, gt_union_box)
                frm_dist = abs(gt_frm_id-img_frm_idx)

            else:
                raise ValueError('Unknown query type: {}.'.format(response_query_type))

            key = 'acc/qa/' + query_type
            new_key = 'acc/qa/' + question_type_new            

            if question_type_new=='expression' and question_sub_type.startswith('object'):
                new_key_v2 = 'acc/mIoU/' + question_sub_type             
                new_key_v3 = 'acc/mIoU/' + question_type_new             
                new_key_object = 'acc/mIoU/object'
                new_key_acc = 'acc/qa/object'
                monitors.setdefault(key, []).append((int(overlap>=ground_thre), acc_w))
                monitors.setdefault('acc/qa', []).append((int(overlap>=ground_thre), acc_w))
                monitors.setdefault(new_key, []).append((int(overlap>=ground_thre), acc_w))
                monitors.setdefault(new_key_acc, []).append((int(overlap>=ground_thre), acc_w))
                monitors.setdefault(new_key_v2, []).append((overlap, acc_w))
                monitors.setdefault(new_key_v3, []).append((overlap, acc_w))
                monitors.setdefault(new_key_object, []).append((overlap, acc_w))
            elif question_type_new=='expression' and question_sub_type.startswith('event'):
                new_key_v2 = 'acc/mIoU/' + question_sub_type           
                new_key_v3 = 'acc/frmDist/' + question_sub_type            
                new_key_frm = 'acc/qa/' + question_sub_type 
                new_key_frm_acc = 'acc/spatial/' + question_sub_type 
                new_key_v3 = 'acc/qa/' + question_sub_type            
                monitors.setdefault(new_key_v2, []).append((overlap, acc_w))
                monitors.setdefault(new_key_v3, []).append((frm_dist, acc_w))
                monitors.setdefault(new_key_frm, []).append((int(frm_dist<frm_thre), acc_w))
                monitors.setdefault(new_key_frm_acc, []).append((int(overlap>ground_thre), acc_w))

    print_monitors(monitors)
    pdb.set_trace()
    return monitors 

def print_monitors(monitors):
    info_list = []
    for key_name, key_info in monitors.items():
        info_str = key_name + ' = '
        sum_acc = 0
        sum_val = 0
        for val_info in key_info:
            sum_acc +=val_info[0]
            sum_val +=val_info[1]
        acc = sum_acc*1.0 /sum_val
        info_str += str(acc)
        #print(info_str)
        info_list.append(info_str)
    info_list_sort = sorted(info_list)
    for ele in info_list_sort:
        print(ele)

def load_options():
    opt = {}
    #grounding_result_path = 'dumps/grounding_cache'
    grounding_result_path = 'dumps/grounding_cache_vis_sup'
    tube_gt_path = '../clevrer/tubeProposalsGt'
    tube_prp_path = '../clevrer/tubeProposalsAttrMatchNoIoUThre/1.0_1.0_0.6_0.7'
    expression_path = '/home/zfchen/code/nsclClevrer/clevrer/expressions/exp_val_grounding_v1/refine_grounding_exp.json'

    opt['grounding_result_path'] = grounding_result_path
    opt['tube_gt_path'] = tube_gt_path
    opt['tube_prp_path'] = tube_prp_path
    opt['start_index'] = 10000
    opt['end_index'] = 15000
    opt['expression_path'] = expression_path 
    opt['ground_thre'] = 0.5
    opt['frm_thre'] = 10
    return opt 

if __name__=='__main__':
    evaluate_grounding()
