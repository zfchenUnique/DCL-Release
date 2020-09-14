from clevrer.utils import pickledump, pickleload, jsonload, compute_LS, compute_IoU_v2, compute_union_box 
import numpy as np
import os 
import pdb
from sklearn.metrics import average_precision_score
import torch

RETRIEVAL_TYPE=['object', 'event_in', 'event_out', 'event_collision']

def prepare_retrieval_exp(retrieval_info):
    for exp_id, exp_info in enumerate(retrieval_info['expressions']):
        exp_info['question_id'] = exp_id
        exp_info['question_type'] = 'retrieval'
        exp_info['question_subtype'] = exp_info['expression_family']
    return retrieval_info['expressions']

def evaluate_retrieval():
    opt = load_options()
    monitors = {}
    retrieval_info = jsonload(opt['expression_path'])
    ques_info_list = prepare_retrieval_exp(retrieval_info)
    grounding_result_path = opt['retrieval_result_path']
    ground_thre = opt['ground_thre']
    frm_thre = opt['frm_thre']
    acc_w = 1
    num_video = opt['end_index'] - opt['start_index']
    num_exp = len(ques_info_list)
    score_matrix = np.zeros((num_video, num_exp))
    #monitors = compute_mAP(score_matrix, retrieval_info, monitors, opt)
    for vid in range(opt['start_index'], opt['end_index']): 
        test_result_path = os.path.join(grounding_result_path, str(vid)+'.pk')
        gt_tube_full_path = os.path.join(opt['tube_gt_path'], 'annotation_'+str(vid).zfill(5)+'.pk')
        prp_full_path = os.path.join(opt['tube_prp_path'], 'proposal_' + str(vid).zfill(5)+'.pk') 
        test_result = pickleload(test_result_path)
        tube_gt_info = pickleload(gt_tube_full_path)['tubes']
        tube_prp_info = pickleload(prp_full_path)['tubes']
        
        answers = test_result['answer'] 
        gts = test_result['gt']
        pos_id_list = retrieval_info['vid2exp'][str(vid)]
        #pdb.set_trace()
        question_type_new = 'retrieval'
        for i, tmp_answer in enumerate(answers):
            query_type, a = tmp_answer 
            acc_w = 1
            j = i 
            gt = ques_info_list[j]['answer']
            question_sub_type = ques_info_list[j]['question_subtype']

            if question_type_new=='retrieval' and question_sub_type.startswith('object'):
                if isinstance(a, str) and a=='error':
                    prp_score = -10.0
                else:
                    prp_score = torch.max(a)
                correct_flag = 0
                if i in pos_id_list and prp_score>0:
                    correct_flag =1
                elif i not in pos_id_list and prp_score<0:
                    correct_flag =1
                pos_sample = 0
                if i in pos_id_list:
                    pos_sample =1

            elif question_type_new=='retrieval' and \
                    (question_sub_type.startswith('event_in') or question_sub_type.startswith('event_out')):
                if isinstance(a, str) and a=='error':
                    prp_score = -10.0
                else:
                    prp_score = torch.max(a[0])
                correct_flag = 0
                if i in pos_id_list and prp_score>0:
                    correct_flag =1
                elif i not in pos_id_list and prp_score<0:
                    correct_flag =1
                pos_sample = 0
                if i in pos_id_list:
                    pos_sample =1
            elif question_type_new=='retrieval' and \
                    question_sub_type.startswith('event_collision'):
                if isinstance(a, str) and a=='error':
                    prp_score = -10.0
                else:
                    prp_score = torch.max(a[0])
                correct_flag = 0
                if i in pos_id_list and prp_score>0:
                    correct_flag =1
                elif i not in pos_id_list and prp_score<0:
                    correct_flag =1
                pos_sample = 0
                if i in pos_id_list:
                    pos_sample =1

            score_matrix[vid-opt['start_index'], i] = float(prp_score)

            key = 'acc/qa/' + query_type
            new_key = 'acc/qa/' + question_type_new            
            
            if question_type_new=='retrieval' and question_sub_type.startswith('object'):
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

    monitors = compute_video_mAP(score_matrix, retrieval_info, monitors, opt)
    monitors = compute_text_mAP(score_matrix, retrieval_info, monitors, opt)

    print_monitors(monitors)
    pdb.set_trace()
    return monitors 

def compute_text_mAP(score_matrix, retrieval_info, monitors, opt):  
    num_video, num_exp  = score_matrix.shape
    text_ap_list = []
    sub_type_ap_list = [[] for retri_type in RETRIEVAL_TYPE]
    for exp_id in range(num_exp):
        tmp_score = score_matrix[:,exp_id]
        pos_vid_list = [vid_info[0] for vid_info in retrieval_info['expressions'][exp_id]['answer']] 
        tmp_label = []
        for vid in range(opt['start_index'], opt['end_index']):
            if vid in pos_vid_list:
                tmp_label.append(1)
            else:
                tmp_label.append(0)
        tmp_label = np.array(tmp_label)
        tmp_ap = average_precision_score(tmp_label, tmp_score)
        # To handle expression that didn't contain positive expressions
        if np.isnan(tmp_ap):
            continue

        for sub_idx, sub_type in enumerate(RETRIEVAL_TYPE):
            if retrieval_info['expressions'][exp_id]['question_subtype'].startswith(sub_type):

                sub_type_ap_list[sub_idx].append((tmp_ap, 1))
        text_ap_list.append((tmp_ap, 1))
        #pdb.set_trace()
    monitors['acc/text/mAP'] = text_ap_list
    for sub_idx, sub_type in enumerate(RETRIEVAL_TYPE):
        monitors['acc/text/mAP/'+sub_type] = sub_type_ap_list[sub_idx]
    return monitors 

def compute_video_mAP(score_matrix, retrieval_info, monitors, opt):  
    num_video, num_exp  = score_matrix.shape
    video_ap_list = []
    for vid in range(opt['start_index'], opt['end_index']):
        vid_2 = vid - opt['start_index']
        tmp_score = score_matrix[vid_2]
        tmp_label = []
        for exp_id in range(num_exp):
            if exp_id in retrieval_info['vid2exp'][str(vid)]:
                tmp_label.append(1)
            else:
                tmp_label.append(0)
        tmp_label = np.array(tmp_label)
        tmp_ap = average_precision_score(tmp_label, tmp_score)
        video_ap_list.append((tmp_ap, 1))
    monitors['acc/video/mAP'] = video_ap_list
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
    #retrieval_result_path = 'dumps/retrieval_cache_prp'
    retrieval_result_path = 'dumps/retrieval_cache_vis_sup'
    tube_gt_path = '../clevrer/tubeProposalsGt'
    tube_prp_path = '../clevrer/tubeProposalsAttrMatchNoIoUThre/1.0_1.0_0.6_0.7'
    expression_path = '/home/zfchen/code/nsclClevrer/clevrer/expressions/exp_val_retrieval_v4/5000_500_0/refine_retrieval_exp.json'

    opt['retrieval_result_path'] = retrieval_result_path
    opt['tube_gt_path'] = tube_gt_path
    opt['tube_prp_path'] = tube_prp_path
    opt['start_index'] = 10000
    opt['end_index'] = 11000
    opt['expression_path'] = expression_path 
    opt['ground_thre'] = 0.5
    opt['frm_thre'] = 10
    return opt 

if __name__=='__main__':
    evaluate_retrieval()
