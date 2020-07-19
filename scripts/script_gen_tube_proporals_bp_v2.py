from opt import parse_opt
import pdb
import os
import json
import sys
#sys.path.append('/home/zfchen/code/baselines/Jacinle/jaclearn/vision/coco')
from pycocotools.coco import COCO
import pycocotools.mask as mask
import numpy as np
import copy
import cv2
import shutil
import subprocess
import math
import pickle

COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'yellow', 'cyan', 'purple']
MATERIALS = ['metal', 'rubber']
SHAPES = ['sphere', 'cylinder', 'cube']

EPS = 1e-10
def compute_IoU_v2(bbox1, bbox2):
    bbox1_area = float((bbox1[2] - bbox1[0] + EPS) * (bbox1[3] - bbox1[1] + EPS))
    bbox2_area = float((bbox2[2] - bbox2[0] + EPS) * (bbox2[3] - bbox2[1] + EPS))
    w = max(0.0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]) + EPS)
    h = max(0.0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]) + EPS)
    inter = float(w * h)
    ovr = inter / (bbox1_area + bbox2_area - inter)
    return ovr

def compute_LS(traj, gt_traj):
    # see http://jvgemert.github.io/pub/jain-tubelets-cvpr2014.pdf
    IoU_list = []
    frm_num = 0
    for frame_ind, gt_box in enumerate(gt_traj):
        box = traj[frame_ind]
        if not (box==[0, 0, 1, 1] and gt_box==[0, 0, 1, 1]):
            frm_num +=1
        if box==[0, 0, 1, 1] or gt_box==[0, 0, 1, 1]:
            continue
        IoU_list.append(compute_IoU_v2(box, gt_box))
    return sum(IoU_list) / frm_num

def pickleload(path):
    f = open(path, 'rb')
    this_ans = pickle.load(f)
    f.close()
    return this_ans

def pickledump(path, this_dic):
    f = open(path, 'wb')
    this_ans = pickle.dump(this_dic, f)
    f.close()

def jsonload(path):
    f = open(path)
    this_ans = json.load(f)
    f.close()
    return this_ans

def jsondump(path, this_dic):
    f = open(path, 'w')
    this_ans = json.dump(this_dic, f)
    f.close()

def set_debugger():
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(call_pdb=True)

set_debugger()

def imread_if_str(img):
    if isinstance(img, str):
        img = cv2.imread(img)
    return img

def draw_rectangle(img, bbox, color=(0,0,255), thickness=3, use_dashed_line=False):
    img = imread_if_str(img)
    if isinstance(bbox, dict):
        bbox = [
            bbox['x1'],
            bbox['y1'],
            bbox['x2'],
            bbox['y2'],
        ]
    bbox[0] = max(bbox[0], 0)
    bbox[1] = max(bbox[1], 0)
    bbox[0] = min(bbox[0], img.shape[1])
    bbox[1] = min(bbox[1], img.shape[0])
    bbox[2] = max(bbox[2], 0)
    bbox[3] = max(bbox[3], 0)
    bbox[2] = min(bbox[2], img.shape[1])
    bbox[3] = min(bbox[3], img.shape[0])
    assert bbox[2] >= bbox[0]
    assert bbox[3] >= bbox[1]
    assert bbox[0] >= 0
    assert bbox[1] >= 0
    assert bbox[2] <= img.shape[1]
    assert bbox[3] <= img.shape[0]
    cur_img = copy.deepcopy(img)
    #pdb.set_trace()
    if use_dashed_line:
        drawrect(
            cur_img,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            color,
            thickness,
            'dotted'
            )
    else:
        #pdb.set_trace()
        cv2.rectangle(
            cur_img,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            color,
            thickness)
    return cur_img

def images2video(image_list, frame_rate, video_path, max_edge=None):
    TMP_DIR = '.tmp'
    FFMPEG = 'ffmpeg'
    SAVE_VIDEO = FFMPEG + ' -y -r %d -i %s/%s.jpg %s'
    
    #pdb.set_trace() 
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.mkdir(TMP_DIR)
    img_size = None
    for cur_num, cur_img in enumerate(image_list):
        cur_fname = os.path.join(TMP_DIR, '%08d.jpg' % cur_num)
        if max_edge is not None:
            cur_img = imread_if_str(cur_img)
        if isinstance(cur_img, str):
            shutil.copyfile(cur_img, cur_fname)
        elif isinstance(cur_img, np.ndarray):
            max_len = max(cur_img.shape[:2])
            #pdb.set_trace()
            if max_edge is not None and max_len > max_edge and img_size is None and max_edge is not None:
                magnif = float(max_edge) / float(max_len)
                img_size = (int(cur_img.shape[1] * magnif), int(cur_img.shape[0] * magnif))
                cur_img = cv2.resize(cur_img, img_size)
            elif max_edge is not None:
                if img_size is None:
                    magnif = float(max_edge) / float(max_len)
                    img_size = (int(cur_img.shape[1] * magnif), int(cur_img.shape[0] * magnif))
                cur_img = cv2.resize(cur_img, img_size)
            cv2.imwrite(cur_fname, cur_img)
        else:
            NotImplementedError()
    print(subprocess.getoutput(SAVE_VIDEO % (frame_rate, TMP_DIR, '%08d', video_path)))
    #pdb.set_trace() 
    shutil.rmtree(TMP_DIR)

def visTube_from_image(frmList, tube, outName):
    image_list   = list()
    for i, bbx in enumerate(tube):
        imName = frmList[i]
        img = draw_rectangle(imName, bbx)
        image_list.append(img)
        images2video(image_list, 10, outName)
    #pdb.set_trace() 

def visual_tube_proposals(results, f_dict, prp_num, opt):
    sub_idx = int(f_dict['video_index']/1000)

    jpg_folder = os.path.join(opt['img_folder_path'], 'image_%s000-%s000'%(str(sub_idx).zfill(2), \
            str(sub_idx+1).zfill(2)), 'video_%s'%(str(f_dict['video_index']).zfill(5)))
    frmImNameList = [os.path.join(jpg_folder, str(frm_info['frame_index']+1) + '.png') for frm_info in f_dict['frames']]
    frmImList = list()
    for fId, imPath  in enumerate(frmImNameList):
        img = cv2.imread(imPath)
        frmImList.append(img)
    vis_frame_num = 24
    visIner = int(len(frmImList) /vis_frame_num)
    #pdb.set_trace() 
    for ii in range(len(results[0])):
        print('visualizing tube %d\n'%(ii))
        tube = results[0][ii]
        frmImList_vis = [frmImList[iii] for iii in range(0, len(frmImList), visIner)]
        tube_vis = [tube[iii] for iii in range(0, len(frmImList), visIner)]
        #tube_vis_resize = resize_tube_bbx(tube_vis, frmImList_vis)
        vd_name_raw = 'video_%d'%(f_dict['video_index'])
        #out_sub_path = 'sample/'+vd_name_raw + '_' + str(opt['connect_w'])+'_'+str(opt['score_w'])
        out_sub_path = opt['vis_path'] + vd_name_raw + '_' + str(opt['connect_w'])+'_'+str(opt['score_w']) + '_'+ str(opt['attr_w'])
        if not os.path.isdir(out_sub_path):
            os.makedirs(out_sub_path)
        out_full_path = os.path.join(out_sub_path, str(prp_num)+'_' + str(ii)+'.gif')
        visTube_from_image(copy.deepcopy(frmImList_vis), tube_vis, out_full_path) 
        #pdb.set_trace()

def compute_IoU(box1, box2):
    KEYS = ['x1', 'y1', 'x2', 'y2']
    if isinstance(box1, list):
        box1 = {key: val for key, val in zip(KEYS, box1)}
    if isinstance(box2, list):
        box2 = {key: val for key, val in zip(KEYS, box2)}
    width = max(min(box1['x2'], box2['x2']) - max(box1['x1'], box2['x1']), 0)
    height = max(min(box1['y2'], box2['y2']) - max(box1['y1'], box2['y1']), 0)
    intersection = width * height
    box1_area = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    box2_area = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
    union = box1_area + box2_area - intersection
    return float(intersection) / (float(union) +0.000001)  # avoid overthow

def get_sub_folder_list(folder_name):
    folder_list = []
    for file_name in os.listdir(folder_name):
        full_sub_path = os.path.join(folder_name, file_name)
        if os.path.isdir(full_sub_path):
            folder_list.append(full_sub_path)
    return folder_list

def decode_mp4_to_jpg(opt):
    video_folder_path = opt['video_folder_path']
    sub_folder_list = get_sub_folder_list(video_folder_path)
    #pdb.set_trace()
    for sub_idx, full_sub_folder in enumerate(sub_folder_list):
        if 'video' not in full_sub_folder:
            continue
        img_sub_path = os.path.basename(full_sub_folder).replace('video', 'image')
        full_img_sub_path = os.path.join(opt['img_folder_path'], img_sub_path)

        for fn_idx, file_name in enumerate(os.listdir(full_sub_folder)):
            if 'mp4' not in file_name:
                continue
            full_sub_video_name = os.path.join(full_sub_folder, file_name)
            full_sub_img_out_path = os.path.join(full_img_sub_path, \
                    file_name.replace('.mp4', ''))
            
            if not os.path.isdir(full_sub_img_out_path):
                os.makedirs(full_sub_img_out_path)

            cmd = 'ffmpeg -i %s -vf fps=25 %s/' %(full_sub_video_name, full_sub_img_out_path) + '%d.png' 
            os.system(cmd)

    print('finish!\n')


def extract_tube_per_video(f_dict, opt):
    connect_w = opt['connect_w']
    score_w = opt['score_w']
    bbx_sc_list = []

    # get object number of a video
    max_obj_num = 0
    for frm_idx, frm_info in enumerate(f_dict['frames']):
        tmp_obj_num = len(frm_info['objects']) 
        if max_obj_num<tmp_obj_num:
            max_obj_num = tmp_obj_num 


    for frm_idx, frm_info in enumerate(f_dict['frames']):
        bbx_mat = []
        sc_mat = []
        color_list = []
        material_list = []
        shape_list = []
        attr_list = []
        tmp_obj_num = len(frm_info['objects']) 
        for obj_idx, obj_info in enumerate(frm_info['objects']):
            bbx_xywh = mask.toBbox(obj_info['mask'])
            bbx_xyxy = copy.deepcopy(bbx_xywh)
            bbx_xyxy[2] =  bbx_xyxy[2] + bbx_xyxy[0]
            bbx_xyxy[3] =  bbx_xyxy[3] + bbx_xyxy[1]
            bbx_mat.append(bbx_xyxy)
            sc_mat.append(obj_info['score']*score_w)
            
            if opt['use_attr_flag']:
                attr_list.append([obj_info['color'], obj_info['material'], obj_info['shape']])

        frm_size = frm_info['objects'][0]['mask']['size']
        for tmp_idx in range(tmp_obj_num, max_obj_num):
            tmp_box = np.array([0, 0, 1, 1])
            bbx_mat.append(tmp_box)
            sc_mat.append(-100)
            
            if opt['use_attr_flag']:
                attr_list.append(['', '', ''])

        bbx_mat = np.stack(bbx_mat, axis=0)
        sc_mat = np.array(sc_mat)
        sc_mat = np.expand_dims(sc_mat, axis=1 )

        #pdb.set_trace()

        if not opt['use_attr_flag']:
            bbx_sc_list.append([sc_mat, bbx_mat])
        else:
            bbx_sc_list.append([sc_mat, bbx_mat, attr_list])

    tube_list, score_list = get_tubes(bbx_sc_list, connect_w, opt['use_attr_flag'])
    return tube_list, score_list, bbx_sc_list  


def compare_attr_score(attr_list1, attr_list2):
    attr_score = 0
    for att_idx, att1 in enumerate(attr_list1):
        att2 = attr_list2[att_idx]
        if att1==att2 and att1 !='':
            attr_score +=1
    return attr_score 

def get_tubes(det_list_org, alpha, use_attr_flag=False, attr_w=1.0):
    """
    det_list_org: [score_list, bbx_list]
    alpha: connection weight
    """
    det_list = copy.deepcopy(det_list_org)
    tubes = []
    continue_flg = True
    tube_scores = []

    while continue_flg:
        timestep = 0
        obj_num = det_list[timestep][0].shape[0]
        
        if use_attr_flag:
            acc_time_list = []
            acc_attr_list = []
            for obj_id in range(obj_num):
                tmp_obj_dict = {}
                for attr_id, attr_concept in enumerate(['colors', 'materials', 'shapes']):
                    attr_upper = attr_concept.upper()
                    tmp_obj_num = 0
                    concept_dict = {}
                    for concept in globals()[attr_upper]:
                        concept_dict[concept] = 0.0
                    obj_concept = det_list[timestep][2][obj_id][attr_id]
                    concept_dict[obj_concept] = 1.0
                    tmp_obj_dict[attr_upper] = concept_dict 
                acc_attr_list.append(tmp_obj_dict) 
                acc_time_list.append(acc_attr_list)

            for t_id in range(1, len(det_list)):
                acc_time_list.append([])
                for obj_id in range(obj_num):
                    acc_time_list[t_id].append({})

        score_list = []
        score_list.append(np.zeros(det_list[timestep][0].shape[0]))
        prevind_list = []
        prevind_list.append([-1] * det_list[timestep][0].shape[0])
        timestep += 1


        while timestep < len(det_list):
            n_curbox = det_list[timestep][0].shape[0]
            n_prevbox = score_list[-1].shape[0]
            cur_scores = np.zeros(n_curbox) - np.inf
            prev_inds = [-1] * n_curbox
            for i_prevbox in range(n_prevbox):
                prevbox_coods = det_list[timestep-1][1][i_prevbox, :]
                prevbox_score = det_list[timestep-1][0][i_prevbox, 0]

                for i_curbox in range(n_curbox):
                    curbox_coods = det_list[timestep][1][i_curbox, :]
                    curbox_score = det_list[timestep][0][i_curbox, 0]
                    #try:
                    if True:
                        e_score = compute_IoU(prevbox_coods.tolist(), curbox_coods.tolist())
                        link_score = prevbox_score + curbox_score + alpha * (e_score)
                        
                        if use_attr_flag:
                            #prevbox_attr = det_list[timestep-1][2][i_prevbox]
                            det_list
                            prevbox_attr = acc_time_list[timestep-1][i_prevbox]
                            curbox_attr = det_list[timestep][2][i_curbox]
                            #attr_score = compare_attr_score(prevbox_attr, curbox_attr)
                            attr_score = 0.0
                            for attr_id, attr_concept in enumerate(['colors', 'materials', 'shapes']):
                                attr_upper = attr_concept.upper()
                                concept = curbox_attr[attr_id] 
                                if concept!='':
                                    attr_score += prevbox_attr[attr_upper][concept] 
                            attr_score  /= timestep 
                            link_score +=  attr_score * attr_w  
                    #except:
                    #    pdb.set_trace()
                    
                    cur_score = score_list[-1][i_prevbox] + link_score
                    if cur_score > cur_scores[i_curbox]:
                        cur_scores[i_curbox] = cur_score
                        prev_inds[i_curbox] = i_prevbox
                        if use_attr_flag:
                            acc_time_list[timestep][i_curbox] = copy.deepcopy(acc_time_list[timestep-1][i_prevbox])
                            curbox_attr = det_list[timestep][2][i_curbox]
                            for attr_id, attr_concept in enumerate(['colors', 'materials', 'shapes']):
                                attr_upper = attr_concept.upper()
                                concept = curbox_attr[attr_id] 
                                if concept!='':
                                    acc_time_list[timestep][i_curbox][attr_upper][concept] +=1  

            score_list.append(cur_scores)
            prevind_list.append(prev_inds)
            timestep += 1

        # get path and remove used boxes
        cur_tube = [None] * len(det_list)
        tube_score = np.max(score_list[-1]) / len(det_list)
        prev_ind = np.argmax(score_list[-1])
        timestep = len(det_list) - 1
        while timestep >= 0:
            cur_tube[timestep] = det_list[timestep][1][prev_ind, :].tolist()
            det_list[timestep][0] = np.delete(det_list[timestep][0], prev_ind, axis=0)
            det_list[timestep][1] = np.delete(det_list[timestep][1], prev_ind, axis=0)
            if use_attr_flag:
                det_list[timestep][2].pop(prev_ind)
            prev_ind = prevind_list[timestep][prev_ind]
            if det_list[timestep][1].shape[0] == 0:
                continue_flg = False
            timestep -= 1
        assert prev_ind < 0
        tubes.append(cur_tube)
        tube_scores.append(tube_score)
    return tubes, tube_scores

def extract_tube_v0(opt):
    sample_folder_path= '../clevrer/proposals'
    file_list = get_sub_file_list(sample_folder_path, '.json')
    file_list.sort()
    out_path = os.path.join(opt['tube_folder_path'] , str(opt['connect_w'])+'_'+str(opt['score_w'])+'_'+str(opt['attr_w']))
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    for file_idx, sample_file in enumerate(file_list):
        
        #if file_idx<10000:
        #    continue 

        out_fn_path = os.path.join(out_path, os.path.basename(sample_file.replace('json', 'pk')))
        if os.path.isfile(out_fn_path):
            continue 
            #pass 

        fh = open(sample_file, 'r')
        f_dict = json.load(fh)
        max_obj_num = 0
        for frm_idx, frm_info in enumerate(f_dict['frames']):
            tmp_obj_num = len(frm_info['objects']) 
            if max_obj_num<tmp_obj_num:
                max_obj_num = tmp_obj_num 

        if opt['use_attr_flag']:
            attr_dict_path = os.path.join(opt['extract_att_path'], 'attribute_' + str(file_idx).zfill(5) +'.json')
            if not os.path.isfile(attr_dict_path):
                continue 
            attr_dict_list = jsonload(attr_dict_path) 
            tube_list, score_list, bbx_sc_list = extract_tube_per_video_attribute(f_dict, opt, attr_dict_list) 
        else:
            tube_list, score_list, bbx_sc_list = extract_tube_per_video_attribute(f_dict, opt) 
        out_dict = {'tubes': tube_list, 'scores': score_list, 'bbx_list': bbx_sc_list }
        #pdb.set_trace()
        pickledump(out_fn_path, out_dict)
        if file_idx%5000==0:
            print('finish processing %d/%d videos' %(file_idx, len(file_list)))
        if file_idx<=10100 and 0:
            visual_tube_proposals([tube_list, score_list], f_dict, max_obj_num, opt)

def visual_specific_tube(opt):
    sample_folder_path= '../clevrer/proposals'
    file_list = get_sub_file_list(sample_folder_path, '.json')
    file_list.sort()
    #out_path = os.path.join(opt['tube_folder_path'] , str(opt['connect_w'])+'_'+str(opt['score_w']))
    out_path = os.path.join('../clevrer/tubeProposalsGt')

    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    
    for file_idx, sample_file in enumerate(file_list):
        #if file_idx not in [10000, 10001]:
        #if file_idx not in [10002, 10003, 10004]:
        if file_idx not in list(range(10000, 10011)):
            continue 
        #pdb.set_trace()
        out_fn_path = os.path.join(out_path, os.path.basename(sample_file.replace('json', 'pk').replace('proposal', 'annotation')))

        fh = open(sample_file, 'r')
        f_dict = json.load(fh)
        max_obj_num = 0
        for frm_idx, frm_info in enumerate(f_dict['frames']):
            tmp_obj_num = len(frm_info['objects']) 
            if max_obj_num<tmp_obj_num:
                max_obj_num = tmp_obj_num 

        out_dict = pickleload(out_fn_path)
        tube_list = out_dict['tubes']
        #score_list = out_dict['scores']
        #bbx_sc_list = out_dict['bbx_list']
        if file_idx%100==0:
            print('finish processing %d/%d videos' %(file_idx, len(file_list)))
        visual_tube_proposals([tube_list, None], f_dict, max_obj_num, opt)



def get_sub_file_list(folder_name, file_type=None):
    file_list = []
    for file_name in os.listdir(folder_name):
        full_sub_path = os.path.join(folder_name, file_name)
        if file_type is None:
            file_list.append(full_sub_path)
        elif file_type in full_sub_path:
            file_list.append(full_sub_path)
    return file_list


def extract_tube_attribute(opt):
    sample_folder_path= '../clevrer/proposals'
    file_list = get_sub_file_list(sample_folder_path, '.json')
    file_list.sort()
    out_path = os.path.join(opt['tube_folder_new_path'] , str(opt['connect_w'])+'_'+str(opt['score_w']))
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    for file_idx, sample_file in enumerate(file_list):

        if file_idx!=0:
            continue 
        #if file_idx>=100:
        #    break 
        out_fn_path = os.path.join(out_path, os.path.basename(sample_file.replace('json', 'pk')))
        #if os.path.isfile(out_fn_path):
        #    continue 

        fh = open(sample_file, 'r')
        f_dict = json.load(fh)
        max_obj_num = 0
        for frm_idx, frm_info in enumerate(f_dict['frames']):
            tmp_obj_num = len(frm_info['objects']) 
            if max_obj_num<tmp_obj_num:
                max_obj_num = tmp_obj_num 

        tube_list, score_list, bbx_sc_list = extract_tube_per_video(f_dict, opt) 
        out_dict = {'tubes': tube_list, 'scores': score_list, 'bbx_list': bbx_sc_list }
        visual_tube_proposals([tube_list, score_list], f_dict, max_obj_num, opt)
        pdb.set_trace()
        pickledump(out_fn_path, out_dict)
        if file_idx%100==0:
            print('finish processing %d/%d videos' %(file_idx, len(file_list)))

def decode_box(obj_info):
    bbx_xywh = mask.toBbox(obj_info)
    bbx_xyxy = copy.deepcopy(bbx_xywh)
    bbx_xyxy[2] =  bbx_xyxy[2] + bbx_xyxy[0]
    bbx_xyxy[3] =  bbx_xyxy[3] + bbx_xyxy[1]
    return bbx_xyxy 

def parse_tube_gt(obj_dict_gt, obj_dict_prp, opt):

    tube_num = len(obj_dict_gt['object_property'])
    tube_dict = [ [] for idx in range(tube_num)]
    for frm_idx, motion_info in enumerate(obj_dict_gt['motion_trajectory']):
        if frm_idx>=len(obj_dict_prp['frames']):
            print('Warning proposal frame num: %d, parse frame num: %d\n' %(len(obj_dict_prp['frames']), len(obj_dict_gt['motion_trajectory'])))
            continue 
        frm_prp_info =  obj_dict_prp['frames'][frm_idx]
        for obj_idx, obj_info in enumerate(motion_info['objects']):
            
            if not obj_info['inside_camera_view']:
                tube_dict[obj_idx].append([0, 0, 1, 1])
                continue 

            obj_attr_info = obj_dict_gt['object_property'][obj_idx] 

            obj_attr = [obj_attr_info['color'], obj_attr_info['material'], obj_attr_info['shape']] 
       
            max_score = 0
            max_bbx  = [0, 0, 1, 1]

            for prp_idx, prp_info in enumerate(frm_prp_info['objects']):
                prp_xyxy = decode_box(prp_info['mask']).tolist()

                prp_attr = [prp_info['color'], prp_info['material'], prp_info['shape'] ]
                attr_score = compare_attr_score(obj_attr, prp_attr)
                conf_score = prp_info['score']
                if len(tube_dict[obj_idx])==0:
                    iou_score = 0
                else:
                    iou_score = compute_IoU(tube_dict[obj_idx][-1], prp_xyxy)
                match_score = attr_score + conf_score*opt['conf_w'] + iou_score*opt['iou_w']
                if match_score>max_score:
                    max_score = match_score
                    max_bbx =  prp_xyxy
            
            tube_dict[obj_idx].append(max_bbx)

    return tube_dict 


def parse_object_track(opt):
    #pdb.set_trace()
    sub_folder_list = get_sub_folder_list(opt['ann_path'])
    ann_file_list = []
    for sub_folder in sub_folder_list:
        if 'annotation' not in sub_folder:
            continue  
        sub_file_list = get_sub_file_list(sub_folder, 'json')
        ann_file_list += sub_file_list
    ann_file_list.sort()
    pdb.set_trace()
    out_path = os.path.join(opt['tube_folder_new_path'])
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    for ann_idx, ann_file in enumerate(ann_file_list):

        out_fn_path = os.path.join(out_path, os.path.basename(ann_file.replace('json', 'pk')))
        if os.path.isfile(out_fn_path):
            continue 
        if ann_idx <10000:
            continue 
        ann_dict  = jsonload(ann_file)
        prp_file_path = os.path.join(opt['prp_path'], 'proposal_'+str(ann_dict['scene_index']).zfill(5)+'.json')
        prp_dict =  jsonload(prp_file_path)
        
        tube_list = parse_tube_gt(ann_dict, prp_dict, opt)
        #visual_tube_proposals([tube_list], prp_dict, len(tube_list), opt)

        out_dict = {'tubes': tube_list}
        pickledump(out_fn_path, out_dict)
        if ann_idx%100==0:
            print('finish processing %d/%d videos' %(ann_idx, len(ann_file_list)))
        #visual_tube_proposals([tube_list, score_list], f_dict, max_obj_num, opt)

def processing_tube_list(opt):
    tube_prp_path = os.path.join(opt['tube_folder_path'] , str(opt['connect_w'])+'_'+str(opt['score_w']))
    pk_fn_list = get_sub_file_list(tube_prp_path, 'pk')
    pk_fn_list.sort()
    minimum_len = 10
    for f_id, prp_fn in enumerate(pk_fn_list):
        prp_tube_dict = pickleload(prp_fn)
        remove_tube_ids= []
        for prp_id, prp_info in enumerate(prp_tube_dict['tubes']):
            new_tube_list = []
            new_st_id_list = []
            new_flag=True 
            for box_id, box_info in enumerate(prp_info): 
                if box_info==[0, 0, 1, 1]:
                    if len(new_st_id_list)!=0 and len(new_st_id_list[-1])==1:
                        new_st_id_list[-1].append(box_id)
                    new_flag=True
                    continue
                if new_flag:
                    new_st_id_list.append([box_id])
                    new_flag=False
            if len(new_st_id_list[-1])==1:
                new_st_id_list[-1].append(len(prp_info))

            new_list_len = len(new_st_id_list)
            remove_ids = []
            if new_list_len>1:
                for new_id in range(new_list_len):
                    if new_st_id_list[new_id][1]-new_st_id_list[new_id][0]<minimum_len:
                        remove_ids.append(new_id)
                if (len(new_st_id_list) - len(remove_ids))>1:
                    remove_tube_ids.append(prp_id)
                    for new_id, tube_bound in enumerate(new_st_id_list):
                        if new_id in remove_ids:
                            continue 
                        st_id, end_id  = tube_bound
                        new_list = []
                        for frm_id, box_info in enumerate(prp_info):
                            if frm_id <st_id or frm_id>=end_id:
                                new_list.append([0, 0, 1, 1])
                            else:
                                new_list.append(box_info)
                        new_tube_list.append(new_list)

        new_prp_list= []
        for prp_idx, prp_info in enumerate(prp_tube_dict['tubes']):
            if prp_idx in remove_tube_ids:
                continue 
            new_prp_list.append(prp_info)
        new_prp_list+=new_tube_list

        if len(new_tube_list)>0:
            file_idx = prp_fn.split('_')[-1].split('.')[0]
            prp_file_path = os.path.join(opt['prp_path'], 'proposal_'+file_idx +'.json')
            f_dict =  jsonload(prp_file_path)
            visual_tube_proposals([new_prp_list], f_dict, len(new_prp_list), opt)
            #pdb.set_trace()


def compute_recall_and_precision(opt):
    
    iou_thre_list = [0.5, 0.6, 0.7, 0.8, 0.9]
    precision_list = [0 for i in range(len(iou_thre_list))]
    recall_list = [0 for i in range(len(iou_thre_list))]
    prp_num = 0
    gt_num = 0

    if opt['use_attr_flag']:
        tube_prp_path = os.path.join(opt['tube_folder_path'] , str(opt['connect_w'])+'_'+str(opt['score_w']) + '_'  +str(opt['attr_w']))
    else:
        tube_prp_path = os.path.join(opt['tube_folder_path'] , str(opt['connect_w'])+'_'+str(opt['score_w']))
    out_gt_path = os.path.join(opt['tube_folder_new_path'])

    pk_fn_list = get_sub_file_list(out_gt_path, 'pk')
    pk_fn_list.sort()

    for f_id, pk_fn in enumerate(pk_fn_list):
        gt_tube_dict = pickleload(pk_fn)
        id_str = pk_fn.split('_')[-1].split('.')[0]
        tube_prp_pk_fn = os.path.join(tube_prp_path, 'proposal_'+id_str+'.pk')
        if not os.path.isfile(tube_prp_pk_fn):
            continue 
        prp_tube_dict = pickleload(tube_prp_pk_fn)

        #iou_mat = np.zeros([len(prp_tube_dict['tubes']), len(gt_tube_dict['tubes'])])

        for prp_idx, prp in enumerate(prp_tube_dict['tubes']):
            for gt_idx, gt in enumerate(gt_tube_dict['tubes']):
                iou = compute_LS(prp, gt)
                #iou_mat[prp_idx, gt_idx] = iou

                for thre_idx, iou_thre in enumerate(iou_thre_list):
                    if iou>=iou_thre:
                        precision_list[thre_idx]+=1
                        recall_list[thre_idx]+=1

        prp_num +=len(prp_tube_dict['tubes'])
        gt_num +=len(gt_tube_dict['tubes'])

        if f_id % 10000==0 or f_id==(len(pk_fn_list)-1):
            print('processing %d/%d videos.\n' %(f_id, len(pk_fn_list)))
            for thre_idx, iou_thre in enumerate(iou_thre_list):
                print('precision@%3f is %3f\n' %(iou_thre, precision_list[thre_idx]*1.0/prp_num))
            for thre_idx, iou_thre in enumerate(iou_thre_list):
                print('recall@%3f is %3f\n' %(iou_thre, recall_list[thre_idx]*1.0/gt_num))
            print('\n')

def extract_tube_per_video_attribute(f_dict, opt, attr_dict_list):
    connect_w = opt['connect_w']
    score_w = opt['score_w']
    bbx_sc_list = []

    assert len(f_dict['frames'])==len(attr_dict_list)

    # get object number of a video
    max_obj_num = 0
    for frm_idx, frm_info in enumerate(f_dict['frames']):
        tmp_obj_num = len(frm_info['objects']) 
        if max_obj_num<tmp_obj_num:
            max_obj_num = tmp_obj_num 


    for frm_idx, frm_info in enumerate(f_dict['frames']):
        bbx_mat = []
        sc_mat = []
        color_list = []
        material_list = []
        shape_list = []
        attr_list = []
        tmp_obj_num = len(frm_info['objects']) 

        attr_frm_dict = attr_dict_list[frm_idx]
        assert len(frm_info['objects']) ==  len(attr_frm_dict['color'])
        for obj_idx, obj_info in enumerate(frm_info['objects']):
            bbx_xywh = mask.toBbox(obj_info['mask'])
            bbx_xyxy = copy.deepcopy(bbx_xywh)
            bbx_xyxy[2] =  bbx_xyxy[2] + bbx_xyxy[0]
            bbx_xyxy[3] =  bbx_xyxy[3] + bbx_xyxy[1]
            bbx_mat.append(bbx_xyxy)
            sc_mat.append(obj_info['score']*score_w)
            
            if opt['use_attr_flag']:
                tmp_color = COLORS[attr_frm_dict['color'][obj_idx]]
                tmp_material = MATERIALS[attr_frm_dict['material'][obj_idx]]
                tmp_shape = SHAPES[attr_frm_dict['shape'][obj_idx]]
                attr_list.append([tmp_color, tmp_material, tmp_shape])
            #pdb.set_trace()


        frm_size = frm_info['objects'][0]['mask']['size']
        for tmp_idx in range(tmp_obj_num, max_obj_num):
            tmp_box = np.array([0, 0, 1, 1])
            bbx_mat.append(tmp_box)
            sc_mat.append(-100)
            
            if opt['use_attr_flag']:
                attr_list.append(['', '', ''])

        bbx_mat = np.stack(bbx_mat, axis=0)
        sc_mat = np.array(sc_mat)
        sc_mat = np.expand_dims(sc_mat, axis=1 )

        #pdb.set_trace()

        if not opt['use_attr_flag']:
            bbx_sc_list.append([sc_mat, bbx_mat])
        else:
            bbx_sc_list.append([sc_mat, bbx_mat, attr_list])

    tube_list, score_list = get_tubes(bbx_sc_list, connect_w, opt['use_attr_flag'], opt['attr_w'])
    return tube_list, score_list, bbx_sc_list  



if __name__=='__main__':
    parms, opt = parse_opt()
    extract_tube_v0(opt)
    compute_recall_and_precision(opt)
    #evaluate_tube_performance(opt)
