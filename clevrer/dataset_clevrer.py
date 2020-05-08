from torch.utils.data import Dataset, DataLoader
import pdb
import os
import sys
from .utils import jsonload, pickleload, pickledump, transform_conpcet_forms_for_nscl, set_debugger, decode_mask_to_xyxy, transform_conpcet_forms_for_nscl_v2       
import argparse 
from PIL import Image
import copy
import numpy as np
import torch
from jacinle.utils.tqdm import tqdm
from nscl.datasets.definition import gdef
from nscl.datasets.common.vocab import Vocab
import operator
import math
import random
#import cv2
#torch.multiprocessing.set_sharing_strategy('file_system')
#set_debugger()
#_ignore_list = ['get_counterfact', 'unseen_events', 'filter_ancestor', 'filter_in', 'filter_out', 'filter_order', 'start', 'filter_moving', 'filter_stationary', 'filter_order', 'end']
#_ignore_list = ['get_counterfact', 'unseen_events', 'filter_ancestor', 'filter_order']
#_ignore_list = ['get_counterfact', 'unseen_events', 'filter_ancestor']
_ignore_list = []
#_ignore_list = ['get_counterfact']
#_used_list = ['filter_order']

def merge_img_patch(img_0, img_1):

    ret = img_0.copy()
    idx = img_1[:, :, 0] > 0
    idx = np.logical_or(idx, img_1[:, :, 1] > 0)
    idx = np.logical_or(idx, img_1[:, :, 2] > 0)
    ret[idx] = img_1[idx]

    return ret

def gen_vocab(dataset):
    all_words = dataset.parse_concepts_and_attributes()

    import jaclearn.embedding.constant as const
    vocab = Vocab()
    vocab.add(const.EBD_ALL_ZEROS)
    for w in sorted(all_words):
        vocab.add(w)
    for w in [const.EBD_UNKNOWN, const.EBD_BOS, const.EBD_EOS]:
        vocab.add(w)

    return vocab

def mapping_detected_tubes_to_objects(tube_key_dict, gt_obj_list):
    prp_id_to_gt_id = {}
    gt_id_to_prp_id = {} 
    for gt_obj in gt_obj_list:
        max_score = -1
        match_obj_id = -1
        for obj_id, obj_prp in tube_key_dict.items():
            match_score = 0
            for attr, concept in obj_prp.items():
                if concept == gt_obj[attr]:
                    match_score +=1
            if match_score >max_score:
                match_obj_id =  obj_id #gt_obj['object_id']
                max_score = match_score 
        #prp_id_to_gt_id[obj_id] = match_obj_id  
        gt_id_to_prp_id[gt_obj['object_id']] = match_obj_id 

    for gt_id, prp_id in gt_id_to_prp_id.items():
        prp_id_to_gt_id[prp_id] = gt_id 

    return  prp_id_to_gt_id, gt_id_to_prp_id  

def parse_static_attributes_for_tubes(tube_info, mask_gt, ratio):
    tube_to_attribute ={}
    for t_id, t_info in tube_info.items():
        if not isinstance(t_id, int):
            continue 
        attr_dict = gdef.all_concepts_clevrer['attribute'] 
        attr_dict_dict = {}
        tube_to_attribute[t_id] ={}
        for  attr_name, concept_list in attr_dict.items():
            attr_dict_dict[attr_name] = {} 
            for concept in concept_list:
                attr_dict_dict[attr_name][concept] = 0

        for f_idx, frm_id in enumerate(tube_info[t_id]['frm_name']):
            frm_mask_info = mask_gt['frames'][frm_id]

            for obj_info in frm_mask_info['objects']:
                obj_bbx = decode_mask_to_xyxy(obj_info['mask'])
                obj_bbx = torch.tensor(obj_bbx).float()
                if torch.abs(obj_bbx*ratio-tube_info[t_id]['boxes'][f_idx]).sum()<1:
                    for attr_name in attr_dict.keys():
                        concept_name = obj_info[attr_name]
                        attr_dict_dict[attr_name][concept_name] +=1
                    break 
        for attr_name, concept_dict in attr_dict_dict.items():
            max_concept = max(concept_dict.items(), key=operator.itemgetter(1))[0]
            tube_to_attribute[t_id][attr_name] = max_concept 

    return tube_to_attribute  


class clevrerDataset(Dataset):
    def __init__(self, args, phase, img_transform=None):
        self.args = args
        self.phase = phase
        self.img_transform = img_transform  
        question_ann_full_path = os.path.join(args.question_path, phase+'.json') 
        self.question_ann = jsonload(question_ann_full_path)
        self.vocab = gen_vocab(self)
        self.W = 480; self.H = 320
        self._filter_program_types()
        if self.args.extract_region_attr_flag:
            self.__intialize_frm_ann()
        self.background = None

    def merge_frames_for_prediction(self, img_list, obj_list):
        if self.background is None:
            self.background = Image.open(self.args.background_path).convert('RGB')
            self.background = self.background.resize((self.args.bgW, self.args.bgH), Image.ANTIALIAS)
            self.background = np.array(self.background)
        img_patch = copy.deepcopy(self.background)
        W, H = self.args.bgW, self.args.bgH
        for p_id, patch in enumerate(img_list):
            obj = obj_list[p_id]
            if math.isnan(obj['x']):
                continue
            if math.isnan(obj['y']):
                continue 
            y = int(obj['y'] * H )
            x = int(obj['x'] * W )

            img = np.array(patch)
            # print(x, y, H, W)
            h, w = img.shape[0], img.shape[1]
            x_ = max(-x, 0)
            y_ = max(-y, 0)
            x = max(x, 0)
            y = max(y, 0)
            h_ = min(h - y_, H - y)
            w_ = min(w - x_, W - x)

            if y + h_ < 0 or y >= H or x + w_ < 0 or x >= W:
                continue

            img_patch[y:y+h_, x:x+w_] = merge_img_patch(
                img_patch[y:y+h_, x:x+w_], img[y_:y_+h_, x_:x_+w_])
        img_patch = Image.fromarray(img_patch)
        return img_patch

    def load_counterfacts_info(self, scene_index, frm_dict, padding_img=None):
        predictions = {}
        full_pred_path = os.path.join(self.args.unseen_events_path, 'sim_'+str(scene_index).zfill(5)+'.json')
        pred_ann = jsonload(full_pred_path)
        # load prediction for future
        future_frm_list = []
        tube_box_dict = {}
        obj_num = len(frm_dict) - 2
        tmp_dict = {'boxes': [], 'frm_name': []}
        frm_list_unique = []
        tube_box_list = []
        for obj_id in range(obj_num):
            tube_box_dict[obj_id] = copy.deepcopy(tmp_dict)
            tube_box_list.append([])

        tube_box_list_list = [ copy.deepcopy(tube_box_list) for obj_id in range(obj_num)] 
        tube_box_dict_list = [ copy.deepcopy(tube_box_dict) for obj_id in range(obj_num)] 
        frm_list_unique_list = [ copy.deepcopy(frm_list_unique) for obj_id in range(obj_num)] 
        future_frm_list_list = [ copy.deepcopy(future_frm_list) for obj_id in range(obj_num)] 

        for pred_id, pred_info in enumerate(pred_ann['predictions']):
            what_if_flag = pred_info['what_if']
            if what_if_flag ==-1:
                continue
            for traj_id, traj_info in enumerate(pred_info['trajectory']):
                frame_index = traj_info['frame_index']
                # TODO may have bug if events happens in the prediction frame
                if self.args.n_seen_frames < frame_index:
                    continue
                # preparing rgb features
                img_list = traj_info['imgs']
                obj_list = traj_info['objects']
                syn_img = self.merge_frames_for_prediction(img_list, obj_list)
                #cv2.imwrite('img_%d.png'%(frame_index) , syn_img)
                _exist_obj_flag = False 
                for r_id, obj_id in enumerate(traj_info['ids']):
                    
                    obj = traj_info['objects'][r_id]
                    if math.isnan(obj['x']):
                        continue
                    if math.isnan(obj['y']):
                        continue 
                    if math.isnan(obj['h']):
                        continue
                    if math.isnan(obj['w']):
                        continue 
                    if obj['x']<0 or obj['x']>1:
                        continue 
                    if obj['y']<0 or obj['y']>1:
                        continue 
                    if obj['h']<0 or obj['h']>1:
                        continue 
                    if obj['w']<0 or obj['w']>1:
                        continue 

                    _exist_obj_flag = True

                    x = copy.deepcopy(obj['x'])
                    y = copy.deepcopy(obj['y'])
                    h = copy.deepcopy(obj['h'])
                    w = copy.deepcopy(obj['w'])
                    x2 = x + w
                    y2 = y + h
                    tube_box_dict_list[what_if_flag][obj_id]['boxes'].append(np.array([x, y, x2, y2]).astype(np.float32))
                    tube_box_dict_list[what_if_flag][obj_id]['frm_name'].append(frame_index)
            
                if not _exist_obj_flag:
                    continue

                frm_list_unique_list[what_if_flag].append(frame_index)
                syn_img2, _ = self.img_transform(syn_img, np.array([0, 0, 1, 1]))
                future_frm_list_list[what_if_flag].append(syn_img2)
                for obj_id in range(obj_num):
                    if obj_id in traj_info['ids']:
                        index = traj_info['ids'].index(obj_id)
                        obj = traj_info['objects'][index]
                        
                        if math.isnan(obj['x']) or math.isnan(obj['y']) or \
                                math.isnan(obj['h']) or math.isnan(obj['w']) or\
                                obj['x']<0 or obj['x']>1 or \
                                obj['y']<0 or obj['y']>1 or \
                                obj['h']<0 or obj['h']>1 or \
                                obj['w']<0 or obj['w']>1:

                            tube_box_list_list[what_if_flag][obj_id].append(np.array([-1.0, -1.0, 0.0, 0.0]).astype(np.float32))
                            continue 

                        x = copy.deepcopy(obj['x'])
                        y = copy.deepcopy(obj['y'])
                        h = copy.deepcopy(obj['h'])
                        w = copy.deepcopy(obj['w'])
                        x +=  w*0.5
                        y +=  h*0.5
                        tube_box_list_list[what_if_flag][obj_id].append(np.array([x, y, w, h]).astype(np.float32))
                    else:
                        tube_box_list_list[what_if_flag][obj_id].append(np.array([-1.0, -1.0, 0.0, 0.0]).astype(np.float32))
        img_tensor_list = []
        for what_if_id in range(obj_num):
            future_frm_list = future_frm_list_list[what_if_id]
            if len(future_frm_list)==0:
                frm_list_unique = [frm_dict['frm_list'][-1]] 
                tube_box_dict_list[what_if_id]['frm_list'] = frm_list_unique 
                last_tube_box_list = [ [tmp_list[-1]] for tmp_list in frm_dict['box_seq']['tubes'] ] 
                tube_box_dict_list[what_if_id]['box_seq'] = last_tube_box_list
                img_tensor = padding_img.unsqueeze(0)
                for obj_id, obj_info in frm_dict.items():
                    if not isinstance(obj_id, int):
                        continue
                    tube_box_dict_list[what_if_id][obj_id]={}
                    tube_box_dict_list[what_if_id][obj_id]['boxes'] = [obj_info['boxes'][-1]]
                    tube_box_dict_list[what_if_id][obj_id]['frm_name'] = [obj_info['frm_name'][-1]]
                img_tensor_list.append(img_tensor)
            else:
                frm_list_unique = frm_list_unique_list[what_if_id] 
                tube_box_dict_list[what_if_id]['frm_list'] = frm_list_unique_list[what_if_id] 
                tube_box_dict_list[what_if_id]['box_seq'] = tube_box_list_list[what_if_id]
                img_tensor = torch.stack(future_frm_list, 0)
                img_tensor_list.append(img_tensor)
        return tube_box_dict_list, img_tensor_list  

    def load_predict_info(self, scene_index, frm_dict, padding_img=None):
        predictions = {}
        full_pred_path = os.path.join(self.args.unseen_events_path, 'sim_'+str(scene_index).zfill(5)+'.json')
        pred_ann = jsonload(full_pred_path)
        # load prediction for future
        future_frm_list = []
        tube_box_dict = {}
        obj_num = len(frm_dict) - 2
        tmp_dict = {'boxes': [], 'frm_name': []}
        frm_list_unique = []
        tube_box_list = []
        for obj_id in range(obj_num):
            tube_box_dict[obj_id] = copy.deepcopy(tmp_dict)
            tube_box_list.append([])

        for pred_id, pred_info in enumerate(pred_ann['predictions']):
            what_if_flag = pred_info['what_if']
            if what_if_flag !=-1:
                continue 
            for traj_id, traj_info in enumerate(pred_info['trajectory']):
                frame_index = traj_info['frame_index']
                if self.args.n_seen_frames > frame_index:
                    continue
                # preparing rgb features
                img_list = traj_info['imgs']
                obj_list = traj_info['objects']
                syn_img = self.merge_frames_for_prediction(img_list, obj_list)
                #cv2.imwrite('img_%d.png'%(frame_index) , syn_img)
                _exist_obj_flag = False 
                for r_id, obj_id in enumerate(traj_info['ids']):
                    
                    obj = traj_info['objects'][r_id]
                    if math.isnan(obj['x']):
                        continue
                    if math.isnan(obj['y']):
                        continue 
                    if math.isnan(obj['h']):
                        continue
                    if math.isnan(obj['w']):
                        continue 
                    if obj['x']<0 or obj['x']>1:
                        continue 
                    if obj['y']<0 or obj['y']>1:
                        continue 
                    if obj['h']<0 or obj['h']>1:
                        continue 
                    if obj['w']<0 or obj['w']>1:
                        continue 

                    _exist_obj_flag = True

                    x = copy.deepcopy(obj['x'])
                    y = copy.deepcopy(obj['y'])
                    h = copy.deepcopy(obj['h'])
                    w = copy.deepcopy(obj['w'])
                    x2 = x + w
                    y2 = y + h
                    tube_box_dict[obj_id]['boxes'].append(np.array([x, y, x2, y2]).astype(np.float32))
                    tube_box_dict[obj_id]['frm_name'].append(frame_index)
            
                if not _exist_obj_flag:
                    continue

                frm_list_unique.append(frame_index)
                syn_img2, _ = self.img_transform(syn_img, np.array([0, 0, 1, 1]))
                future_frm_list.append(syn_img2)
                for obj_id in range(obj_num):
                    if obj_id in traj_info['ids']:
                        index = traj_info['ids'].index(obj_id)
                        obj = traj_info['objects'][index]
                        
                        if math.isnan(obj['x']) or math.isnan(obj['y']) or \
                                math.isnan(obj['h']) or math.isnan(obj['w']) or\
                                obj['x']<0 or obj['x']>1 or \
                                obj['y']<0 or obj['y']>1 or \
                                obj['h']<0 or obj['h']>1 or \
                                obj['w']<0 or obj['w']>1:

                            tube_box_list[obj_id].append(np.array([-1.0, -1.0, 0.0, 0.0]).astype(np.float32))
                            continue 

                        x = copy.deepcopy(obj['x'])
                        y = copy.deepcopy(obj['y'])
                        h = copy.deepcopy(obj['h'])
                        w = copy.deepcopy(obj['w'])
                        x +=  w*0.5
                        y +=  h*0.5
                        tube_box_list[obj_id].append(np.array([x, y, w, h]).astype(np.float32))
                    else:
                        tube_box_list[obj_id].append(np.array([-1.0, -1.0, 0.0, 0.0]).astype(np.float32))

        if len(future_frm_list)==0:
            frm_list_unique = [frm_dict['frm_list'][-1]] 
            tube_box_dict['frm_list'] = frm_list_unique 
            last_tube_box_list = [ [tmp_list[-1]] for tmp_list in frm_dict['box_seq']['tubes'] ] 
            tube_box_dict['box_seq'] = last_tube_box_list
            img_tensor = padding_img.unsqueeze(0)
            for obj_id, obj_info in frm_dict.items():
                if not isinstance(obj_id, int):
                    continue
                tube_box_dict[obj_id]={}
                tube_box_dict[obj_id]['boxes'] = [obj_info['boxes'][-1]]
                tube_box_dict[obj_id]['frm_name'] = [obj_info['frm_name'][-1]]
        else:
            frm_list_unique = frm_list_unique 
            tube_box_dict['frm_list'] = frm_list_unique 
            tube_box_dict['box_seq'] = tube_box_list
            img_tensor = torch.stack(future_frm_list, 0)
        return tube_box_dict, img_tensor 

    def __intialize_frm_ann(self):
        frm_ann = []
        for index, meta_ann in enumerate(self.question_ann):
            scene_idx = meta_ann['scene_index']
            if scene_idx<self.args.start_index:
                continue
            mask_gt_path = os.path.join(self.args.mask_gt_path, 'proposal_'+str(scene_idx).zfill(5)+'.json') 
            mask_gt = jsonload(mask_gt_path)
            for frm_id in range(len(mask_gt['frames'])):
                frm_ann.append([index, frm_id])
        self.frm_ann = frm_ann 
        #pdb.set_trace()

    def __get_video_frame__(self, frm_index):
        data = {}
        index = self.frm_ann[frm_index][0]
        frm_id = self.frm_ann[frm_index][1]
        meta_ann = self.question_ann[index]
        scene_idx = meta_ann['scene_index']
        data['frm_id'] = frm_id 
        sub_idx = int(scene_idx/1000)
        sub_img_folder = 'image_'+str(sub_idx).zfill(2)+'000-'+str(sub_idx+1).zfill(2)+'000'
        img_full_folder = os.path.join(self.args.frm_img_path, sub_img_folder) 
       
        # getting image frames 
        mask_gt_path = os.path.join(self.args.mask_gt_path, 'proposal_'+str(scene_idx).zfill(5)+'.json') 
        mask_gt = jsonload(mask_gt_path)
        tube_info = self.sample_tube_frames(scene_idx)
        frm_dict = self.sample_frames_v3(mask_gt, tube_info, frm_id)   
        H=0; W=0
        img_list = []
        img_full_path = os.path.join(img_full_folder, 'video_'+str(scene_idx).zfill(5), str(frm_id+1)+'.png')
        img = Image.open(img_full_path).convert('RGB')
        W, H = img.size
        img, _ = self.img_transform(img, np.array([0, 0, 1, 1]))
        img_list.append(img)
        img_tensor = torch.stack(img_list, 0)
        data['img'] = img_tensor 
        
        # resize frame boxes
        img_size = self.args.img_size
        ratio = img_size / min(H, W)
        for key_id, tube_box_info in frm_dict.items():
            if not isinstance(key_id, int):
                continue 
            for box_id, box in enumerate(tube_box_info['boxes']):
                tmp_box = torch.tensor(box).float()*ratio
                tube_box_info['boxes'][box_id] = tmp_box
            frm_dict[key_id]=tube_box_info 
        data['tube_info'] = frm_dict  
        data['meta_ann'] = meta_ann 
       
        # adding scene supervision
        if self.args.scene_supervision_flag:
            sub_idx = int(scene_idx/1000)
            sub_ann_folder = 'annotation_'+str(sub_idx).zfill(2)+'000-'+str(sub_idx+1).zfill(2)+'000'
            ann_full_folder = os.path.join(self.args.scene_gt_path, sub_ann_folder) 
            scene_gt_path = os.path.join(ann_full_folder, 'annotation_'+str(scene_idx).zfill(5)+'.json') 
            scene_gt = jsonload(scene_gt_path)
        
            gt_id_to_tube_id = {}
            for attr, concept_group in gdef.all_concepts_clevrer['attribute'].items():
                attr_list = []
                for obj_id, obj_info in enumerate(mask_gt['frames'][frm_id]['objects']):
                    obj_bbx = decode_mask_to_xyxy(obj_info['mask'])
                    obj_bbx = torch.tensor(obj_bbx).float()
                    tube_num = len(data['tube_info']) -2 
                    # get mapping id
                    for t_id in range(tube_num):
                        if len(data['tube_info'][t_id]['boxes'])==1:
                            tube_box = data['tube_info'][t_id]['boxes'][0]
                        else:
                            continue 
                        if torch.abs(obj_bbx*ratio-tube_box).sum()<1:
                            gt_id_to_tube_id[obj_id] = t_id
                            break 
                    # get attribute
                    concept_index = concept_group.index(obj_info[attr])
                    attr_list.append(concept_index)
                attr_key = 'attribute_' + attr
                data[attr_key] = torch.tensor(attr_list)
            data['prp_id_to_t_id'] = gt_id_to_tube_id  
        return data 

    def parse_concepts_and_attributes(self):
        word_list = []
        for questions_info in self.question_ann:
            for ques in questions_info['questions']:
                tmp_word_list = ques['question'].replace('?', '').replace(',', '').replace('.','').replace('\'s', '').split(' ')
                if 'answer' in ques.keys():
                    tmp_word_list +=ques['answer'].replace('.', '').replace(',', '').replace('\'s','').split(' ')
        word_list_unique = set(word_list)
        return word_list_unique 

    def parse_program_dict(self):
        prog_list = []
        for questions_info in self.question_ann:
            for ques in questions_info['questions']:
                prog_list +=ques['program']
        prog_list_unique = list(set(prog_list))
        
        COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'yellow', 'cyan', 'purple']
        MATERIALS = ['metal', 'rubber']
        SHAPES = ['sphere', 'cylinder', 'cube']
        ORDER  = ['first', 'second', 'last']
        OTHERS = ['all_events']
        ALL_CONCEPTS= COLORS + MATERIALS + SHAPES +ORDER  

        tar_list = []
        for op in prog_list_unique:
            if op in ALL_CONCEPTS:
                continue
            if op.startswith('filter'):
                continue
            #if op.startswith('query'):
            #    continue
            tar_list.append(op)
        pdb.set_trace()

    def _filter_program_types(self):
        new_question_ann = []
        ori_ques_num = 0
        filt_ques_num = 0
        #pdb.set_trace()
        for idx, meta_ann in enumerate(self.question_ann):
            meta_new = copy.deepcopy(meta_ann)
            meta_new['questions'] = []
            for ques_info in meta_ann['questions']:
                valid_flag = True
                for pg in ques_info['program']:
                    if pg in _ignore_list:
                        valid_flag = False
                        break
                    
                if not valid_flag:
                    continue
                #if 'answer' not in ques_info.keys() and ques_info['question_type']!='explanatory':
                #if ('answer' not in ques_info.keys() and ques_info['question_type']!='explanatory' and ques_info['question_type']!='predictive' ):
                #    continue 
                meta_new['questions'].append(ques_info)
            if len(meta_new['questions'])>0:
                new_question_ann.append(meta_new)
            filt_ques_num  +=len(meta_new['questions'])
            ori_ques_num +=len(meta_ann['questions'])
        print('Videos: oriinal: %d, target: %d\n'%(len(self.question_ann), len(new_question_ann)))
        print('Questions: oriinal: %d, target: %d\n'%(ori_ques_num, filt_ques_num))
        self.question_ann = new_question_ann 

    def __getitem__(self, index):
        if self.args.extract_region_attr_flag:
            return self.__get_video_frame__(index)
        else:
            if self.args.version == 'v2' or self.args.version == 'v3':
                return self.__getitem__model_v2(index)
            else:
                return self.__getitem__model(index)

    def __getitem__model_v2(self, index):
        data = {}
        meta_ann = copy.deepcopy(self.question_ann[index])
        scene_idx = meta_ann['scene_index']
        sub_idx = int(scene_idx/1000)
        sub_img_folder = 'image_'+str(sub_idx).zfill(2)+'000-'+str(sub_idx+1).zfill(2)+'000'
        img_full_folder = os.path.join(self.args.frm_img_path, sub_img_folder) 
       
       # getting image frames 
        tube_info = self.sample_tube_frames(scene_idx)
        if self.args.even_smp_flag:
            frm_dict, valid_flag_dict = self.sample_frames_v2(tube_info, self.args.frm_img_num)   
        else:
            frm_dict = self.sample_frames(tube_info, self.args.frm_img_num)   
        frm_list = frm_dict['frm_list']
        H=0; W=0
        img_list = []
        for i, frm in enumerate(frm_list): 
            img_full_path = os.path.join(img_full_folder, 'video_'+str(scene_idx).zfill(5), str(frm+1)+'.png')
            img = Image.open(img_full_path).convert('RGB')
            W, H = img.size
            img, _ = self.img_transform(img, np.array([0, 0, 1, 1]))
            img_list.append(img)
        img_tensor = torch.stack(img_list, 0)
        data['img'] = img_tensor 
        data['valid_seq_mask'] = valid_flag_dict  
        # resize frame boxes
        img_size = self.args.img_size
        ratio = img_size / min(H, W)
        for key_id, tube_box_info in frm_dict.items():
            if not isinstance(key_id, int):
                continue 
            for box_id, box in enumerate(tube_box_info['boxes']):
                tmp_box = torch.tensor(box)*ratio
                tube_box_info['boxes'][box_id] = tmp_box
            frm_dict[key_id]=tube_box_info 
        data['tube_info'] = frm_dict  
    
        load_predict_flag = False
        load_counter_fact_flag = False
        counterfact_list = [q_id for q_id, ques_info in enumerate(meta_ann['questions']) if ques_info['question_type']=='counterfactual']
        sample_counterfact_list = random.sample(counterfact_list, self.args.max_counterfact_num) if self.phase=='train' and len(counterfact_list)>=self.args.max_counterfact_num else  counterfact_list 
        # getting programs
        for q_id, ques_info in enumerate(meta_ann['questions']):
            valid_flag = True
            for pg in ques_info['program']:
                if pg in _ignore_list:
                    valid_flag = False
                    break
            if not valid_flag:
                continue
            #if 'answer' not in ques_info.keys():
            #if 'answer' not in ques_info.keys() and ques_info['question_type']!='explanatory':
            #if ('answer' not in ques_info.keys() and ques_info['question_type']!='explanatory' and ques_info['question_type']!='predictive' ):
            #    continue
                        
            if ques_info['question_type']=='predictive':
                load_predict_flag = True
            if ques_info['question_type']=='counterfactual':
                if q_id in sample_counterfact_list:
                    load_counter_fact_flag = True
                else:
                    continue

            if 'answer'in ques_info.keys() and ques_info['answer'] == 'no':
                ques_info['answer'] = False
            elif 'answer' in ques_info.keys() and ques_info['answer'] == 'yes':
                ques_info['answer'] = True

            program_cl = transform_conpcet_forms_for_nscl_v2(ques_info['program'])
            meta_ann['questions'][q_id]['program_cl'] = program_cl 
            if 'answer'in ques_info.keys():
                meta_ann['questions'][q_id]['answer'] = ques_info['answer']
            else:
                for choice_id, choice_info in enumerate(meta_ann['questions'][q_id]['choices']):
                    meta_ann['questions'][q_id]['choices'][choice_id]['program_cl'] = \
                        transform_conpcet_forms_for_nscl_v2(choice_info['program'])

        q_num_ori = len(meta_ann['questions']) 
        for q_id in sorted(counterfact_list, reverse=True):
            if q_id in sample_counterfact_list:
                continue 
            del meta_ann['questions'][q_id]
        data['meta_ann'] = meta_ann 
        #pdb.set_trace()
        # loadding unseen events
        if load_predict_flag  and self.args.version=='v2':
            scene_index = meta_ann['scene_index']
            data['predictions'], data['img_future'] = self.load_predict_info(scene_index, frm_dict, padding_img= data['img'][-1])
            _, c, tarH, tarW = img_tensor.shape
            for key_id, tube_box_info in data['predictions'].items():
                if not isinstance(key_id, int):
                    continue
                for box_id, box in enumerate(tube_box_info['boxes']):
                    tmp_box = torch.tensor(box).float()
                    tmp_box[0] = tmp_box[0]*tarW
                    tmp_box[2] = tmp_box[2]*tarW
                    tmp_box[1] = tmp_box[1]*tarH 
                    tmp_box[3] = tmp_box[3]*tarH
                    data['predictions'][key_id]['boxes'][box_id] = tmp_box
        else:
            # just padding for the dataloader
            data['predictions'] = {}
            data['img_future'] = torch.zeros(1, 1, 1, 1)
        data['load_predict_flag'] =  load_predict_flag 


        # loadding counterfact events
        if load_counter_fact_flag and self.args.version=='v2':
            scene_index = meta_ann['scene_index']
            data['counterfacts'], data['img_counterfacts'] = self.load_counterfacts_info(scene_index, frm_dict, padding_img=data['img'][0])
        else:
            # just padding for the dataloader
            data['counterfacts'] = {}
            data['img_counterfacts'] = torch.zeros(1, 1, 1, 1)


        # adding scene supervision
        if self.args.scene_supervision_flag:
            mask_gt_path = os.path.join(self.args.mask_gt_path, 'proposal_'+str(scene_idx).zfill(5)+'.json') 
            sub_idx = int(scene_idx/1000)
            sub_ann_folder = 'annotation_'+str(sub_idx).zfill(2)+'000-'+str(sub_idx+1).zfill(2)+'000'
            ann_full_folder = os.path.join(self.args.scene_gt_path, sub_ann_folder) 
            scene_gt_path = os.path.join(ann_full_folder, 'annotation_'+str(scene_idx).zfill(5)+'.json') 
            scene_gt = jsonload(scene_gt_path)
            mask_gt = jsonload(mask_gt_path)
            tube_key_dict = parse_static_attributes_for_tubes(data['tube_info'], mask_gt, ratio)
            # TODO: this may raise bug since it hack the data property for gt
            prp_id_to_gt_id, gt_id_to_prp_id = mapping_detected_tubes_to_objects(tube_key_dict, scene_gt['object_property'])
            #pdb.set_trace()
            for attri_group, attribute in gdef.all_concepts_clevrer.items():
                if attri_group=='attribute':
                    for attr, concept_group in attribute.items(): 
                        attr_list = []
                        obj_num = len(data['tube_info']) -2 
                        for t_id in range(obj_num):
                            concept_index = concept_group.index(tube_key_dict[t_id][attr])
                            attr_list.append(concept_index)
                        attr_key = attri_group + '_' + attr 
                        data[attr_key] = torch.tensor(attr_list)
                elif attri_group=='relation':
                    for attr, concept_group in attribute.items(): 
                        if attr=='event1':
                            obj_num = len(data['tube_info']) -2 
                            rela_coll = torch.zeros(obj_num, obj_num)

                            for event_id, event in enumerate(scene_gt['collision']):
                                obj_id_pair = event['object_ids']
                                gt_id1 = obj_id_pair[0]; gt_id2 = obj_id_pair[1]
                                prp_id1 = gt_id_to_prp_id[gt_id1]
                                prp_id2 = gt_id_to_prp_id[gt_id2]
                                rela_coll[prp_id1, prp_id2] = 1
                                rela_coll[prp_id2, prp_id1] = 1
                            attr_key = attri_group + '_' + 'collision'
                            data[attr_key] = rela_coll
                elif attri_group=='temporal':
                    for attr, concept_group in attribute.items(): 
                        if attr=='event2':
                            obj_num = len(data['tube_info']) -2 
                            attr_frm_id_st = []
                            attr_frm_id_ed = []
                            min_frm = 2
                            box_thre = 0.0001

                            for t_id in range(obj_num):
                                box_seq = data['tube_info']['box_seq']['tubes'][t_id]
                                box_seq_np = np.stack(box_seq, axis=0)
                                tar_area = box_seq_np[:, 2] * box_seq_np[:, 3]
                                
                                time_step = len(tar_area)
                                # filter_in 
                                for t_id in range(time_step):
                                    end_id = min(t_id + min_frm, time_step-1)
                                    if np.sum(tar_area[t_id:end_id]>box_thre)>=(end_id-t_id):
                                        attr_frm_id_st.append(t_id)
                                        #pdb.set_trace()
                                        break 
                                    if t_id == time_step - 1:
                                        attr_frm_id_st.append(0)
                                # filter out
                                for t_id in range(time_step, -1, -1):
                                    st_id = max(t_id - min_frm, 0)
                                    if np.sum(tar_area[st_id:t_id]>box_thre)>=(t_id-st_id):
                                        attr_frm_id_ed.append(t_id)
                                        break 
                                    if t_id == 0:
                                        attr_frm_id_ed.append(time_step-1)

                            attr_key = attri_group + '_in'
                            data[attr_key] = torch.tensor(attr_frm_id_st)
                            attr_key = attri_group + '_out'
                            data[attr_key] = torch.tensor(attr_frm_id_ed)
                            #pdb.set_trace()
        return data 


    def __getitem__model(self, index):
        #pdb.set_trace()
        data = {}
        meta_ann = self.question_ann[index]
        scene_idx = meta_ann['scene_index']
        sub_idx = int(scene_idx/1000)
        sub_img_folder = 'image_'+str(sub_idx).zfill(2)+'000-'+str(sub_idx+1).zfill(2)+'000'
        img_full_folder = os.path.join(self.args.frm_img_path, sub_img_folder) 
       
       # getting image frames 
        tube_info = self.sample_tube_frames(scene_idx)
        if self.args.even_smp_flag:
            frm_dict = self.sample_frames_v2(tube_info, self.args.frm_img_num)   
        else:
            frm_dict = self.sample_frames(tube_info, self.args.frm_img_num)   
        frm_list = frm_dict['frm_list']
        H=0; W=0
        img_list = []
        for i, frm in enumerate(frm_list): 
            img_full_path = os.path.join(img_full_folder, 'video_'+str(scene_idx).zfill(5), str(frm+1)+'.png')
            img = Image.open(img_full_path).convert('RGB')
            W, H = img.size
            img, _ = self.img_transform(img, np.array([0, 0, 1, 1]))
            img_list.append(img)
        img_tensor = torch.stack(img_list, 0)
        data['img'] = img_tensor 
        
        # resize frame boxes
        img_size = self.args.img_size
        ratio = img_size / min(H, W)
        for key_id, tube_box_info in frm_dict.items():
            if not isinstance(key_id, int):
                continue 
            for box_id, box in enumerate(tube_box_info['boxes']):
                tmp_box = torch.tensor(box)*ratio
                tube_box_info['boxes'][box_id] = tmp_box
            frm_dict[key_id]=tube_box_info 
        data['tube_info'] = frm_dict  
        
        # getting programs
        for q_id, ques_info in enumerate(meta_ann['questions']):
            valid_flag = True
            for pg in ques_info['program']:
                if pg in _ignore_list:
                    valid_flag = False
                    break
            if not valid_flag:
                continue
            if 'answer' not in ques_info.keys():
                continue 
            if ques_info['answer'] == 'no':
                ques_info['answer'] = False
            elif ques_info['answer'] == 'yes':
                ques_info['answer'] = True

            program_cl = transform_conpcet_forms_for_nscl(ques_info['program'])
            meta_ann['questions'][q_id]['program_cl'] = program_cl 
            meta_ann['questions'][q_id]['answer'] = ques_info['answer']

        data['meta_ann'] = meta_ann 
       
        # adding scene supervision
        if self.args.scene_supervision_flag:
            mask_gt_path = os.path.join(self.args.mask_gt_path, 'proposal_'+str(scene_idx).zfill(5)+'.json') 
            sub_idx = int(scene_idx/1000)
            sub_ann_folder = 'annotation_'+str(sub_idx).zfill(2)+'000-'+str(sub_idx+1).zfill(2)+'000'
            ann_full_folder = os.path.join(self.args.scene_gt_path, sub_ann_folder) 
            scene_gt_path = os.path.join(ann_full_folder, 'annotation_'+str(scene_idx).zfill(5)+'.json') 
            scene_gt = jsonload(scene_gt_path)
            mask_gt = jsonload(mask_gt_path)
            tube_key_dict = parse_static_attributes_for_tubes(data['tube_info'], mask_gt, ratio)
            # TODO: this may raise bug since it hack the data property for gt
            prp_id_to_gt_id, gt_id_to_prp_id = mapping_detected_tubes_to_objects(tube_key_dict, scene_gt['object_property'])
            #pdb.set_trace()
            for attri_group, attribute in gdef.all_concepts_clevrer.items():
                if attri_group=='attribute':
                    for attr, concept_group in attribute.items(): 
                        attr_list = []
                        obj_num = len(data['tube_info']) -2 
                        for t_id in range(obj_num):
                            concept_index = concept_group.index(tube_key_dict[t_id][attr])
                            attr_list.append(concept_index)
                        attr_key = attri_group + '_' + attr 
                        data[attr_key] = torch.tensor(attr_list)
                elif attri_group=='relation':
                    for attr, concept_group in attribute.items(): 
                        if attr=='events':
                            obj_num = len(data['tube_info']) -2 
                            rela_coll = torch.zeros(obj_num, obj_num)

                            for event_id, event in enumerate(scene_gt['collision']):
                                obj_id_pair = event['object_ids']
                                gt_id1 = obj_id_pair[0]; gt_id2 = obj_id_pair[1]
                                prp_id1 = gt_id_to_prp_id[gt_id1]
                                prp_id2 = gt_id_to_prp_id[gt_id2]
                                rela_coll[prp_id1, prp_id2] = 1
                                rela_coll[prp_id2, prp_id1] = 1
                            attr_key = attri_group + '_' + 'collision'
                            data[attr_key] = rela_coll
                elif attri_group=='temporal':
                    for attr, concept_group in attribute.items(): 
                        if attr=='scene':
                            obj_num = len(data['tube_info']) -2 
                            attr_frm_id_st = []
                            attr_frm_id_ed = []
                            min_frm = 2
                            box_thre = 0.0001

                            for t_id in range(obj_num):
                                box_seq = data['tube_info']['box_seq']['tubes'][t_id]
                                box_seq_np = np.stack(box_seq, axis=0)
                                tar_area = box_seq_np[:, 2] * box_seq_np[:, 3]
                                
                                time_step = len(tar_area)
                                # filter_in 
                                for t_id in range(time_step):
                                    end_id = min(t_id + min_frm, time_step-1)
                                    if np.sum(tar_area[t_id:end_id]>box_thre)>=(end_id-t_id):
                                        attr_frm_id_st.append(t_id)
                                        #pdb.set_trace()
                                        break 
                                    if t_id == time_step - 1:
                                        attr_frm_id_st.append(0)
                                # filter out
                                for t_id in range(time_step, -1, -1):
                                    st_id = max(t_id - min_frm, 0)
                                    if np.sum(tar_area[st_id:t_id]>box_thre)>=(t_id-st_id):
                                        attr_frm_id_ed.append(t_id)
                                        break 
                                    if t_id == 0:
                                        attr_frm_id_ed.append(time_step-1)

                            attr_key = attri_group + '_in'
                            data[attr_key] = torch.tensor(attr_frm_id_st)
                            attr_key = attri_group + '_out'
                            data[attr_key] = torch.tensor(attr_frm_id_ed)
                            #pdb.set_trace()
        return data 

    def sample_frames(self, tube_info, img_num):
        tube_box_dict = {}
        frm_list = []
        for tube_id, tmp_tube in enumerate(tube_info['tubes']):
            tmp_dict = {}
            frm_num = len(tmp_tube) 
            tmp_list = []
            frm_ids = []
            count_idx = 0
            for frm_id in range(frm_num-1, -1 , -1):
                if tmp_tube[frm_id] == [0, 0, 1, 1]:
                    continue 
                tmp_list.append(copy.deepcopy(tmp_tube[frm_id]))
                frm_ids.append(frm_id)
                count_idx +=1
                if count_idx>=img_num:
                    break
            tmp_dict['boxes'] = tmp_list
            tmp_dict['frm_name'] = frm_ids  
            tube_box_dict[tube_id] = tmp_dict 
            frm_list +=frm_ids  
        frm_list_unique = list(set(frm_list))
        frm_list_unique.sort()
        tube_box_dict['frm_list'] = frm_list_unique  

        if self.args.normalized_boxes:
            
            for tube_id, tmp_tube in enumerate(tube_info['tubes']):
                tmp_dict = {}
                frm_num = len(tmp_tube) 
                for frm_id in range(frm_num):
                    tmp_box = tmp_tube[frm_id]
                    if tmp_box == [0, 0, 1, 1]:
                        if self.args.new_mask_out_value_flag:
                            tmp_box = [-1*self.W, -1*self.H, -1*self.W, -1*self.H]
                        else:
                            tmp_box = [0, 0, 0, 0]
                    x_c = (tmp_box[0] + tmp_box[2])* 0.5
                    y_c = (tmp_box[1] + tmp_box[3])* 0.5
                    w = tmp_box[2] - tmp_box[0]
                    h = tmp_box[3] - tmp_box[1]
                    tmp_array = np.array([x_c, y_c, w, h])
                    tmp_array[0] = tmp_array[0] / self.W
                    tmp_array[1] = tmp_array[1] / self.H
                    tmp_array[2] = tmp_array[2] / self.W
                    tmp_array[3] = tmp_array[3] / self.H
                    tube_info['tubes'][tube_id][frm_id] = tmp_array 
        tube_box_dict['box_seq'] = tube_info  

        return tube_box_dict 

    def sample_frames_v3(self, mask_gt, tube_info, frm_id):
        tube_box_dict = {}
        frm_num = len(tube_info['tubes'][0]) 
        tmp_tube = tube_info['tubes'][0]
        for obj_id, obj_info in enumerate(mask_gt['frames'][frm_id]['objects']):
            obj_bbx = decode_mask_to_xyxy(obj_info['mask'])
            #obj_bbx = torch.tensor(obj_bbx).float()
            tmp_dict = {}
            tmp_list = []
            count_idx = 0
            frm_ids = []
            frm_list = [frm_id]
            for frm_id in frm_list:
                tmp_list.append(obj_bbx)
                frm_ids.append(frm_id)

            tmp_dict['boxes'] = tmp_list
            tmp_dict['frm_name'] = frm_ids  
            tube_box_dict[obj_id] = tmp_dict 
        frm_list_unique = list(set(frm_list))
        frm_list_unique.sort()
        tube_box_dict['frm_list'] = frm_list_unique  

        if self.args.normalized_boxes:
            new_tube_info = {'tubes': []} 
            for tube_id, obj_info in enumerate(mask_gt['frames'][frm_id]['objects']):
                tmp_tube  = copy.deepcopy(tube_info['tubes'][0])
                tmp_dict = {}
                frm_num = len(tmp_tube)
                new_tube_info['tubes'].append([])
                for frm_id in range(frm_num):
                    tmp_box = [0, 0, 0, 0]
                    x_c = (tmp_box[0] + tmp_box[2])* 0.5
                    y_c = (tmp_box[1] + tmp_box[3])* 0.5
                    w = tmp_box[2] - tmp_box[0]
                    h = tmp_box[3] - tmp_box[1]
                    tmp_array = np.array([x_c, y_c, w, h])
                    tmp_array[0] = tmp_array[0] / self.W
                    tmp_array[1] = tmp_array[1] / self.H
                    tmp_array[2] = tmp_array[2] / self.W
                    tmp_array[3] = tmp_array[3] / self.H
                    new_tube_info['tubes'][tube_id].append(tmp_array)
        tube_box_dict['box_seq'] = new_tube_info  

        return tube_box_dict 



    def sample_frames_v2(self, tube_info, img_num):
        tube_box_dict = {}
        frm_num = len(tube_info['tubes'][0]) 
        smp_diff = int(frm_num/img_num)
        frm_offset = 0 if img_num==6 else int(img_num/2)
        frm_list = list(range(frm_offset, frm_num, smp_diff))
        for tube_id, tmp_tube in enumerate(tube_info['tubes']):
            tmp_dict = {}
            tmp_list = []
            count_idx = 0
            frm_ids = []
            for frm_id in frm_list:
                if tmp_tube[frm_id] == [0, 0, 1, 1]:
                    continue 
                tmp_list.append(copy.deepcopy(tmp_tube[frm_id]))
                frm_ids.append(frm_id)
                count_idx +=1
            # make sure each tube has at least one rgb
            if count_idx == 0:
                for frm_id in range(frm_num):
                    if tmp_tube[frm_id] == [0, 0, 1, 1]:
                        continue 
                    tmp_list.append(copy.deepcopy(tmp_tube[frm_id]))
                    frm_ids.append(frm_id)
                    count_idx +=1
                    frm_list.append(frm_id) 

            tmp_dict['boxes'] = tmp_list
            tmp_dict['frm_name'] = frm_ids  
            tube_box_dict[tube_id] = tmp_dict 
        frm_list_unique = list(set(frm_list))
        frm_list_unique.sort()

        tube_box_dict['frm_list'] = frm_list_unique  

        if self.args.normalized_boxes:
            frm_num = len(tube_info['tubes'][0]) 
            tube_num = len(tube_info['tubes']) 
            valid_flag_dict = np.ones((tube_num, frm_num))
            for tube_id, tmp_tube in enumerate(tube_info['tubes']):
                tmp_dict = {}
                frm_num = len(tmp_tube) 
                for frm_id in range(frm_num):
                    tmp_box = tmp_tube[frm_id]
                    if tmp_box == [0, 0, 1, 1]:
                        #tmp_box = [0, 0, 0, 0]
                        if self.args.new_mask_out_value_flag:
                            tmp_box = [-1*self.W, -1*self.H, -1*self.W, -1*self.H]
                        else:
                            tmp_box = [0, 0, 0, 0]
                        valid_flag_dict[tube_id, frm_id] = 0
                    x_c = (tmp_box[0] + tmp_box[2])* 0.5
                    y_c = (tmp_box[1] + tmp_box[3])* 0.5
                    w = tmp_box[2] - tmp_box[0]
                    h = tmp_box[3] - tmp_box[1]
                    tmp_array = np.array([x_c, y_c, w, h])
                    tmp_array[0] = tmp_array[0] / self.W
                    tmp_array[1] = tmp_array[1] / self.H
                    tmp_array[2] = tmp_array[2] / self.W
                    tmp_array[3] = tmp_array[3] / self.H
                    tube_info['tubes'][tube_id][frm_id] = tmp_array 
        tube_box_dict['box_seq'] = tube_info   
        return tube_box_dict , valid_flag_dict 

    def sample_tube_frames(self, index):
        prp_full_path = os.path.join(self.args.tube_prp_path, 'proposal_' + str(index).zfill(5)+'.pk') 
        # using gt proposal
        if not os.path.isfile(prp_full_path):
            if 'proposal' in prp_full_path:
                prp_full_path = prp_full_path.replace('proposal', 'annotation')
            elif 'annotation' in prp_full_path:
                prp_full_path = prp_full_path.replace('annotation', 'proposal')
        tube_info = pickleload(prp_full_path)
        return tube_info 

    def __len__(self):
        if self.args.debug:
            return 50
        else:
            if self.args.extract_region_attr_flag:
                return len(self.frm_ann)
            else:
                return len(self.question_ann)

    def make_dataloader(self, batch_size, shuffle, drop_last, nr_workers):
        from jactorch.data.dataloader import JacDataLoader

        def collate_dict(batch):
            return batch


        return JacDataLoader(
            self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
            num_workers=nr_workers, pin_memory=False,
            collate_fn=collate_dict)


def build_clevrer_dataset(args, phase):
    import jactorch.transforms.bbox as T
    image_transform = T.Compose([
        T.Resize(args.img_size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = clevrerDataset(args, phase=phase, img_transform=image_transform)
    
    return dataset


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.question_path = '/home/zfchen/code/nsclClevrer/clevrer/questions'
    args.tube_prp_path = '/home/zfchen/code/nsclClevrer/clevrer/tubeProposals/1.0_1.0' 
    args.frm_prp_path = '/home/zfchen/code/nsclClevrer/clevrer/proposals' 
    args.frm_img_path = '/home/zfchen/code/nsclClevrer/clevrer' 
    args.frm_img_num = 4
    args.img_size = 256
    phase = 'train'
    build_clevrer_dataset(args, phase)
    #dataset = clevrerDataset(args, phase)
    #dataset.parse_concepts_and_attributes()
    pdb.set_trace()
