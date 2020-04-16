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

#torch.multiprocessing.set_sharing_strategy('file_system')
#set_debugger()
#_ignore_list = ['get_counterfact', 'unseen_events', 'filter_ancestor', 'filter_in', 'filter_out', 'filter_order', 'start', 'filter_moving', 'filter_stationary', 'filter_order', 'end']
#_ignore_list = ['get_counterfact', 'unseen_events', 'filter_ancestor', 'filter_order']
_ignore_list = ['get_counterfact', 'unseen_events', 'filter_ancestor']
_used_list = ['filter_order']

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
                if 'answer' not in ques_info.keys():
                    continue 
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
            if self.args.version == 'v2':
                return self.__getitem__model_v2(index)
            else:
                return self.__getitem__model(index)

    def __getitem__model_v2(self, index):
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

            program_cl = transform_conpcet_forms_for_nscl_v2(ques_info['program'])
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
        frm_list = list(range(0, frm_num, smp_diff))
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
