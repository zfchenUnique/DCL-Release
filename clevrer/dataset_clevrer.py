from torch.utils.data import Dataset, DataLoader
import pdb
import os
import sys
from .utils import jsonload, pickleload, pickledump, transform_conpcet_forms_for_nscl, set_debugger     
import argparse 
from PIL import Image
import copy
import numpy as np
import torch
from jacinle.utils.tqdm import tqdm
from nscl.datasets.definition import gdef
from nscl.datasets.common.vocab import Vocab
torch.multiprocessing.set_sharing_strategy('file_system')
#set_debugger()

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

class clevrerDataset(Dataset):
    def __init__(self, args, phase, img_transform=None):
        self.args = args
        self.phase = phase
        self.img_transform = img_transform  
        question_ann_full_path = os.path.join(args.question_path, phase+'.json') 
        self.question_ann = jsonload(question_ann_full_path)
        self.vocab = gen_vocab(self)
        self.W = 480
        self.H = 320

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

    def __getitem__(self, index):
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
        H=0
        W=0
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
            # ignoring temporal reasoning
            ignore_list = ['get_counterfact', 'unseen_events', 'filter_ancestor']
            valid_flag = True
            for pg in ques_info['program']:
                if pg in ignore_list:
                    valid_flag = False
                    break
            if not valid_flag:
                del meta_ann['questions'][q_id]
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
        #data['meta_ann'] = meta_used 
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
        tube_info = pickleload(prp_full_path)
        return tube_info 

    def __len__(self):
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
