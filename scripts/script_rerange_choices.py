import json
import argparse
import pdb
import copy


def set_debugger():
    import sys
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(call_pdb=True)

set_debugger()

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=str, required=True)
parser.add_argument('--out_path', type=str, default='test_to_submit.json')
parser.add_argument('--raw_ques_path', type=str, default='../language_parsing/data/raw_questions/multiple_choice_questions.json')
parser.add_argument('--release_ques_path', type=str, default='../clevrer/questions/test.json')

def jsondump(path, this_dic):
    f = open(path, 'w')
    this_ans = json.dump(this_dic, f)
    f.close()

def jsonload(path):
    f = open(path)
    this_ans = json.load(f)
    f.close()
    return this_ans

def rerange_choices(args):
    release_dict_list = jsonload(args.release_ques_path)
    raw_ques_dict_list = jsonload(args.raw_ques_path)
    pred_dict_list = jsonload(args.in_path)
    new_list_output = copy.deepcopy(pred_dict_list)
    
    for i in range(len(pred_dict_list)):
        pred = pred_dict_list[i]
        scene_index = pred['scene_index']
        multi_id = 0
        assert scene_index == release_dict_list[i]['scene_index']
        assert scene_index == raw_ques_dict_list[i+15000]['scene_index']
        #pdb.set_trace()
        for q_id, ques_info in enumerate(pred['questions']):
            ques_type = release_dict_list[i]['questions'][q_id]['question_type']
            if ques_type == 'descriptive':
                continue
            assert  raw_ques_dict_list[i+15000]['questions'][multi_id]['question'] == \
                release_dict_list[i]['questions'][q_id]['question']

            raw_ques_info  = raw_ques_dict_list[i+15000]['questions'][multi_id]
            release_choice_info = release_dict_list[i]['questions'][q_id]['choices']
            
            new_pred_answer = reorder_choice(raw_ques_info, release_choice_info, ques_info['choices'])
            new_list_output[i]['questions'][q_id]['choices'] = new_pred_answer          

            #pdb.set_trace()
            multi_id +=1

    jsondump(args.out_path, new_list_output)


def reorder_choice(raw_ques_info, release_choice_info, pred_answer): 
    raw_to_release = []
    raw_id = 0
    release_to_raw = {}
    for cc in raw_ques_info['correct']:
        raw_q = cc[0]
        for tmp_id, choice_info in enumerate(release_choice_info):
            if raw_q == choice_info['choice']:
                raw_to_release.append(choice_info['choice_id'])
                relea_id = choice_info['choice_id']
                if relea_id not in release_to_raw.keys():
                    release_to_raw[relea_id] = [raw_id]
                else:
                    release_to_raw[relea_id].append(raw_id)
        raw_id +=1 
    
    for wc in raw_ques_info['wrong']:
        raw_q = wc[0]
        for tmp_id, choice_info in enumerate(release_choice_info):
            if raw_q == choice_info['choice']:
                raw_to_release.append(choice_info['choice_id'])
                relea_id = choice_info['choice_id']
                if relea_id not in release_to_raw.keys():
                    release_to_raw[relea_id] = [raw_id]
                else:
                    release_to_raw[relea_id].append(raw_id)
        
        raw_id +=1 

    new_answer = [{'choice_id': idx} for idx in range(len(pred_answer))]
    for tmp_idx, rel_to_raw in release_to_raw.items():
        if len(rel_to_raw) >1:
            print('duplicate choice')
            assert pred_answer[rel_to_raw[0]]['answer'] == pred_answer[rel_to_raw[1]]['answer']
        
        new_answer[tmp_idx]['answer'] = pred_answer[rel_to_raw[0]]['answer']
    return new_answer 

if __name__ == '__main__':
    args = parser.parse_args()
    rerange_choices(args)
