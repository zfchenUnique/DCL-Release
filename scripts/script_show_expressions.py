import json
import pdb

def print_exp():
    exp_path = '/home/zfchen/code/nsclClevrer/clevrer/expressions/exp_val_retrieval_v5/5000_100_0/refine_retrieval_exp.json'
    fh = open(exp_path, 'r')
    f_dict = json.load(fh)
    exp_id_list = f_dict['vid2exp']['10000']
    exp_list = [f_dict['expressions'][exp_id]['question'] for exp_id in exp_id_list]
    print(exp_list)
    pdb.set_trace()

def print_stat():
    exp_path = '/home/zfchen/code/nsclClevrer/clevrer/expressions/exp_train_retrieval_v5/refine_retrieval_exp.json'
    fh = open(exp_path, 'r')
    f_dict = json.load(fh)
    total_exp_num = 0
    for key_id, f_info in f_dict.items():
        for exp_type, exp_list in f_info.items():
            total_exp_num +=len(exp_list)
    avg_num = total_exp_num / len(f_dict)
    print('average expression number: %f' %(avg_num))
    pdb.set_trace()

def print_stat_grounding():
    exp_path = '/home/zfchen/code/nsclClevrer/clevrer/expressions/exp_train_grounding_v2/refine_grounding_exp.json'
    fh = open(exp_path, 'r')
    f_dict = json.load(fh)
    total_exp_num = 0
    for key_id, f_info in f_dict.items():
        for exp_type, exp_list in f_info.items():
            total_exp_num +=len(exp_list)

    exp_path = '/home/zfchen/code/nsclClevrer/clevrer/expressions/exp_val_grounding_v2/refine_grounding_exp.json'
    fh = open(exp_path, 'r')
    f_dict_val = json.load(fh)
    for key_id, f_info in f_dict_val.items():
        for exp_type, exp_list in f_info.items():
            total_exp_num +=len(exp_list)
    avg_num = total_exp_num / (len(f_dict_val)+len(f_dict))
    print('average expression number: %f' %(avg_num))
    pdb.set_trace()

def sample_specific_question(f_dict, q_id):
    q_list_dict = f_dict[q_id]
    q_list = [(q_id, q_info['question']) for q_id, q_info in enumerate(q_list_dict['questions']) if q_info['question_type']=='explanatory' and 'not' not in q_info['question']]
    print(q_list)

def sample_video():
    fn_path = '../clevrer/questions/validation.json'
    fh = open(fn_path, 'r')
    f_dict = json.load(fh)
    vid = 10004
    q_id = vid - 10000
    sample_specific_question(f_dict, q_id)
    pdb.set_trace()

if __name__=='__main__':
    #print_exp()
    #print_stat()
    #print_stat_grounding()
    sample_video()
