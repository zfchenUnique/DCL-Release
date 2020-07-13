import json
import pdb

def evaluate_accuracy():
    gt_ques_path = '../question_parsing/data/raw_questions/open_ended_questions.json'
    prp_ques_path = '../question_parsing/data/new_results/oe_1000pg_train.json'
    
    fh_gt = open(gt_ques_path, 'r')
    gt_feed_dict = json.load(fh_gt)
    fh_prp = open(prp_ques_path, 'r')
    prp_feed_dict = json.load(fh_prp)
    pdb.set_trace()

if __name__=='__main__':
    evaluate_accuracy()
