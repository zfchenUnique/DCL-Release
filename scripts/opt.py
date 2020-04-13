from pprint import pprint
import argparse

def parse_opt():

    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--video_folder_path', type=str, default='../clevrer', help='')
    parser.add_argument('--img_folder_path', type=str, default='../clevrer', help='')
    parser.add_argument('--video_sample_num', type=int, default=25, help='')
    parser.add_argument('--connect_w', type=float, default=1.0, help='')
    parser.add_argument('--score_w', type=float, default=1.0, help='')
    parser.add_argument('--tube_folder_path', type=str, default='../clevrer/tubeProposalsAttrV0', help='')
    parser.add_argument('--tube_folder_new_path', type=str, default='../clevrer/tubeProposalsGt', help='')
    parser.add_argument('--use_attr_flag', type=int, default=1, help='')
    parser.add_argument('--ann_path', type=str, default='../clevrer', help='')
    parser.add_argument('--prp_path', type=str, default='../clevrer/proposals', help='')
    parser.add_argument('--inter_path', type=str, default='../clevrer/new_annotation', help='')
    parser.add_argument('--iou_w', type=float, default=0, help='')
    parser.add_argument('--conf_w', type=float, default=0, help='')
    parser.add_argument('--extract_att_path', type=str, default='dumps/clevrer/tmpProposalsAttr', help='')
    parser.add_argument('--vis_path', type=str, default='../videoParser/samples/attrVideo', help='')

    # parse 
    args = parser.parse_args()
    opt = vars(args)
    pprint('parsed input parameters:')
    pprint(opt)
    return args, opt

if __name__ == '__main__':

    opt = parse_opt()
    print('opt[\'id\'] is ', opt['id'])




