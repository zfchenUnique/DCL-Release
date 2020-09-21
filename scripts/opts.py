"""
Paring parameters for training and evaulating the Neuro-Symbolic Concept Learner.
"""
from jacinle.cli.argument import JacArgumentParser

def load_param_parser():
    parser = JacArgumentParser(description=__doc__.strip())

    parser.add_argument('--desc', required=True, type='checked_file', metavar='FILE')
    parser.add_argument('--configs', default='', type='kv', metavar='CFGS')

    # training_target and curriculum learning
    parser.add_argument('--expr', default=None, metavar='DIR', help='experiment name')
    parser.add_argument('--training-target', required=True, choices=['derender', 'v2'])
    parser.add_argument('--training-visual-modules', default='all', choices=['none', 'object', 'relation', 'all'])
    parser.add_argument('--curriculum', default='all', choices=['off', 'scene', 'program', 'all'])
    parser.add_argument('--question-transform', default='off', choices=['off', 'basic', 'parserv1-groundtruth', 'parserv1-candidates', 'parserv1-candidates-executed'])
    parser.add_argument('--concept-quantization-json', default=None, metavar='FILE')

    # running mode
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--evaluate', action='store_true', help='run the validation only; used with --resume')

    # training hyperparameters
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of total epochs to run')
    parser.add_argument('--enums-per-epoch', type=int, default=1, metavar='N', help='number of enumerations of the whole dataset per epoch')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='N', help='initial learning rate')
    parser.add_argument('--iters-per-epoch', type=int, default=0, metavar='N', help='number of iterations per epoch 0=one pass of the dataset (default: 0)')
    parser.add_argument('--acc-grad', type=int, default=1, metavar='N', help='accumulated gradient (default: 1)')
    parser.add_argument('--clip-grad', type=float, metavar='F', help='gradient clipping')
    parser.add_argument('--validation-interval', type=int, default=1, metavar='N', help='validation inverval (epochs) (default: 1)')

    # finetuning and snapshot
    parser.add_argument('--load', type='checked_file', default=None, metavar='FILE', help='load the weights from a pretrained model (default: none)')
    parser.add_argument('--resume', type='checked_file', default=None, metavar='FILE', help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', type=int, default=0, metavar='N', help='manual epoch number')
    parser.add_argument('--save-interval', type=int, default=2, metavar='N', help='model save interval (epochs) (default: 10)')

    # data related
    parser.add_argument('--dataset', required=True, choices=['clevrer'], help='dataset')
    parser.add_argument('--data-dir', required=True, type='checked_dir', metavar='DIR', help='data directory')
    parser.add_argument('--data-trim', type=float, default=0, metavar='F', help='trim the dataset')
    parser.add_argument('--data-split',type=float, default=0.75, metavar='F', help='fraction / numer of training samples')
    parser.add_argument('--data-vocab-json', type='checked_file', metavar='FILE')
    parser.add_argument('--data-scenes-json', type='checked_file', metavar='FILE')
    parser.add_argument('--data-questions-json', type='checked_file', metavar='FILE', nargs='+')

    parser.add_argument('--extra-data-dir', type='checked_dir', metavar='DIR', help='extra data directory for validation')
    parser.add_argument('--extra-data-scenes-json', type='checked_file', nargs='+', default=None, metavar='FILE', help='extra scene json file for validation')
    parser.add_argument('--extra-data-questions-json', type='checked_file', nargs='+', default=None, metavar='FILE', help='extra question json file for validation')

    parser.add_argument('--data-workers', type=int, default=4, metavar='N', help='the num of workers that input training data')

    # misc
    parser.add_argument('--use-gpu', type='bool', default=True, metavar='B', help='use GPU or not')
    parser.add_argument('--use-tb', type='bool', default=False, metavar='B', help='use tensorboard or not')
    parser.add_argument('--embed', action='store_true', help='entering embed after initialization')
    parser.add_argument('--force-gpu', action='store_true', help='force the script to use GPUs, useful when there exists on-the-ground devices')

    # for clevrer dataset
    parser.add_argument('--question_path', default='../clevrer/questions')
    parser.add_argument('--tube_prp_path', default='../clevrer/tubeProposals/1.0_1.0') 
    parser.add_argument('--frm_prp_path', default='../clevrer/proposals')
    parser.add_argument('--frm_img_path', default='../clevrer') 
    parser.add_argument('--frm_img_num', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--normalized_boxes', type=int, default=0)
    parser.add_argument('--even_smp_flag', type=int, default=0)
    parser.add_argument('--rel_box_flag', type=int, default=0)
    parser.add_argument('--dynamic_ftr_flag', type=int, default=1)
    parser.add_argument('--version', type=str, default='v0')
    parser.add_argument('--scene_supervision_flag', type=int, default=0)
    parser.add_argument('--scene_gt_path', type=str, default='../clevrer')
    parser.add_argument('--mask_gt_path', type=str, default='../clevrer/proposals/')
    parser.add_argument('--box_only_for_collision_flag', type=int, default=0)
    parser.add_argument('--scene_add_supervision', type=int, default=0)
    parser.add_argument('--scene_supervision_weight', type=float, default=1.0)
    parser.add_argument('--box_iou_for_collision_flag', type=int, default=1)
    parser.add_argument('--diff_for_moving_stationary_flag', type=int, default=1)
    parser.add_argument('--new_mask_out_value_flag', type=int, default=1)
    parser.add_argument('--apply_gaussian_smooth_flag', type=int, default=0)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--extract_region_attr_flag', type=int, default=0)
    parser.add_argument('--smp_coll_frm_num', type=int, default=32)
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--colli_ftr_type', type=int, default=1, help='0 for average rgb, 1 for KNN sampling')
    parser.add_argument('--n_seen_frames', type=int, default=128, help='')
    parser.add_argument('--unseen_events_path', type=str, default='/home/zfchen/code/nsclClevrer/temporal_reasoning-master/propnet_predictions_v1.0_noAttr_noEdgeSuperv', help='')
    parser.add_argument('--background_path', type=str, default='/home/zfchen/code/nsclClevrer/temporal_reasoning-master/background.png', help='')
    parser.add_argument('--bgH', type=int, default=100)
    parser.add_argument('--bgW', type=int, default=150)
    parser.add_argument('--max_counterfact_num', type=int, default=2)
    
    # for temporal prediction model
    parser.add_argument('--pred_model_path', type=str, default='')
    #parser.add_argument('--pretrain_pred_model_path', required=True, type='checked_file', metavar='FILE')
    parser.add_argument('--pretrain_pred_model_path', type=str,  default='')
    parser.add_argument('--attr_dim', type=int, default=5)
    # [dx, dy, dw, dh, ftr_dim]
    parser.add_argument('--state_dim', type=int, default=260)
    parser.add_argument('--n_his', type=int, default=2)
    parser.add_argument('--nf_relation', type=int, default=128)
    parser.add_argument('--nf_particle', type=int, default=128)
    parser.add_argument('--nf_effect', type=int, default=128*4)
    parser.add_argument('--use_attr', type=int, default=0, help='whether using attributes or not')
    parser.add_argument('--pred_frm_num', type=int, default=12, help='number of frames to predict')
    parser.add_argument('--pstep', type=int, default=2)
    parser.add_argument('--frame_offset', type=int, default=4)
    parser.add_argument('--colli_threshold', type=float, default=0.0)
    # use correct question parser
    parser.add_argument('--correct_question_path', type=str, default='../language_parsing/data/new_results/')
    parser.add_argument('--correct_question_flag', type=int, default=1)
    parser.add_argument('--dataset_stage', type=int, default=-1, help='0 for descriptive only')
    parser.add_argument('--data_train_length', type=int, default=-1, help='for evaluating data efficiency.')
    parser.add_argument('--testing_flag', type=int, default=0, help='1 for testing on the testing set')
    parser.add_argument('--test_result_path', type=str, default='', help='file path to store the result')
    parser.add_argument('--visualize_flag', type=int, default=0, help='1 for visualizing data')
    parser.add_argument('--regu_flag', type=int, default=0, help='1 for visualizing data')
    parser.add_argument('--pred_normal_num', type=int, default=12, help='number of frames to predict for regularization')
    parser.add_argument('--regu_weight', type=float, default=10.0)
    parser.add_argument('--regu_only_flag', type=int, default=0, help='1 for visualizing data')
    parser.add_argument('--freeze_learner_flag', type=int, default=0, help='1 for visualizing data')
    parser.add_argument('--residual_rela_prop', type=int, default=0, help='1 for residual encoding for relations')
    parser.add_argument('--residual_rela_pred', type=int, default=0, help='1 for residual encoding for relations')
    parser.add_argument('--rela_spatial_only', type=int, default=0, help='1 for residual encoding for relations')
    # [dx, dy, dw, dh, collision_ftr]
    parser.add_argument('--relation_dim', type=int, default=260)
    parser.add_argument('--rela_spatial_dim', type=int, default=4)
    parser.add_argument('--rela_ftr_dim', type=int, default=256)
    parser.add_argument('--pred_res_flag', type=int, default=0, help='1 for residual encoding for prediction')
    parser.add_argument('--add_rela_dist_mode', type=int, default=0)
    parser.add_argument('--rela_dist_thre', type=float, default=0.2)
    parser.add_argument('--rela_dist_loss_flag', type=int, default=0)
    
    # for v5 that separately encode spatial and semantics
    parser.add_argument('--pred_spatial_model_path', type=str, default='')
    parser.add_argument('--pretrain_pred_spatial_model_path', type=str,  default='')
    parser.add_argument('--box_only_flag', type=int, default=0)
    parser.add_argument('--bbox_size', type=int, default=24)
    parser.add_argument('--tube_mode', type=int, default=0)
    parser.add_argument('--semantic_only_flag', type=int, default=0)
    parser.add_argument('--residual_obj_pred', type=int, default=0)
    parser.add_argument('--ftr_in_collision_space_flag', type=int, default=0)
    parser.add_argument('--pretrain_pred_feature_model_path', type=str,  default='')
    parser.add_argument('--add_kl_regu_flag', type=int,  default=0)
    parser.add_argument('--kl_weight', type=float, default=1.0)
    parser.add_argument('--reconstruct_flag', type=int, default=0)
    parser.add_argument('--reconstruct_weight', type=float, default=0.01)
    # for expression
    parser.add_argument('--expression_mode', type=int, default=-1)
    parser.add_argument('--expression_path', type=str, default='')
    parser.add_argument('--tube_gt_path', default='../clevrer/tubeProposalsGt') 
    parser.add_argument('--exp_ground_thre', type=float, default=0.5)
    # for retireval expression
    parser.add_argument('--retrieval_mode', type=int, default=-1)
    parser.add_argument('--visualize_retrieval_id', type=int, default=-1)
    parser.add_argument('--visualize_gif_flag', type=int, default=0)
    parser.add_argument('--visualize_ground_vid', type=int, default=-1)
    parser.add_argument('--expression_result_path', type=str, default='', help='file path to store the grounding/ retrieval result')
    args = parser.parse_args()
    return args 

if __name__ == '__main__':
    args = load_param_parser()


