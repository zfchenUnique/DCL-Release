#! /usr/bin/env python3
## -*- coding: utf-8 -*-
# Distributed under terms of the MIT license.

"""
Extract frame attributes for frames with Neuro-Symbolic Concept Learner.
"""

import pdb

import time
import os.path as osp

import torch.backends.cudnn as cudnn
import torch.cuda as cuda

from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger, set_output_file
from jacinle.utils.imp import load_source
from jacinle.utils.tqdm import tqdm_pbar

from jactorch.cli import escape_desc_name, ensure_path, dump_metainfo
from jactorch.cuda.copy import async_copy_to
from jactorch.train import TrainerEnv
from jactorch.utils.meta import as_float

from nscl.datasets import get_available_datasets, initialize_dataset, get_dataset_builder
from clevrer.dataset_clevrer import build_clevrer_dataset  

from clevrer.utils import set_debugger, jsondump, jsonload 
from nscl.datasets.definition import gdef
import torch
import os
set_debugger()

logger = get_logger(__file__)

parser = JacArgumentParser(description='')

parser.add_argument('--desc', required=True, type='checked_file', metavar='FILE')
parser.add_argument('--configs', default='', type='kv', metavar='CFGS')

# training_target and curriculum learning
parser.add_argument('--expr', default=None, metavar='DIR', help='experiment name')
parser.add_argument('--training-target', required=True, choices=['derender', 'parser', 'all'])
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
parser.add_argument('--dynamic_ftr_flag', type=int, default=0)
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
parser.add_argument('--setname', type=str, default='validation')
parser.add_argument('--extract_region_attr_flag', type=int, default=0)
parser.add_argument('--output_attr_path', type=str, default='dumps/clevrer/tmpProposalsAttr')
parser.add_argument('--start_index', type=int, default=0)
parser.add_argument('--correct_question_path', type=str, default='../question_parsing/data/new_results/')
parser.add_argument('--correct_question_flag', type=int, default=1)
parser.add_argument('--dataset_stage', type=int, default=-1, help='0 for descriptive only')
parser.add_argument('--data_train_length', type=int, default=-1, help='for evaluating data efficiency.')
parser.add_argument('--colli_ftr_type', type=int, default=1, help='0 for average rgb, 1 for KNN sampling')
parser.add_argument('--smp_coll_frm_num', type=int, default=32)

args = parser.parse_args()

if args.data_vocab_json is None:
    args.data_vocab_json = osp.join(args.data_dir, 'vocab.json')

args.data_image_root = osp.join(args.data_dir, 'images')
if args.data_scenes_json is None:
    args.data_scenes_json = osp.join(args.data_dir, 'scenes.json')
if args.data_questions_json is None:
    args.data_questions_json = osp.join(args.data_dir, 'questions.json')

if args.extra_data_dir is not None:
    args.extra_data_image_root = osp.join(args.extra_data_dir, 'images')
    if args.extra_data_scenes_json is None:
        args.extra_data_scenes_json = osp.join(args.extra_data_dir, 'scenes.json')
    if args.extra_data_questions_json is None:
        args.extra_data_questions_json = osp.join(args.extra_data_dir, 'questions.json')

# filenames
args.series_name = args.dataset
args.desc_name = escape_desc_name(args.desc)
args.run_name = 'run-{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))

# directories

if args.use_gpu:
    nr_devs = cuda.device_count()
    if args.force_gpu and nr_devs == 0:
        nr_devs = 1
    assert nr_devs > 0, 'No GPU device available'
    args.gpus = [i for i in range(nr_devs)]
    args.gpu_parallel = (nr_devs > 1)

desc = load_source(args.desc)
configs = desc.configs
args.configs.apply(configs)


def main():
    args.dump_dir = ensure_path(osp.join(
        'dumps', args.series_name, args.desc_name, (
            args.training_target)
    ))
    if args.normalized_boxes:
        args.dump_dir = args.dump_dir + '_norm_box'
    if args.even_smp_flag:
        args.dump_dir = args.dump_dir + '_even_smp'+str(args.frm_img_num)
    if args.even_smp_flag:
        args.dump_dir = args.dump_dir + '_col_box_ftr'
    args.dump_dir +=  '_' + args.version 

    if not args.debug:
        args.ckpt_dir = ensure_path(osp.join(args.dump_dir, 'checkpoints'))
        args.meta_dir = ensure_path(osp.join(args.dump_dir, 'meta'))
        args.meta_file = osp.join(args.meta_dir, args.run_name + '.json')
        args.log_file = osp.join(args.meta_dir, args.run_name + '.log')
        args.meter_file = osp.join(args.meta_dir, args.run_name + '.meter.json')

        logger.critical('Writing logs to file: "{}".'.format(args.log_file))
        set_output_file(args.log_file)

        logger.critical('Writing metainfo to file: "{}".'.format(args.meta_file))
        with open(args.meta_file, 'w') as f:
            f.write(dump_metainfo(args=args.__dict__, configs=configs))

        # Initialize the tensorboard.
        if args.use_tb:
            args.tb_dir_root = ensure_path(osp.join(args.dump_dir, 'tensorboard'))
            args.tb_dir = ensure_path(osp.join(args.tb_dir_root, args.run_name))

    initialize_dataset(args.dataset)
    # to replace dataset
    validation_dataset = build_clevrer_dataset(args, args.setname)
    main_train(validation_dataset)

def main_train(validation_dataset):
    logger.critical('Building the model.')
    model = desc.make_model(args)

    if args.use_gpu:
        model.cuda()
        # Use the customized data parallel if applicable.
        if args.gpu_parallel:
            from jactorch.parallel import JacDataParallel
            # from jactorch.parallel import UserScatteredJacDataParallel as JacDataParallel
            model = JacDataParallel(model, device_ids=args.gpus).cuda()
        # Disable the cudnn benchmark.
        cudnn.benchmark = False

    trainer = TrainerEnv(model, None)

    if args.load:
        if trainer.load_weights(args.load):
            logger.critical('Loaded weights from pretrained model: "{}".'.format(args.load))

        from jacinle.utils.meter import GroupMeters
        meters = GroupMeters()

    logger.critical('Building the data loader.')
    validation_dataloader = validation_dataset.make_dataloader(args.batch_size, shuffle=False, drop_last=False, nr_workers=args.data_workers)

    meters.reset()
    model.eval()

    if not os.path.isdir(args.output_attr_path):
        os.makedirs(args.output_attr_path)
    validate_attribute(model, validation_dataloader, meters, args.setname, logger, args.output_attr_path)
    logger.critical(meters.format_simple(args.setname, {k: v for k, v in meters.avg.items() if v != 0}, compressed=False))
    return meters


def validate_attribute(model, val_dataloader, meters, meter_prefix='validation', logger=None, output_attr_path=''):
    end = time.time()
    video_num  = len(val_dataloader)
    #pdb.set_trace()
    with tqdm_pbar(total= int(len(val_dataloader)*args.batch_size/128)) as pbar:
        output_dict_list = []
        frame_id_list = []
        for feed_dict_list in val_dataloader: 
        #for vid in range(video_num):
            end_frm_flag = False
            #while (not end_frm_flag):
            for idx, feed_dict in enumerate(feed_dict_list):
                scene_idx = feed_dict['meta_ann']['scene_index']
                full_path = os.path.join(output_attr_path, 'attribute_'+str(scene_idx).zfill(5)+'.json') 
                if os.path.isfile(full_path):
                    print('File exists. %s\n' %(full_path))
                    tmp_dict = jsonload(full_path)
                    if len(tmp_dict)== len(feed_dict['tube_info']['box_seq']['tubes'][0]):
                        continue 
                    print('size didn\'t match. %s\n' %(full_path))
                    #pdb.set_trace()
                if args.use_gpu:
                    if not args.gpu_parallel:
                        feed_dict = async_copy_to(feed_dict, 0)
                frm_id = feed_dict['frm_id']                
                data_time = time.time() - end; end = time.time()
                
                f_scene = model.resnet(feed_dict['img'])
                f_sng = model.scene_graph(f_scene, feed_dict)
                output_dict = parse_scene(feed_dict, f_sng, model.reasoning.embedding_attribute, frm_id)
                #pdb.set_trace()
                output_dict_list.append(output_dict)
                frame_id_list.append(frm_id) 

                step_time = time.time() - end; end = time.time()
                if frm_id == len(feed_dict['tube_info']['box_seq']['tubes'][0])-1:
                    video_attr_list = [] 
                    for idx, result_dict in enumerate(output_dict_list): 
                        mon_dict = result_dict.pop('monitors')
                        result_dict['frm_id'] = frame_id_list[idx]
                        video_attr_list.append(result_dict)
                        monitors = {meter_prefix + '/' + k: v for k, v in as_float(mon_dict).items()}

                        n = 1
                        meters.update(monitors, n=n)
                        meters.update({'time/data': data_time, 'time/step': step_time})

                    jsondump(full_path, video_attr_list)

                    if args.use_tb:
                        meters.flush()

                    pbar.set_description(meters.format_simple(
                        '({})'.format(args.setname),
                        {k: v for k, v in meters.val.items() if k.startswith('validation') and k.count('/') <= 2},
                        compressed=True
                    ))
                    pbar.update()

                    end = time.time()
                    output_dict_list = []
                    frame_id_list = []
                    if logger is not None:
                        logger.critical(meters.format_simple(meter_prefix, {k: v for k, v in meters.avg.items() if v != 0}, compressed=False))


def parse_scene(feed_dict, f_sng, attribute_embedding, frm_id):
    all_f = f_sng[1]
    monitors = {}
    output_dict = {}
    for attribute, concepts in gdef.all_concepts_clevrer['attribute'].items():
        if 'attribute_' + attribute not in feed_dict:
            continue
        all_scores = []
        for v in concepts:
            this_score = attribute_embedding.similarity(all_f, v)
            all_scores.append(this_score)

        all_scores = torch.stack(all_scores, dim=-1)
        seleted_score_list  = []
        for gt_id, t_id in feed_dict['prp_id_to_t_id'].items():
            seleted_score_list.append(all_scores[t_id]) 
        all_scores = torch.stack(seleted_score_list, dim=0)

        all_labels = feed_dict['attribute_' + attribute]

        if all_labels.dim() == all_scores.dim() - 1:
            acc_key = 'acc/scene/attribute/' + attribute
            monitors[acc_key] = (
                ((all_scores > 0).float().sum(dim=-1) == 1) *
                (all_scores.argmax(-1) == all_labels.long())
            ).float().mean()
        output_dict[attribute] = all_scores.argmax(-1).cpu().data.tolist()
    output_dict['monitors'] = monitors 
    return output_dict         

if __name__ == '__main__':
    main()

