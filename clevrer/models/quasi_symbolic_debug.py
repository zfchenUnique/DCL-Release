#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : quasi_symbolic_debug.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/03/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

"""
A context for debugging differentiable reasoning results.
"""

import os
import json

import numpy as np

from PIL import Image
import pdb

DEBUG = os.getenv('REASONING_DEBUG', 'OFF').upper() 
__all__ = ['make_debug_ctx', 'embed']


def make_debug_ctx(fd, buffer, i):
    class Context(object):
        def __init__(ctx):
            ctx.stop = False
            ctx.fd = fd
            ctx.buffer = buffer
            ctx.i = i
            ctx.program = json.loads(fd['program'][i])
            ctx.scene = json.loads(fd['scene'][i])
            ctx.raw_program = json.loads(fd['raw_program'][i])

            import visdom
            ctx.vis = visdom.Visdom(port=7002)

        def backtrace(ctx):
            print('----- backtrace ----- ')
            print('See localhost:7002 for the image.')
            ctx.vis_image()
            print(ctx.program['question'])
            for p, b, raw_p in zip(ctx.program['program'], ctx.buffer, ctx.raw_program):
                print(p)
                print('Output:', b)
                print('GT:', raw_p['_output'])

        def vis_image(ctx):
            img = Image.open(ctx.fd['image_path'][i]).convert('RGB')
            img = np.array(img).transpose((2, 0, 1))
            ctx.vis.image(img)

    return Context()


def embed(self, i, buffer, result, fd, valid_num=None):

    DEBUG = 'ALL' if self.args.debug and self.args.visualize_flag==1 else 'OFF'
    #DEBUG = 'ALL' if self.args.debug else 'OFF'
    result_idx = valid_num if valid_num is not None else i
    #if not self.training and DEBUG != 'OFF':

    #if prog[-1]['op']=='get_col_partner':
    #if prog[-1]['op']=='filter_in' and 0:
    #if 0 and prog[-1]['op']=='filter_collision' and buffer[-1]!='error':
    #if 0 and prog[-1]['op']=='get_col_partner' and buffer[-1]!='error':
    if 0 and prog[-2]['op']=='filter_temporal' and buffer[-1]!='error':
        #tmp_gt = feed_dict['meta_ann']['questions'][i]['answer'] 
        #tmp_an = int(torch.argmax(buffer[-1][0]))
        #tmp_an = torch.max(buffer[-1][0])
        tmp_an = torch.max(buffer[-1])
        if i in feed_dict['meta_ann']['pos_id_list'] and tmp_an<0:
            pdb.set_trace()





    if  DEBUG != 'OFF':
    #if True:
        p, l = result[result_idx][1], fd['answer'][i]
        if isinstance(p, tuple):
            p, word2idx = p
            p = p.argmax(-1).item()
            idx2word = {v: k for k, v in word2idx.items()}
            p = idx2word[p]
        elif fd['question_type'][i] == 'exist':
            p, l = int((p > 0).item()), int(l)
        elif isinstance(p, list):
            new_p = []; new_l = []
            for idx in range(len(p)):
                new_l.append(None)
                new_p.append(None)
                new_p[idx], new_l[idx] = int((p[idx] > 0).item()), int(l[idx])
        else:
            p, l = int(p.item()), int(l)

        if not isinstance(p, list):
            new_p = p
            new_l = l

        if fd['meta_ann']['questions'][i]['question_type']!='counterfactual' and fd['meta_ann']['questions'][i]['question_type']!='predictive': 
            return  
        gogogo = False
        if new_p == new_l:
            print('Correct:', new_p)
            if DEBUG in ('ALL', 'CORRECT'):
                gogogo = True
                if fd['meta_ann']['questions'][i]['question_type']=='counterfactual' or fd['meta_ann']['questions'][i]['question_type']=='predictive': 
                    print('%s'%(fd['meta_ann']['questions'][i]['question']))
                    #pdb.set_trace()
        else:
            if DEBUG in ('ALL', 'WRONG'):
                gogogo = True
                if fd['meta_ann']['questions'][i]['question_type']=='counterfactual' or \
                     fd['meta_ann']['questions'][i]['question_type']=='predictive': 
                    print('Wrong: ', new_p, new_l)
                    print('%s'%(fd['meta_ann']['questions'][i]['question']))
                    for choice_info in fd['meta_ann']['questions'][i]['choices']:
                        print(choice_info['program'])
                    print('\n')
                    print(fd['meta_ann']['scene_index'])
                    print('\n')
                    #pdb.set_trace()
                #print('%s'%(fd['meta_ann']['questions'][i]['program']))

