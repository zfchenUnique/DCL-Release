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
    #DEBUG='ALL'
    DEBUG='OFF'
    result_idx = valid_num if valid_num is not None else i
    if not self.training and DEBUG != 'OFF':
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

        gogogo = False
        if p == l:
            print('Correct:', p)
            if DEBUG in ('ALL', 'CORRECT'):
                print('%s'%(fd['meta_ann']['questions'][i]['question']))
                gogogo = True
                #pdb.set_trace()
        else:
            print('Wrong: ', p, l)
            if DEBUG in ('ALL', 'WRONG'):
                gogogo = True
                print('%s'%(fd['meta_ann']['questions'][i]['question']))
                #print('%s'%(fd['meta_ann']['questions'][i]['program']))
                pdb.set_trace()

        if gogogo and 0:
            print('Starting the tracker.')
            ctx = make_debug_ctx(fd, buffer, i)
            from IPython import embed; embed()
            if ctx.stop:
                import sys; sys.exit()
