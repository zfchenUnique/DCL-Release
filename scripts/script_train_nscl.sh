GPU_ID=$1
jac-crun ${GPU_ID} scripts/trainval_tube.py --desc clevrer/desc_nscl_derender_clevrer.py\
    --training-target derender --curriculum all\
    --dataset clevrer --data-dir ../clevrer\
    --batch-size 4 --epoch 100 --validation-interval 5\
    --save-interval 5 --data-split 0.95 --data-workers 2 \
    --normalized_boxes 1 --frm_img_num 6 --even_smp_flag 1\
    --rel_box_flag 0 --acc-grad 4 --dynamic_ftr_flag 1 --version v5_gt_new_col_scene_super_05\
    --scene_supervision_flag 1 \
    --tube_prp_path ../clevrer/tubeProposalsGt \
    --scene_add_supervision 1 \
    --scene_supervision_weight 0.5 \
    --box_iou_for_collision_flag 1
    #--resume dumps/clevrer/desc_nscl_derender_clevrer/derender_norm_box_even_smp6_col_box_ftr_v5_gt_thre_in_out_new/checkpoints/epoch_5.pth \
    #--resume dumps/clevrer/desc_nscl_derender_clevrer/derender_norm_box_even_smp6_col_box_ftr_v5_gt_sym_col_scene_super_balance_thre_in_out_new/checkpoints/epoch_5.pth
    #--load dumps/remote_models/scene_super_25.pth
    #--load dumps/remote_models/epoch_15.pth
    #--resume dumps/clevrer/desc_nscl_derender_clevrer/derender_norm_box_even_smp6_col_box_ftr_v5_gt_sym_col_scene_super_balance_thre_in_out_new/checkpoints/epoch_5.pth
    #--load dumps/clevrer/desc_nscl_derender_clevrer/derender_norm_box_even_smp6_col_box_ftr_v5_gt_sym_col_scene_supervision_balance/checkpoints/epoch_10.pth
    #--box_only_for_collision_flag 1 \
    #--resume dumps/clevrer/desc_nscl_derender_clevrer/derender_norm_box_even_smp6_col_box_ftr_v5_gt/checkpoints/epoch_24.pth  \

