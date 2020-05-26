GPU_ID=$1
jac-crun ${GPU_ID} scripts/script_extract_attribute_for_frames.py --desc clevrer/desc_nscl_derender_clevrer_v2.py\
    --training-target derender --curriculum all\
    --dataset clevrer --data-dir ../clevrer\
    --batch-size 8 --epoch 100 --validation-interval 5\
    --save-interval 5 --data-split 0.95 --data-workers 0 \
    --normalized_boxes 1 --frm_img_num 31 --even_smp_flag 1\
    --rel_box_flag 0 --evaluate \
    --dynamic_ftr_flag 1 --version v2  \
    --scene_supervision_flag 1 \
    --tube_prp_path ../clevrer/tubeProposals/1.0_1.0 \
    --box_only_for_collision_flag 0 \
    --scene_supervision_flag 1 \
    --box_iou_for_collision_flag 1 \
    --diff_for_moving_stationary_flag 1 \
    --extract_region_attr_flag 1 \
    --setname train \
    --load dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp31_col_box_ftr_v2_prp_stage1_ori/checkpoints/epoch_10.pth \
    --start_index 2537 \
    --output_attr_path dumps/clevrer/tmpProposalsAttrNew
    #--load dumps/clevrer/desc_nscl_derender_clevrer/derender_norm_box_even_smp6_col_box_ftr_v5_prp_new_col_diff_mov_new_mask_gau_std1/checkpoints/epoch_10.pth
    #--load dumps/clevrer/desc_nscl_derender_clevrer/derender_norm_box_even_smp6_col_box_ftr_v5_gt_new_col_diff_mov/checkpoints/epoch_20.pth \
    #--load dumps/clevrer/desc_nscl_derender_clevrer/derender_norm_box_even_smp6_col_box_ftr_v5_gt_new_col_scene_super_05/checkpoints/epoch_15.pth
    #--load dumps/clevrer/desc_nscl_derender_clevrer/derender_norm_box_even_smp6_col_box_ftr_v5_gt_sym_col_scene_supervision_balance/checkpoints/epoch_10.pth
    #--load dumps/clevrer/desc_nscl_derender_clevrer/derender_norm_box_even_smp6_col_box_ftr_v5_gt/checkpoints/epoch_16.pth \
    #--load dumps/clevrer/desc_nscl_derender_clevrer/derender_norm_box_even_smp6_col_box_ftr_v5_gt/epoch_16.pth --debug \

