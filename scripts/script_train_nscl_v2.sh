GPU_ID=$1
jac-crun ${GPU_ID} scripts/trainval_tube_v2.py --desc clevrer/desc_nscl_derender_clevrer_v2.py\
    --training-target v2 \
    --dataset clevrer --data-dir ../clevrer\
    --batch-size 4 --epoch 100 --validation-interval 5\
    --save-interval 5 --data-split 0.95 --data-workers 0 \
    --normalized_boxes 1 --frm_img_num 6 --even_smp_flag 1\
    --rel_box_flag 0 --acc-grad 4 --dynamic_ftr_flag 1 --version v2\
    --scene_supervision_flag 1 \
    --tube_prp_path ../clevrer/tubeProposalsGt \
    --scene_add_supervision 0 \
    --scene_supervision_weight 0.5 \
    --box_iou_for_collision_flag 1 \
    --diff_for_moving_stationary_flag 1 \
    --new_mask_out_value_flag 1 \
    --apply_gaussian_smooth_flag 1 \

