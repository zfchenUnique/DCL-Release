GPU_ID=$1
jac-crun ${GPU_ID} scripts/trainval_tube_v2.py --desc clevrer/desc_nscl_derender_clevrer_v2.py\
    --batch-size 1 --epoch 30 --validation-interval 5 \
    --save-interval 5 \
    --normalized_boxes 1 \
    --rel_box_flag 0 --acc-grad 1 --dynamic_ftr_flag  1 \
    --box_iou_for_collision_flag 1 \
    --diff_for_moving_stationary_flag 1 \
    --new_mask_out_value_flag 1 \
    --apply_gaussian_smooth_flag 1 \
    --colli_ftr_type 1 \
    --lr 0.0001 \
    --tube_mode 1 \
    --version v2 \
    --dataset blocks --data-dir ../real_blocks/block_camera_combined_scaled2 \
    --question_path ../question_blocks/dump/questions/balance_dataset.json \
    --tube_prp_path ../real_blocks/video_annotations_v2 \
    --frm_img_path ../real_blocks/block_camera_combined_scaled2 \
    --scene_supervision_flag 1 \
    --scene_supervision_weight 0.5 \
    --scene_add_supervision 0 \
    --prefix block \
    --data-workers 2 \
    --scene_gt_path ../real_blocks/video_annotations_v2 \
    --frm_img_num 16 --even_smp_flag 1 \
    #--debug \
