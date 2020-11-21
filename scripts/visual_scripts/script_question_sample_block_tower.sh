GPU_ID=$1
#MODEL_PATH=dumps/blocks/desc_nscl_derender_clevrer_v2_norm_box_even_smp16_col_box_ftr_v2_block_data_v2_4/checkpoints/epoch_25.pth
MODEL_PATH=dumps/blocks/desc_nscl_derender_clevrer_v2_norm_box_even_smp16_col_box_ftr_v2_block_data_v2_4_1/checkpoints/epoch_100.pth
TEST_PATH=v2_4_1_epoch_100
jac-crun ${GPU_ID} scripts/trainval_tube_v2.py --desc clevrer/desc_nscl_derender_clevrer_v2.py\
    --batch-size 1 --epoch 100 --validation-interval 5 \
    --save-interval 5 \
    --normalized_boxes 1 \
    --rel_box_flag 0 --acc-grad 1 --dynamic_ftr_flag  1 \
    --box_iou_for_collision_flag 1 \
    --colli_ftr_type 1 \
    --lr 0.0001 \
    --tube_mode 1 \
    --version v2 \
    --dataset blocks --data-dir ../real_blocks/block_camera_combined_scaled2 \
    --question_path ../question_blocks/dump/questions/balance_dataset_v2_4.json \
    --frm_img_path ../real_blocks/block_camera_combined_scaled2 \
    --scene_supervision_flag 1 \
    --scene_gt_path ../real_blocks/video_annotations_v3 \
    --tube_prp_path ../real_blocks/video_annotations_v3 \
    --frm_img_num 16 --even_smp_flag 1 \
    --scene_supervision_weight 10 \
    --new_mask_out_value_flag 1 \
    --apply_gaussian_smooth_flag 0 \
    --diff_for_moving_stationary_flag 0 \
    --prefix block_data_v2_4_1 \
    --scene_add_supervision 0 \
    --test_result_path ${TEST_PATH} \
    --load ${MODEL_PATH} \
    --prefix qa_blocks \
    --evaluate \
    --visualize_flag 1 \
    --visualize_gif_flag 1 \
    --debug \
    --data-workers 0 \
    --visualize_video_index 420 \
    #--visualize_video_index 224 \
    #--visualize_video_index 452 \
    # 459 \
    #--resume dumps/blocks/desc_nscl_derender_clevrer_v2_norm_box_even_smp16_col_box_ftr_v2_block_data_v2_4_1/checkpoints/epoch_5.pth \
    #--resume dumps/blocks/desc_nscl_derender_clevrer_v2_norm_box_even_smp16_col_box_ftr_v2_block_data_v2_4_1/checkpoints/epoch_5.pth \
    #--resume dumps/blocks/desc_nscl_derender_clevrer_v2_norm_box_even_smp16_col_box_ftr_v2_block/checkpoints/epoch_30.pth \
    #--evaluate \
    #--debug \
