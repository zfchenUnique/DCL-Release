GPU_ID=$1
MODEL_PATH=data/models/block_model/epoch_100.pth
ANN_PATH=data/raw_data/block_annotation/video_annotations_v3
VIDEO_PATH=./data/raw_data/block_annotation/block_camera_combined_scaled2
QUESTION_PATH=data/raw_data/block_annotation/balance_dataset_v2_4.json
TEST_PATH=v2_4_epoch_100
jac-crun ${GPU_ID} scripts/trainval_tube_v2.py --desc clevrer/desc_nscl_derender_clevrer_v2.py\
    --batch-size 1 --epoch 30 --validation-interval 5 \
    --save-interval 5 \
    --normalized_boxes 1 \
    --rel_box_flag 0 --acc-grad 1 --dynamic_ftr_flag  1 \
    --box_iou_for_collision_flag 1 \
    --new_mask_out_value_flag 1 \
    --colli_ftr_type 1 \
    --lr 0.0001 \
    --tube_mode 1 \
    --version v2 \
    --scene_supervision_flag 1 \
    --scene_add_supervision 0 \
    --frm_img_num 16 --even_smp_flag 1 \
    --evaluate \
    --diff_for_moving_stationary_flag 0 \
    --apply_gaussian_smooth_flag 0 \
    --data-workers 2 \
    --dataset blocks \
    --prefix ${TEST_PATH} \
    --load ${MODEL_PATH} \
    --tube_prp_path ${ANN_PATH} \
    --scene_gt_path ${ANN_PATH} \
    --data-dir ${VIDEO_PATH} \
    --question_path ${QUESTION_PATH} \
    --frm_img_path ${VIDEO_PATH} \
