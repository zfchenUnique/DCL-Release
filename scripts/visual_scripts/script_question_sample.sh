GPU_ID=$1
MODEL_PATH='dumps/remote_models/attrMatchNoIoU_epoch_7_prp.pth'
TEST_PATH='attrMatchNoIoU_epoch_prp_ep7'
jac-crun ${GPU_ID} scripts/trainval_tube_v2.py --desc clevrer/desc_nscl_derender_clevrer_v2.py\
    --dataset clevrer --data-dir ../clevrer \
    --epoch 100 --validation-interval 5 \
    --save-interval 1 \
    --normalized_boxes 1 \
    --rel_box_flag 0 --acc-grad 1 --dynamic_ftr_flag  1 \
    --scene_supervision_flag 1 \
    --scene_supervision_weight 0.5 \
    --box_iou_for_collision_flag 1 \
    --diff_for_moving_stationary_flag 1 \
    --new_mask_out_value_flag 1 \
    --apply_gaussian_smooth_flag 1 \
    --colli_ftr_type 1 \
    --frm_img_num 31 --even_smp_flag 1 \
    --lr 0.0001 \
    --tube_mode 1 \
    --scene_add_supervision 0 \
    --version v2 \
    --background_path ../temporal_reasoning-master/background.png \
    --prefix new_collision_val \
    --unseen_events_path ../temporal_reasoning-master/dumps/annos/tubeNetAttrV3_offset4_noIoUThre_separate_realOffset5_noAttr_noEdgeSuperv \
    --tube_prp_path ../clevrer/tubeProposalsAttrMatchNoIoUThre/1.0_1.0_0.6_0.7 \
    --data-workers 0 \
    --batch-size 1 \
    --test_result_path ${TEST_PATH} \
    --load ${MODEL_PATH} \
    --prefix qa_clevrer \
    --evaluate \
    --visualize_flag 1 \
    --visualize_gif_flag 1 \
    --dataset_stage -1 \
    --debug \
