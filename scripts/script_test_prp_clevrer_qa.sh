GPU_ID=$1
PREFIX='rgb_sep_prp_ep7'
TEST_RESULT_PATH='dumps/test_ep7_release_raw.json'
MODEL_PATH='data/models/DCL_model/epoch_7.pth'
TUBE_PRP_PATH='data/raw_data/tubeProposalsAttrMatchTest/1.0_1.0_0.6_0.7'
UNSEEN_EVENTS_PATH='./submodules/clevrer_dynamic_propnet/dumps/annos/val_release_v2_separate_realOffset5_noAttr_noEdgeSuperv'
CORRECT_QUESTION_PATH='./data/raw_data/results_v2'
jac-crun ${GPU_ID} scripts/trainval_tube_v2.py --desc clevrer/desc_nscl_derender_clevrer_v2.py\
    --dataset clevrer --data-dir ../clevrer \
    --batch-size 1 --epoch 100 --validation-interval 5 \
    --save-interval 1 --data-split 0.95 \
    --data-workers 4 \
    --normalized_boxes 1 \
    --rel_box_flag 0 --acc-grad 1 --dynamic_ftr_flag  1 \
    --box_iou_for_collision_flag 1 \
    --diff_for_moving_stationary_flag 1 \
    --new_mask_out_value_flag 1 \
    --apply_gaussian_smooth_flag 1 \
    --colli_ftr_type 1 \
    --frm_img_num 31 --even_smp_flag 1 \
    --lr 0.0001 \
    --tube_mode 1 \
    --scene_add_supervision 0 \
    --scene_supervision_flag 1 \
    --scene_supervision_weight 0.5 \
    --version v2 \
    --background_path _assets/background.png \
    --evaluate \
    --testing_flag 1 \
    --load ${MODEL_PATH} \
    --prefix ${PREFIX} \
    --test_result_path ${TEST_RESULT_PATH} \
    --tube_prp_path ${TUBE_PRP_PATH} \
    --unseen_events_path ${UNSEEN_EVENTS_PATH} \
    --correct_question_path ${CORRECT_QUESTION_PATH} \
