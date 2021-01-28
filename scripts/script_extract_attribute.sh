GPU_ID=$1
PARSE_PRG_PATH=../language_parsing/data/new_results
#MODEL_PATH=data/models/DCL_model/epoch_7.pth
#COARSE_PRP_PATH=../clevrer/tubeProposals/1.0_1.0 
jac-crun ${GPU_ID} scripts/script_extract_attribute_for_frames.py --desc clevrer/desc_nscl_derender_clevrer_v2.py \
    --training-target derender --curriculum all \
    --dataset clevrer --data-dir ../clevrer \
    --batch-size 8 --epoch 100 --validation-interval 5 \
    --save-interval 5 --data-split 0.95 --data-workers 0 \
    --normalized_boxes 1 --frm_img_num 31 --even_smp_flag 1 \
    --rel_box_flag 0 --evaluate \
    --dynamic_ftr_flag 1 \
    --scene_supervision_flag 1 \
    --box_only_for_collision_flag 0 \
    --scene_supervision_flag 1 \
    --box_iou_for_collision_flag 1 \
    --diff_for_moving_stationary_flag 1 \
    --version v2  \
    --extract_region_attr_flag 1 \
    --setname train \
    --start_index 0 \
    --output_attr_path dumps/clevrer/tmpProposalsAttrTestRefineMatch \
    --load ${MODEL_PATH} \
    --tube_prp_path ${COARSE_PRP_PATH} \
    --correct_question_path ${PARSE_PRG_PATH} \
