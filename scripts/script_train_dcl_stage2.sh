GPU_ID=$1
tube_folder_path='../clevrer/tubeProposalsRelease'
unseen_events_pat='../temporal_reasoningv2/dumps/annos/tubeNetAttrV3_offset4_noIoUThre_separate_realOffset'
jac-crun ${GPU_ID} scripts/trainval_tube_v2.py --desc clevrer/desc_nscl_derender_clevrer_v2.py\
    --dataset clevrer --data-dir ../clevrer \
    --save-interval 1 --data-split 0.95 --data-workers 2 \
    --normalized_boxes 1 \
    --rel_box_flag 0 --acc-grad 4 --dynamic_ftr_flag  1 \
    --box_iou_for_collision_flag 1 \
    --diff_for_moving_stationary_flag 1 \
    --new_mask_out_value_flag 1 \
    --apply_gaussian_smooth_flag 1 \
    --colli_ftr_type 1 \
    --frm_img_num 31 --even_smp_flag 1 \
    --version v2 \
    --lr 0.001 \
    --tube_prp_path ${tube_folder_path} \
    --scene_add_supervision 0 \
    --correct_question_flag 1 \
    --scene_supervision_flag 1 \
    --batch-size 1 --epoch 100 --validation-interval 5 \
    --prefix prp_dataset_stage \
    --dataset_stage -1 \
    --unseen_events_path ${unseen_events_path} \
