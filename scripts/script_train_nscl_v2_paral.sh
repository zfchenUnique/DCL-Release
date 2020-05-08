GPU_ID=$1
jac-crun ${GPU_ID} scripts/trainval_tube_v2_paral.py --desc clevrer/desc_nscl_derender_clevrer_v2.py\
    --training-target v2 \
    --dataset clevrer --data-dir ../clevrer\
    --batch-size 1 --epoch 100 --validation-interval 1\
    --save-interval 1 --data-workers 4 \
    --normalized_boxes 1 \
    --frm_img_num 31 --even_smp_flag 1\
    --rel_box_flag 0 --acc-grad 1 --dynamic_ftr_flag 1 --version v3\
    --scene_supervision_flag 1 \
    --tube_prp_path ../clevrer/tubeProposalsGt \
    --scene_add_supervision 1 \
    --scene_supervision_weight 0.5 \
    --box_iou_for_collision_flag 1 \
    --diff_for_moving_stationary_flag 1 \
    --new_mask_out_value_flag 1 \
    --apply_gaussian_smooth_flag 1 \
    --prefix counterfact_v1 \
    --colli_ftr_type 1 \
    --load dumps/remote_models/frm_31_epoch_24.pth \
    --pred_model_path ../temporal_reasoning-master/models_latent.py \
    --pretrain_pred_model_path  ../temporal_reasoning-master/latent_prp_CLEVRER_noAttr_noEdgeSuperv_pn_pstep_2_tubemode_1/tube_net_epoch_0_iter_500000.pth 
    #--resume  dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp6_col_box_ftr_v2_new_visual_colli/checkpoints/epoch_4.pth
    #--resume dumps/remote_models/new_colli_16.pth \
    #--resume dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp6_col_box_ftr_v2_new_causal_fix_station_moving_bug/checkpoints/epoch_3.pth
    #--load dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp6_col_box_ftr_v2/checkpoints/epoch_15.pth \

