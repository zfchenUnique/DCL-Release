GPU_ID=$1
jac-crun ${GPU_ID} scripts/trainval_tube_v2.py --desc clevrer/desc_nscl_derender_clevrer_v2.py\
    --training-target v2 \
    --dataset clevrer --data-dir ../clevrer \
    --batch-size 1 --epoch 100 --validation-interval 1 \
    --save-interval 1 --data-split 0.95 --data-workers 2 \
    --normalized_boxes 1 \
    --rel_box_flag 0 --acc-grad 1 --dynamic_ftr_flag  1 \
    --scene_supervision_flag 1 \
    --scene_supervision_weight 0.5 \
    --box_iou_for_collision_flag 1 \
    --diff_for_moving_stationary_flag 1 \
    --new_mask_out_value_flag 1 \
    --apply_gaussian_smooth_flag 1 \
    --prefix prp_v3_no_refined \
    --colli_ftr_type 1 \
    --pred_model_path ../temporal_reasoningv2/models_latent.py \
    --frm_img_num 31 --even_smp_flag 1 \
    --version v3 \
    --lr 0.0001 \
    --tube_prp_path ../clevrer/tubeProposalsAttrV0/1.0_1.0_0.5 \
    --scene_add_supervision 0 \
    --load /home/zfchen/code/nsclClevrer/dynamicNSCL/dumps/remote_models/refine_epoch_10.pth \
    --pretrain_pred_model_path dumps/prpRefine_latent_norm_ftr_0001_CLEVRER_noAttr_noEdgeSuperv_pn_pstep_2_version_v3/tube_net_epoch_0_iter_400000.pth
    #--load dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp31_col_box_ftr_v2_31_no_scene_refine/checkpoints/epoch_10.pth \
    #--debug \
    #--load dumps/remote_models/frm_31_epoch_24.pth \
    #--pretrain_pred_model_path ../temporal_reasoning-master/dumps/latent_norm_ftr_CLEVRER_noAttr_noEdgeSuperv_pn_pstep_2_version_v3/tube_net_epoch_0_iter_100000.pth  
    #--pretrain_pred_model_path ../temporal_reasoning-master/dumps/latent_prp_CLEVRER_noAttr_noEdgeSuperv_pn_pstep_2_version_v3/tube_net_epoch_0_iter_400000.pth 
    #--pretrain_pred_model_path ../temporal_reasoning-master/latent_prp_CLEVRER_noAttr_noEdgeSuperv_pn_pstep_2_tubemode_1/tube_net_epoch_2_iter_700000.pth \
    #--pretrain_pred_model_path ../temporal_reasoning-master/dumps/latent_prp_CLEVRER_noAttr_noEdgeSuperv_pn_pstep_2_version_v2/tube_net_epoch_2_iter_700000.pth \
    #--pretrain_pred_model_path  ../temporal_reasoning-master/latent_prp_CLEVRER_noAttr_noEdgeSuperv_pn_pstep_2_tubemode_1/tube_net_epoch_2_iter_700000.pth \
    #--pretrain_pred_model_path ../temporal_reasoning-master/latent_prp_CLEVRER_noAttr_noEdgeSuperv_pn_pstep_2_version_v2/net_best.pth \
    #--debug
    #--debug 
    #--debug \
    #--debug \
    #--resume  dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp6_col_box_ftr_v2_new_visual_colli/checkpoints/epoch_4.pth
    #--resume dumps/remote_models/new_colli_16.pth \
    #--resume dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp6_col_box_ftr_v2_new_causal_fix_station_moving_bug/checkpoints/epoch_3.pth
    #--load dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp6_col_box_ftr_v2/checkpoints/epoch_15.pth \

