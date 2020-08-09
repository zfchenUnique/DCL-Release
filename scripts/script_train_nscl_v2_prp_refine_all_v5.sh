GPU_ID=$1
jac-crun ${GPU_ID} scripts/trainval_tube_v2.py --desc clevrer/desc_nscl_derender_clevrer_v2.py\
    --training-target v2 \
    --dataset clevrer --data-dir ../clevrer \
    --batch-size 1 --epoch 100 --validation-interval 5 \
    --save-interval 1 --data-split 0.95 --data-workers 2 \
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
    --tube_prp_path ../clevrer/tubeProposalsAttrV3/1.0_1.0_0.4_0.7 \
    --scene_add_supervision 0 \
    --pred_model_path ../temporal_reasoning-master/models_latent.py \
    --dataset_stage -1 \
    --scene_add_supervision 0 \
    --ftr_in_collision_space_flag 0 \
    --pred_spatial_model_path ../temporal_reasoning-master/models.py \
    --pred_model_path ../temporal_reasoning-master/models_latent.py \
    --rela_spatial_dim 3 \
    --relation_dim 259 \
    --state_dim 256 \
    --rela_spatial_only 1 \
    --residual_obj_pred 1 \
    --pred_frm_num 12 \
    --tube_mode 1 \
    --reconstruct_flag 0 \
    --pred_normal_num 28 \
    --pretrain_pred_spatial_model ../temporal_reasoning-master/dumps/box_only_tubeProposalAttrV3_CLEVR_noAttr_noEdgeSuperv_pn_pstep_2/net_best.pth \
    --ftr_in_collision_space_flag 0 \
    --regu_only_flag 0 \
    --regu_flag 0 \
    --version v2 \
    --freeze_learner_flag 0 \
    --prefix rgb_separate_propNet \
    --unseen_events_path /home/zfchen/code/nsclClevrer/temporal_reasoning-master/dumps/annos/tubeNetAttrV3_offset4_separate_realOffset5_noAttr_noEdgeSuperv \
    --evaluate \
    --load dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp31_col_box_ftr_v2_rgb_separate_propNet/checkpoints/epoch_2.pth \
    #--visualize_flag 1 \
    #--debug \
    #--load /home/zfchen/code/nsclClevrer/dynamicNSCL/dumps/remote_models/refine_epoch_10.pth \
    #--load dumps/remote_models/regu_colli_10.pth \
    #--unseen_events_path /home/zfchen/code/nsclClevrer/temporal_reasoning-master/dumps/attrV3_offset4_noAttr_noEdgeSuperv \
    #--load /home/zfchen/code/nsclClevrer/dynamicNSCL/dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp31_col_box_ftr_v2_attrV3/checkpoints/epoch_2.pth \
    #--load dumps/remote_models/regu_colli_10.pth \
    #--unseen_events_path /home/zfchen/code/nsclClevrer/temporal_reasoning-master/dumps/ori_offset4_noAttr_noEdgeSuperv \
    #--unseen_events_path /home/zfchen/code/nsclClevrer/temporal_reasoning-master/dumps/attrV3_offset4_noAttr_noEdgeSuperv \
    #--load dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp31_col_box_ftr_v4_prp_regu_only_12/checkpoints/epoch_5.pth \
    #--visualize_flag 0 \
    #--load /home/zfchen/code/nsclClevrer/dynamicNSCL/dumps/remote_models/refine_epoch_10.pth \
    #--pretrain_pred_spatial_model ../temporal_reasoning-master/dumps/box_only_tubeGt_v2_CLEVR_noAttr_noEdgeSuperv_pn_pstep_2/tube_net_epoch_1_iter_250000.pth \
    #--debug \
    #--load /home/zfchen/code/nsclClevrer/dynamicNSCL/dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp31_col_box_ftr_v4_prp_regu_only_decode_v5_debug/checkpoints/epoch_15.pth \
    #--visualize_flag 1 \
    #--visualize_flag 1 \
    #--evaluate \
    #--pretrain_pred_spatial_model ../temporal_reasoning-master/dumps/box_only_CLEVR_noAttr_noEdgeSuperv_pn_pstep_2_abs/net_best.pth \

    #--resume dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp31_col_box_ftr_v3_prp_all_pretrain_epoch_3_iter_30k_joint_train_fix/checkpoints/epoch_5.pth \

    #--pretrain_pred_model_path ../temporal_reasoning-master/dumps/prpNewRefine_latent_norm_ftr_n_his_2_CLEVRER_noAttr_noEdgeSuperv_pn_pstep_2_version_v3/tube_net_epoch_3_iter_300000.pth \
    #--debug 
    #--regu_flag 1 \
    #--pred_normal_num 1 \
    #--testing_flag 1 \
    #--visualize_flag 1 \
    #--pred_model_path ../temporal_reasoningv2/models_latent.py \
    #--pretrain_pred_model_path ../remote_models/ori_epoch_1_iter_200000.pth
    #--load dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp31_col_box_ftr_v2_31_no_scene_refine/checkpoints/epoch_10.pth \
    #--resume  dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp31_col_box_ftr_v2_prp_v2_new_refined/checkpoints/epoch_1.pth\
    #--load dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp31_col_box_ftr_v2_31_no_scene_refine/checkpoints/epoch_10.pth \
    #--debug \
    #--correct_question_flag 1 \
    #--tube_prp_path ../clevrer/tubeProposalsAttrV3/1.0_1.0_0.4_0.7 \

    #--load /home/zfchen/code/nsclClevrer/dynamicNSCL/dumps/remote_models/refine_epoch_10.pth \
    #--load dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp31_col_box_ftr_v2_31_no_scene_refine/checkpoints/epoch_10.pth \
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

