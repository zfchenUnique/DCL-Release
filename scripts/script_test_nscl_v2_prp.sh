GPU_ID=$1
jac-crun ${GPU_ID} scripts/trainval_tube_v2.py --desc clevrer/desc_nscl_derender_clevrer_v2.py\
    --training-target derender --curriculum all\
    --dataset clevrer --data-dir ../clevrer\
    --batch-size 1 --epoch 100 --validation-interval 5\
    --save-interval 5 --data-split 0.95 --data-workers 2 \
    --normalized_boxes 1 --frm_img_num 31 --even_smp_flag 1\
    --rel_box_flag 0 --evaluate \
    --dynamic_ftr_flag 1  \
    --scene_supervision_flag 1 \
    --tube_prp_path ../clevrer/tubeProposalsTest/1.0_1.0_0.4_0.7 \
    --box_only_for_collision_flag 0 \
    --scene_supervision_flag 1 \
    --box_iou_for_collision_flag 1 \
    --diff_for_moving_stationary_flag 1 \
    --new_mask_out_value_flag 1 \
    --apply_gaussian_smooth_flag 1 \
    --colli_ftr_type 1 \
    --version v3 \
    --pred_model_path ../temporal_reasoning-master/models_latent.py \
    --colli_threshold 0 \
    --test_result_path 'test_ep0.json' \
    --testing_flag 1 \
    --testing_flag 1 \
    --visualize_flag 1 \
    --regu_flag 1 \
    --regu_only_flag 1 \
    --debug \
    --resume dumps/remote_models/freeze_ep4.pth \
    --pred_normal_num 25 \
    --residual_rela_prop 1 \
    --residual_rela_pred 1 \
    --rela_spatial_only 1
    #--resume dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp31_col_box_ftr_v3_prp_all_pretrain_epoch_3_iter_30k_joint_train_fix_resume/checkpoints/epoch_11.pth

    #--pretrain_pred_model_path ../temporal_reasoning-master/dumps/prpNewRefine_latent_norm_ftr_n_his_2_CLEVRER_noAttr_noEdgeSuperv_pn_pstep_2_version_v3/tube_net_epoch_3_iter_300000.pth \
    #--load /home/zfchen/code/nsclClevrer/dynamicNSCL/dumps/remote_models/refine_epoch_10.pth \
    #--resume dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp31_col_box_ftr_v3_prp_all_pretrain_epoch_3_iter_30k_joint_train_fix_resume/checkpoints/epoch_6.pth
    #--visualize_flag 1 \
    #--pretrain_pred_model_path ../temporal_reasoning-master/dumps/prpNewRefine_latent_norm_ftr_n_his_2_CLEVRER_noAttr_noEdgeSuperv_pn_pstep_2_version_v3/tube_net_epoch_3_iter_300000.pth \
    #--load /home/zfchen/code/nsclClevrer/dynamicNSCL/dumps/remote_models/refine_epoch_10.pth \
    #--pred_normal_num 1 \
    #--debug
    #--resume dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp31_col_box_ftr_v3_prp_all_pretrain_epoch_3_iter_30k_joint_train_fix/checkpoints/epoch_5.pth
    #--pretrain_pred_model_path ../temporal_reasoning-master/dumps/latent_prp_CLEVRER_noAttr_noEdgeSuperv_pn_pstep_2_version_v3/net_best.pth  \
    #--load dumps/remote_models/refine_full_12.pth \
    #--testing_flag 1 \
    #--visualize_flag 1 \
    #--load dumps/remote_models/frm_31_epoch_24.pth \
    #--pretrain_pred_model_path ../temporal_reasoning-master/dumps/latent_prp_CLEVRER_noAttr_noEdgeSuperv_pn_pstep_2_version_v3/net_best.pth  \
    #--debug \
    #--tube_prp_path ../clevrer/tubeProposalsAttrV3/1.0_1.0_0.4_0.7/ \
    #--pretrain_pred_model_path ../temporal_reasoning-master/latent_prp_CLEVRER_noAttr_noEdgeSuperv_pn_pstep_2_tubemode_1/tube_net_epoch_2_iter_700000.pth \
    #--pretrain_pred_model_path ../temporal_reasoning-master/dumps/latent_prp_CLEVRER_noAttr_noEdgeSuperv_pn_pstep_2_version_v2/tube_net_epoch_2_iter_700000.pth \
    #--debug \
    #--load dumps/remote_models/new_colli_16.pth
    #--load dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp6_col_box_ftr_v2_new_causal_fix_station_moving_bug/checkpoints/epoch_8.pth
    #--load dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp6_col_box_ftr_v2_new_causal_fix_station_moving_bug/checkpoints/epoch_3.pth
    #--load dumps/remote_models/causal_colli_8.pth \
    #--load dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp6_col_box_ftr_v2_fix_station_moving_bug/checkpoints/epoch_4.pth
    #--load dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp6_col_box_ftr_v2_causal_fix_station_moving_bug/checkpoints/epoch_4.pth
    #--debug \
    #--box_iou_for_collision_flag 1 \
    #--diff_for_moving_stationary_flag 1 \
    #--new_mask_out_value_flag 1 \
    #--apply_gaussian_smooth_flag 1 \
    #--load dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp6_col_box_ftr_v2_causal_fix_station_moving_bug/checkpoints/epoch_4.pth 
    #--load dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp6_col_box_ftr_v2/checkpoints/epoch_15.pth \
    #--load dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp6_col_box_ftr_v2_fix_station_moving_bug/checkpoints/epoch_8.pth
    #--load dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp6_col_box_ftr_v2_causal/checkpoints/epoch_4.pth \
    #--debug
    #--load dumps/clevrer/desc_nscl_derender_clevrer/derender_norm_box_even_smp6_col_box_ftr_v5_gt_new_col_scene_super_05/checkpoints/epoch_15.pth
    #--load dumps/clevrer/desc_nscl_derender_clevrer/derender_norm_box_even_smp6_col_box_ftr_v5_gt_sym_col_scene_supervision_balance/checkpoints/epoch_10.pth
    #--load dumps/clevrer/desc_nscl_derender_clevrer/derender_norm_box_even_smp6_col_box_ftr_v5_gt/checkpoints/epoch_16.pth \
    #--load dumps/clevrer/desc_nscl_derender_clevrer/derender_norm_box_even_smp6_col_box_ftr_v5_gt/epoch_16.pth --debug \

