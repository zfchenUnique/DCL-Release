import os
import time

def run_main():
    #full_pre_path = 'dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp31_col_box_ftr_v2_rgb_v2/checkpoints/'
    #full_pre_path = 'dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp31_col_box_ftr_v2_rgb_sep_refineMatch_all_from_ep7/checkpoints/'
    #full_pre_path = 'dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp31_col_box_ftr_v2_rgb_sep_refineMatch_all_from_zero_AttrMatch/checkpoints/'
    full_pre_path = 'dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp31_col_box_ftr_v2_rgb_sep_refineMatch_all_from_zero_AttrMatchNoIoU/checkpoints/'
    test_path_ori = 'dumps/testing_results/test_noIoU_ep'
    tar_ep = 6
    sleep = 120
    while True:
        full_path = full_pre_path + 'epoch_' +str(tar_ep) +'.pth'
        test_path = test_path_ori + str(tar_ep) + '.json'
        if os.path.isfile(full_path):
            #cmd = 'sh scripts/script_train_nscl_v2_prp_refine_all_v5.sh 0'
            #cmd = 'sh scripts/script_test_future_v5_1.sh 0 %s' %(full_path)
            cmd = 'sh scripts/script_test_future_v5_1.sh 0 %s %s' %(full_path, test_path_ori)
            os.system(cmd)
            tar_ep +=1
        time.sleep(sleep)
        print('finding target epoch every %d second\n' %(sleep))

if __name__=='__main__':
    run_main()
