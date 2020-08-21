import os
import time

def run_main():
    #full_pre_path = 'dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp31_col_box_ftr_v2_rgb_v2/checkpoints/'
    #full_pre_path = 'dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp31_col_box_ftr_v2_rgb_sep_refineMatch_all/checkpoints/'
    full_pre_path = 'dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp31_col_box_ftr_v2_refineMatch_des_exp/checkpoints/'
    tar_ep = 11
    sleep = 60
    while True:
        full_path = full_pre_path + 'epoch_' +str(tar_ep) +'.pth'
        if os.path.isfile(full_path):
            #cmd = 'sh scripts/script_train_nscl_v2_prp_refine_all_v5.sh 0'
            cmd = 'sh scripts/script_test_future_v2_des_exp.sh 0 %s' %(full_path)
            os.system(cmd)
            tar_ep +=1
        time.sleep(sleep)
        print('finding target epoch every %d second\n' %(sleep))

if __name__=='__main__':
    run_main()
