import os
import time

def run_main():
    full_pre_path = 'dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp31_col_box_ftr_v2_rgb_v2/checkpoints/'
    tar_ep = 4
    sleep = 5
    full_path = full_pre_path + 'epoch_' +str(tar_ep) +'.pth'
    while True:
        if os.path.isfile(full_path):
            cmd = 'sh scripts/script_train_nscl_v2_prp_refine_all_v5.sh 0'
            os.system(cmd)
            tar_ep +=1
        time.sleep(sleep)
        print('finding target epoch every %d second\n' %(sleep))

if __name__=='__main__':
    run_main()
