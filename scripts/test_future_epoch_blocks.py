import os
import time

def run_main():
    #full_pre_path = 'dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp31_col_box_ftr_v2_rgb_v2/checkpoints/'
    #full_pre_path = 'dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp31_col_box_ftr_v2_rgb_sep_refineMatch_all_from_ep7/checkpoints/'
    #full_pre_path = 'dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp31_col_box_ftr_v2_rgb_sep_refineMatch_all_from_zero_AttrMatch/checkpoints/'
    full_pre_path =  'dumps/blocks/desc_nscl_derender_clevrer_v2_norm_box_even_smp16_col_box_ftr_v2_block/checkpoints/'
    block_path_ori = 'block_ep_'
    tar_ep = 5
    sleep = 30
    while True:
        full_path = full_pre_path + 'epoch_' +str(tar_ep) +'.pth'
        test_path = block_path_ori + str(tar_ep) 
        if os.path.isfile(full_path):
            cmd = 'sh scripts/script_eval_nscl_blocks.sh 0 %s %s' %(full_path, test_path)
            os.system(cmd)
            tar_ep +=5
        time.sleep(sleep)
        print('finding target epoch every %d second\n' %(sleep))

if __name__=='__main__':
    run_main()
