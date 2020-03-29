GPU_ID=$1
jac-crun ${GPU_ID} scripts/trainval_tube.py --desc clevrer/desc_nscl_derender_clevrer.py\
    --training-target derender --curriculum all\
    --dataset clevrer --data-dir ../clevrer\
    --batch-size 4 --epoch 100 --validation-interval 5\
    --save-interval 5 --data-split 0.95 --data-workers 2 \
    --normalized_boxes 1 --frm_img_num 12 --even_smp_flag 1

