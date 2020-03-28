GPU_ID=$1
jac-crun ${GPU_ID} scripts/trainval_tube.py --desc clevrer/desc_nscl_derender_clevrer.py\
    --training-target derender --curriculum all\
    --dataset clevrer --data-dir ../clevrer\
    --batch-size 8 --epoch 100 --validation-interval 5\
    --save-interval 5 --data-split 0.95 --data-workers 1 \
    --normalized_boxes 1
