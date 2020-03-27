GPU_ID=$1
jac-crun ${GPU_ID} scripts/trainval.py --desc experiments/clevr/desc_nscl_derender.py --training-target derender --curriculum all --dataset clevr --data-dir data/clevr/train --batch-size 16  --epoch 100 --validation-interval 5 --save-interval 5 --data-split 0.95
