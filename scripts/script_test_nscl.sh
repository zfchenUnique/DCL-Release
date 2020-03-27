GPU_ID=$1
jac-crun ${GPU_ID} scripts/trainval.py --desc experiments/clevr/desc_nscl_derender.py --training-target derender --curriculum all --dataset clevr --data-dir data/clevr/train --data-split 0.95 --extra-data-dir data/clevr/val --evaluate --load dumps/clevr/desc_nscl_derender/derender-curriculum_all-qtrans_off-clevrfull-epoch_100.pth
