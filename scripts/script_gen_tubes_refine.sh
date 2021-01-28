att_path='./dumps/clevrer/tmpProposalsAttr'
python scripts/script_gen_tube_proposals.py \
    --start_index 10000 \
    --end_index 15000 \
    --attr_w 0.6 \
    --match_thre 0.7 \
    --version 2 \
    --visualize_flag 0 \
    --tube_folder_path ../clevrer/tubeProposalsReleaseAttr \
    --use_attr_flag 1 \
    --refine_tube_flag 1 \
    --extract_att_path ${att_path} \
