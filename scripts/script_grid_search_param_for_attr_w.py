import pdb 
import os
import numpy as np

def main():
    #for attr_w in np.arange(0.5,  3.5, 0.5):
    #for attr_w in np.arange(0.1,  1, 0.1):
    for match_thre in np.arange(0.1,  1, 0.1):
        attr_w = 0.5 #float(attr_w)
        match_thre = float(match_thre)
        #pdb.set_trace()
        cmd_text= 'python scripts/script_gen_tube_proporals.py --tube_folder_path ../clevrer/tubeProposalsAttrV2 --extract_att_path dumps/clevrer/tmpProposalsAttr --attr_w %f  --match_thre %f' % (attr_w, match_thre)
        os.system(cmd_text)

if __name__=='__main__':
    main()
    #pdb.set_trace()
