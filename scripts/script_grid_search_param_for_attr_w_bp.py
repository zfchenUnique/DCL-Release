import pdb 
import os
import numpy as np

def main():
    for attr_w in np.arange(0.5,  3.5, 0.5):
        attr_w = float(attr_w)
        #pdb.set_trace()
        cmd_text= 'python scripts/script_gen_tube_proporals.py --tube_folder_path ../clevrer/tubeProposalsAttrV0 --extract_att_path dumps/clevrer/tmpProposalsAttr --attr_w %f' % (attr_w)
        os.system(cmd_text)

if __name__=='__main__':
    main()
    pdb.set_trace()
