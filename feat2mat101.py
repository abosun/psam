import os 
import glob

DIR_FEAT = 'ucf_101_feat_maxgai_sig'
PATH_MAT = 'ucf_101_feat_maxgai_sig.txt'

glob_list = glob.glob(os.path.join(DIR_FEAT,'*','*.txt'))
data_list = []
for path in sorted(glob_list):
    with open(path) as f:
        data_list.append(f.read())
print(len(data_list))
with open(PATH_MAT, 'w') as f:
    f.write('\r\n'.join(data_list))
