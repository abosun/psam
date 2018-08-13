import os 
import glob

DIR_FEAT = 'HMDB51_feat_1568_test'
PATH_MAT = DIR_FEAT+'.txt'

glob_list = glob.glob(os.path.join(DIR_FEAT,'*','*.cnnfeat'))
data_list = []
for path in sorted(glob_list):
    with open(path) as f:
        data_list.append(f.read())
print(len(data_list))
with open(PATH_MAT, 'w') as f:
    f.write('\r\n'.join(data_list))
