import os
import glob

SPLIT_PATH = ['ucfTrainTestlist/testlist01.txt',
            'ucfTrainTestlist/testlist02.txt',
            'ucfTrainTestlist/testlist03.txt']
def read_split(path):
    with open(path, 'r') as split_file:
        split_string = split_file.read()
    split_list = split_string.split('\r\n')
    split_set = set([ os.path.basename(x).split('.')[0] for x in split_list])
    return split_set
split_set = set()
for path in SPLIT_PATH:
    split_set = split_set | read_split(path)

data_dir = ['ucf_101_img_test10_top8', 'ucf_101_img_test10_NECK', 'ucf_101_img_test10_mid35']

for dir_k in data_dir:
	glob_path = os.path.join(dir_k,'*','*')
	glob_list = glob.glob(glob_path)
	for path in glob_list:
		base_name = os.path.basename(path).split('.')[0]
		if not base_name in split_set:
			os.remove(path)