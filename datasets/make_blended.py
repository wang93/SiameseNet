import os
import shutil
root = '/data/usersdata/wyc-datasets/Person'
blended_path = os.path.join(root,'blended')
if os.path.exists(blended_path):
    shutil.rmtree(blended_path)
os.makedirs(blended_path)

datasets = ['cuhk-detect', 'dukemtmc', 'market1501']
subfolders= ['bounding_box_test', 'bounding_box_train', 'query']
for i, data in enumerate(datasets):
    i_str = str(i)
    for subfolder in subfolders:
        source_dir = os.path.join(root, data, subfolder)
        target_dir = os.path.join(blended_path, subfolder)

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        for im in os.listdir(source_dir):
            if im[0] == '-':
                continue

            im_new = i_str + im

            source_path = os.path.join(source_dir, im)
            target_path = os.path.join(target_dir, im_new)
            #shutil.copyfile(source_path, target_path)
            os.symlink(source_path, target_path)

