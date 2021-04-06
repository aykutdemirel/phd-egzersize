import os
import pandas as pd
import shutil
import multiprocessing
from joblib import Parallel, delayed

train_datas = pd.read_csv('data_csv/jester-v1-train.csv', header=None, sep=";", index_col=0, squeeze=True).to_dict()
validation_datas = pd.read_csv('data_csv/jester-v1-validation.csv', header=None, sep=";", index_col=0, squeeze=True).to_dict()

main_path = "E:/MyWorks/Works/20BN-JESTER/"
train_path = "training_samples/"
validation_path = "validation_samples/"
#pool = Pool(6)

def recursive_overwrite(src, dest, ignore=None):
    if os.path.isdir(src):
        if not os.path.isdir(dest):
            os.makedirs(dest)
        files = os.listdir(src)
        if ignore is not None:
            ignored = ignore(src, files)
        else:
            ignored = set()
        for f in files:
            if f not in ignored:
                recursive_overwrite(os.path.join(src, f), os.path.join(dest, f), ignore)
    else:
        shutil.copyfile(src, dest)

def run_image_process(data):
    print(str(data))
    recursive_overwrite(main_path + str(data), validation_path + str(data))

num_cores = multiprocessing.cpu_count()

print("number of cores = " + str(num_cores))

element_run = Parallel(n_jobs=num_cores)(delayed(run_image_process)(data) for data in validation_datas)

dirs = os.listdir(train_path)
dirs_cv = os.listdir(validation_path)

print("training samples size = " + str(len(dirs)))
print("validation samples size = " + str(len(dirs_cv)))
