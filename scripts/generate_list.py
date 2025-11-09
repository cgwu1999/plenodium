import argparse
import os
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", type=str)
args = parser.parse_args()


image_names = sorted(glob(os.path.join(args.data_dir, "images_wb")))
with open(os.path.join(args.data_dir, "train_list.txt"),"w") as train, open(os.path.join(args.data_dir, "test_list.txt"),"w") as test:
    for i, name in enumerate(image_names):
        if i%8==0:
            test.write(f"{name}\n")
        else:
            train.write(f"{name}\n")
