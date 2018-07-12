from os.path import exists, join, basename, isfile
from os import makedirs, remove, listdir
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from PIL import Image
from dataset import DatasetFromFolder
import numpy as np


def download_bsd300(dest="dataset"):
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        makedirs(dest)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)
        
    patch_h = 64
    patch_w = 64
    k = 0
    cur_dir = join(output_image_dir, 'train')
    dest_dir = join(output_image_dir, 'train1')
    
    if not exists(dest_dir):
        
        makedirs(dest_dir)
        for entry in listdir(cur_dir):
            filename = join(cur_dir, entry)
            if isfile(filename):
                img = Image.open(filename)
                rect = np.array(img)


                x = 0
                y = 0

                while(y + patch_h <= img.width):
                    x = 0
                    while(x + patch_w <= img.height):
                        patch = rect[x : x + patch_h, y : y + patch_w]
                        img_hr = Image.fromarray(patch, 'RGB')

#                        img_lr = img_hr.resize((patch_w // 2, patch_h // 2), Image.ANTIALIAS)
#                        img_lr = img_lr.resize((patch_w, patch_h), Image.BICUBIC)

                        out_hr = join(dest_dir, str(k) + "_hr.png")
#                        out_lr = join(dest_dir, str(k) + "_lr.png")

                        k = k + 1

                        img_hr.save(out_hr)
#                        img_lr.save(out_lr)

                        x = x + 42
                    y = y + 42

    return output_image_dir


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


def get_training_set(upscale_factor):
    root_dir = download_bsd300()
    train_dir = join(root_dir, "train1")
    crop_size = calculate_valid_crop_size(64, upscale_factor)

    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))


def get_test_set(upscale_factor):
    root_dir = download_bsd300()
    test_dir = join(root_dir, "test")
    crop_size = calculate_valid_crop_size(64, upscale_factor)

    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))
