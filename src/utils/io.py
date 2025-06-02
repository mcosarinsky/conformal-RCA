import os
import csv
import cv2
import pathlib
import numpy as np
#import SimpleITK as sitk
import itk
import json
from PIL import Image
from json import JSONEncoder
from itertools import chain

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def to_json(data, file_path):
    # Convert data to JSON format
    json_data = json.dumps(data, cls=NumpyArrayEncoder, indent=4)

    # Save JSON data to file
    with open(file_path, 'w') as file:
        file.write(json_data)

def read_json(file_path):
    with open(file_path, 'r') as file:
        # Load JSON data into a Python dictionary
        data = json.load(file)
    
    return data

def process_results(file_path, metric='Dice'):
    eval_results = read_json(file_path)
    supported_metrics = ['Dice', 'Hausdorff', 'ASSD']
    res = {'Real':[], 'Predicted': []}

    if metric not in supported_metrics:
        raise ValueError(f"Metric must be one of {supported_metrics}")

    for sample in eval_results:
        res['Real'].append(sample['Real score'][metric])
        res['Predicted'].append(sample['RCA score'][metric])

    return res

def read_dir(dir_path):
    """ Read list of files in directory sorted by name. """
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
    sort_files = sorted(files)
    return sort_files


def read_txt(file_path):
    """ Read list of files in txt file. """
    with open(file_path, 'r') as f:
        files = [line.strip() for line in f]
        return files


def write_csv(data, header, file_path):
    """ Write data into csv file. """
    with open(file_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)


def read_points_to_numpy(file_path):
    """ Read SimpleElastix point set file into numpy array. """
    input_points = np.genfromtxt(file_path, skip_header=2)
    result_points = input_points.flatten()
    return result_points


def read_image_to_numpy(file_path):
    """ Read image file into numpy array. """
    image = read_image_to_itk(file_path)
    image = itk.GetArrayFromImage(image)
    return image


def read_image_to_itk(file_path):
    """ Read image file into ITK image. """
    image = itk.ReadImage(file_path, itk.itkFloat32)
    return image


def write_image(image, file_path, rescale=False):
    """ Write ITK image into file. """
    output_image = itk.Image(image)
    if rescale:
        output_image = itk.RescaleIntensity(output_image, 0, 255)
    output_image = itk.Cast(output_image, itk.itkUInt8)
    itk.WriteImage(output_image, file_path)

def load_train_folder(path: pathlib.Path):
    img_dir = path / 'images'
    mask_dir = path / 'masks'
    
    paths = []
    extensions = ['*.png', '*.bmp', '*.jpg', '*.tiff']
    img_files = sorted(chain.from_iterable(img_dir.rglob(ext) for ext in extensions))
    
    for img_path in img_files:
        relative_path = img_path.relative_to(img_dir)  # Get relative path from img_dir
        name = str(relative_path.with_suffix('')) 
        mask_path = next(mask_dir.glob(f"{name}*"))
        paths.append((img_path, mask_path))
        
    return paths

def load_test_folder(path: pathlib.Path, dataset: str):
    seg_dir = path / 'segs'
    img_dir = path / 'images'
    label_dir = path / 'masks'

    data = []
    
    for seg_file in sorted(seg_dir.glob("*.png")):
        if 'ISIC' in dataset:
            idx = seg_file.name.split('_')
            idx = idx[0] + '_' + idx[1]
        elif '3D-IRCADB' in dataset:
            idx = seg_file.name.split('_')
            idx = idx[0] + '/' + idx[1] + '_' + idx[2]
        else:
            idx = seg_file.name.split('_')[0]
        
        # Construct paths for image, label, and segmentation
        img_file = next(img_dir.glob(f"{idx}*"))
        label_file = next(label_dir.glob(f"{idx}*"))

        data.append((img_file, label_file, seg_file))

    return data

def save_segmentation(seg, file_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    seg_dir = os.path.join(output_dir, 'segs')
    if not os.path.exists(seg_dir):
        os.makedirs(seg_dir)

    seg_path = os.path.join(seg_dir, file_name)

    seg = (seg.squeeze().numpy()).astype(np.uint8)
    Image.fromarray(seg, mode='L').save(seg_path)

def save_images(images, temp_img_dir='temp_img_dir'):
    # Clean directory
    if os.path.exists(temp_img_dir):
        for filename in os.listdir(temp_img_dir):
            file_path = os.path.join(temp_img_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(temp_img_dir)

    # Save images
    for i, img in enumerate(images):
        if img.ndim == 2:
            img_rgb = np.stack([img] * 3, axis=-1).astype(np.uint8)
        elif img.ndim == 3 and img.shape[-1] == 3:
             img_rgb = img.astype(np.uint8)
        else:
             raise ValueError(f"Unsupported image format with shape {img.shape}")
        img_path = os.path.join(temp_img_dir, f'{i}.png') 
        cv2.imwrite(img_path, img_rgb)