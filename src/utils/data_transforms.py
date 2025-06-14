from PIL import ImageOps, Image
from torchvision import transforms
from typing import Union
import random
import torch
import torch.nn.functional as F
import tempfile
import shutil
import pathlib 
import torch
import os
import zipfile
import numpy as np


def add_padding(img):
    desired_size = max(img.size)
    if img.size[0] == desired_size and img.size[1] == desired_size:
        return img
    delta_width = desired_size - img.size[0]
    delta_height = desired_size - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

def process_img(path: pathlib.Path, size: int, is_seg: bool = False, grayscale: bool = True):
    img = Image.open(path)
    resample_method = Image.NEAREST if is_seg else Image.BILINEAR
    
    # Resize maintaining aspect ratio
    img = add_padding(img)
    if img.size[0] != size or img.size[1] != size:
        img = img.resize((size, size), resample=resample_method) 
 
    if grayscale or is_seg: 
        img = img.convert('L')  # Convert image to grayscale
    else:
        img = img.convert('RGB')
    img_np = np.array(img)
    
    return img_np

class Scale:
    def __call__(self, sample):
        img = sample['image']

        # Normalize image
        mean = np.mean(img)
        std = np.std(img)
        if std > 0:  # To avoid division by zero
            z_normalized_img = (img - mean) / std
            scaled_img = (z_normalized_img - z_normalized_img.min()) / (z_normalized_img.max() - z_normalized_img.min()) * 255
            scaled_img = np.clip(scaled_img, 0, 255).astype(np.uint8)
        else:
            scaled_img = img
        
        sample['image'] = scaled_img
        return sample

class ToTensor:
    def __call__(self, sample):
        to_tensor = transforms.ToTensor()
        
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                sample[key] = to_tensor(value)
               
                # Scale back the masks
                if key != 'image':
                    sample[key] *= 255
        
        return sample    

class OneHot:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        
    def __call__(self, sample):
        for key, value in sample.items():
            if isinstance(value, torch.Tensor) and key != 'image':
                # Process PyTorch tensors
                if self.n_classes == 1:
                    tensor_mask_onehot = value / value.max()
                else:
                    tensor_mask_onehot = F.one_hot(value.squeeze(0).long(), num_classes=self.n_classes + 1)
                    tensor_mask_onehot = tensor_mask_onehot.float().permute(2, 0, 1)
                sample[key] = tensor_mask_onehot
            
            elif isinstance(value, np.ndarray) and key != 'image':
                # Process NumPy arrays
                if self.n_classes == 1:
                    np_mask_onehot = value / value.max()
                else:
                    # Ensure value is of integer type for one-hot encoding
                    value = value.astype(np.int64)
                    # Create one-hot encoded array
                    np_mask_onehot = np.eye(self.n_classes + 1)[value]
                    # Transpose to match (channels, height, width) format
                    np_mask_onehot = np.transpose(np_mask_onehot, (2, 0, 1))
                    # Maintain original dtype (e.g., np.uint8)
                    np_mask_onehot = np_mask_onehot.astype(np.uint8)
                sample[key] = np_mask_onehot
        return sample
    
class HUScale:
    def __init__(self, min_quantile=0.05, max_quantile=0.95):
        self.min_quantile = min_quantile
        self.max_quantile = max_quantile

    def __call__(self, sample):
        img = sample['image']
        
        # Calculate the quantiles and clip the image
        hu_min = np.quantile(img, self.min_quantile)
        hu_max = np.quantile(img, self.max_quantile)
        img_clipped = np.clip(img, hu_min, hu_max)
        scaled_image = (img_clipped - hu_min) / (hu_max - hu_min) * 255.0
        
        # Update the sample with scaled images
        sample['image'] = scaled_image.astype(np.uint8)

        return sample


def compute_ece_for_segmentation(prob_maps, true_labels, n_bins=15):
    """
    Compute the Expected Calibration Error for a single batch in segmentation.
    Args:
        prob_maps (torch.Tensor): Probability maps from the model (shape: [batch_size, n_classes, H, W]).
        true_labels (torch.Tensor): Ground truth labels (shape: [batch_size, H, W]).
        n_bins (int): Number of bins to use for calibration error computation.

    Returns:
        float: ECE for the batch.
    """
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Get mask of probabilities within this bin
        in_bin = (prob_maps > bin_lower) & (prob_maps <= bin_upper)
        prop_in_bin = in_bin.float().mean()  # Proportion of pixels in this bin

        if prop_in_bin.item() > 0:
            # Average accuracy in this bin
            accuracy_in_bin = (in_bin & (true_labels == 1)).float().sum() / in_bin.sum()

            # Average confidence in this bin
            avg_confidence_in_bin = prob_maps[in_bin].mean()

            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def convert_to_onehot(
    labels: torch.Tensor, num_classes: int, channel_dim: int = 1
) -> torch.Tensor:
    out = F.one_hot(labels.long(), num_classes)
    out = out.unsqueeze(channel_dim).transpose(channel_dim, out.dim())
    return out.squeeze(-1)


def find_onehot_dimension(T: torch.Tensor) -> Union[int, None]:
    if float(T.max()) <= 1 and float(T.min()) >= 0:
        for d in range(T.dim()):
            if torch.all((T.sum(dim=d) - 1) ** 2 <= 1e-5):
                return d

    # Otherwise most likely not a one-hot tensor
    return None

def harden_softmax_outputs(T: torch.Tensor, dim: int) -> torch.Tensor:
    num_classes = T.shape[dim]
    out = torch.argmax(T, dim=dim)
    return convert_to_onehot(out, num_classes=num_classes, channel_dim=dim)


def unzip_segs(zip_path: pathlib.Path, dest_dir_name: str):
    base_dir = tempfile.gettempdir()  # Use the system's temp directory
    dest_dir = pathlib.Path(base_dir) / dest_dir_name

    # Check if the directory already exists
    if not dest_dir.exists():
        # Ensure the parent directory exists
        dest_dir.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(dir=base_dir) as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            os.rename(temp_path, dest_dir)
            
            # Extract zip into dest_dir
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)
                
            subfolder = dest_dir / dest_dir.name
            if subfolder.is_dir():
                temp_folder = dest_dir.with_name(dest_dir.name + "_tmp")
                dest_dir.rename(temp_folder)
                    
                for item in temp_folder.iterdir():
                    if item.is_dir():
                        shutil.move(str(item), str(dest_dir))
                shutil.rmtree(temp_folder)
    return dest_dir


def extract_path(zip_path):
    dest_folder_name = zip_path.stem
    base_dir = tempfile.gettempdir()  # Use the system's temp directory
    dest_dir = pathlib.Path(base_dir) / dest_folder_name
    
    # Create destination directory if it doesn't exist
    if not dest_dir.exists():
        with tempfile.TemporaryDirectory(dir=base_dir) as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            os.rename(temp_path, dest_dir)
    
            # Extract the ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)
    
            subfolder = dest_dir / dest_folder_name
            if subfolder.is_dir():
                temp_folder = dest_dir.with_name(dest_folder_name + "_tmp")
                dest_dir.rename(temp_folder)
                    
                for item in temp_folder.iterdir():
                    if item.is_dir():
                        shutil.move(str(item), str(dest_dir))
                shutil.rmtree(temp_folder)

    return dest_dir
    
    
def load_from_colab(path):
    data = {'images': [], 'segs': []}

    zip_path = path.with_suffix('.zip')
    dest_folder = pathlib.Path('../') / path.stem

    # Extract the ZIP file if it wasn't already extracted 
    if not dest_folder.exists():
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('../')
            
    return dest_folder
