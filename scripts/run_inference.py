import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sam2_dir = os.path.join(parent_dir, 'sam2')
sys.path.extend([parent_dir, sam2_dir])

import numpy as np
import torchvision 
import argparse
import json

from torch.utils.data import Subset
from src.utils.io import to_json
from src.metrics import compute_scores, sample_N, sample_balanced
from src.rca import RCA
from src.utils.data_transforms import ToTensor, OneHot, Scale, HUScale
from src.datasets import *

# Uncomment to avoid OpenMP library conflicts
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

def main():
    np.random.seed(1)

    parser = argparse.ArgumentParser(description="Run RCA inference with given arguments")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--classifier', type=str, help='Classifier used to run inference', required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--n_test', type=int, help='Number of images in reference dataset', required=False, default=8)
    parser.add_argument('--emb_model', type=str, help='Model used to compute image embeddings', required=False)
    parser.add_argument('--config', type=str, help='SAM 2 model config', required=False, default='sam2_hiera_t')

    args = parser.parse_args()

    # Supported datasets
    datasets = ['hc18', 'psfhs', 'scd', 'jsrt', 'ph2', 'isic 2018', '3d-ircadb/liver', 'nucls', 'wbc/cv', 'wbc/jtsc']
    if args.dataset not in datasets:
        raise ValueError(f"Dataset '{args.dataset}' is not recognized, must be one of {datasets}")
    
    target_size = 128 if '3d-ircadb' in args.dataset else 256

    # Set the default value for emb_model based on the dataset
    if args.dataset == 'jsrt':
        default_emb_model = 'microsoft/rad-dino'
    else:
        default_emb_model = 'facebook/dinov2-base'

    parser.set_defaults(emb_model=default_emb_model)
    args = parser.parse_args()

    if args.emb_model == 'None':
        args.emb_model = None

    # Define number of classes to segment based on dataset
    if args.dataset in ['psfhs', 'jsrt', 'wbc/cv', 'wbc/jtsc']:
        n_classes = 2
    else:
        n_classes = 1
    
    # Define transforms to apply
    if args.dataset == 'jsrt':
        transforms_list = [chestxray.Rescale(256), chestxray.ToTensorSeg()]        
        if args.classifier != 'universeg':
            transforms_list.append(chestxray.ToNumpy())
        if args.classifier != 'atlas':
            transforms_list.append(OneHot(n_classes=n_classes))
        transforms = torchvision.transforms.Compose(transforms_list)
    else:
        transforms_list = []

        if args.classifier == 'universeg':
            transforms_list.extend([ToTensor(), OneHot(n_classes=n_classes)])
        elif args.classifier == 'sam2':
            transforms_list.append(OneHot(n_classes=n_classes))
        
        transforms = torchvision.transforms.Compose(transforms_list) if transforms_list else None

    if args.dataset == 'jsrt':
        d_reference, d_test, d_cal = chestxray.get_jsrt_datasets(transforms)
    else:
        grayscale = True 
        d_reference = Seg2D_Dataset(split='Train', dataset=args.dataset, transform=transforms, grayscale=grayscale, target_size=target_size)
        d_test = Seg2D_Dataset(split='Test', dataset=args.dataset, transform=transforms, grayscale=grayscale, target_size=target_size)
        d_cal = Seg2D_Dataset(split='Calibration', dataset=args.dataset, transform=transforms, grayscale=grayscale, target_size=target_size)
        #d_test_np = Seg2D_Dataset(split='Test', dataset=args.dataset, grayscale=grayscale, target_size=target_size)
        #scores = compute_scores(d_test_np, n_classes)
        #idxs = sample_balanced(scores, n_buckets=5, min_val=0)
        print('Evaluating', len(d_test), 'samples for test and', len(d_cal), 'samples for calibration')

    rca_args = {'eval_metrics': ['Dice', 'Hausdorff', 'HD95', 'ASSD'],
                'n_test': args.n_test,
                'n_classes': n_classes,
                'emb_model': args.emb_model
                }
    
    # Set checkpoint and configuration in case of using SAM 2 as In-Context classifier
    if args.classifier == 'sam2':
        sam2_checkpoints = 'sam2/checkpoints/'
        model_type_dict = {'sam2_hiera_t':'sam2_hiera_tiny', 'sam2_hiera_s':'sam2_hiera_small', 'sam2_hiera_b+':'sam2_hiera_base_plus','sam2_hiera_l':'sam2_hiera_large', 'sam2.1_hiera_t':'sam2.1_hiera_tiny'}
        
        for k,v in model_type_dict.items():
            model_type_dict[k] = os.path.join(sam2_checkpoints, f'{v}.pt')
        
        rca_args['config'] = f"{args.config}.yaml"
        rca_args['checkpoint'] = model_type_dict[args.config]

    rca_clf = RCA(model=args.classifier, **rca_args)
    
    preds_test = rca_clf.run_evaluation(d_reference, d_test)
    to_json(preds_test, args.output_file + '_test.json')

    preds_cal = rca_clf.run_evaluation(d_reference, d_cal)
    to_json(preds_cal, args.output_file + '_cal.json')

if __name__ == "__main__":
    main()

    