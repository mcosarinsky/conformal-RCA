import medpy.metric.binary as metrics
from itertools import cycle
import numpy as np

is_overlap = {
    'Dice': True
}

def multiclass_score(result, reference, metric, num_classes):
    scores = []
    
    for i in range(1, num_classes+1): 
        result_i, reference_i = (result == i).astype(int), (reference==i).astype(int)
        scores.append(metric(result_i, reference_i))
    
    return scores

def Hausdorff(result, reference):
    if np.count_nonzero(result) == 0 or np.count_nonzero(reference) == 0:
        return np.inf
    return metrics.hd(result, reference)

def HD95(result, reference):
    if np.count_nonzero(result) == 0 or np.count_nonzero(reference) == 0:
        return np.inf
    return metrics.hd95(result, reference)

def ASSD(result, reference):
    if np.count_nonzero(result) == 0 or np.count_nonzero(reference) == 0:
        return np.inf
    return metrics.assd(result, reference)

def Dice(result, reference):
    return metrics.dc(result, reference)

def compute_scores(data, num_classes, metric=Dice):
    scores = []
    
    for sample in data:
        result, reference = sample['seg'], sample['GT']
        score = multiclass_score(result, reference, metric, num_classes)
        scores.append(score)
    
    return np.array(scores)


def sample_N(scores, N, n_buckets=10):
    """
    Samples a specific total number of items, distributing samples across buckets.
    If a bucket cannot fulfill its quota, the remaining samples are distributed
    evenly among the other buckets that have capacity.
    
    Args:
        scores: Array of scores.
        N: Total number of samples to take.
        n_buckets: Number of buckets to create between 0 and 1 (default is 10).

    Returns:
        np.array: Indices of the sampled items.
    """
    bins = np.linspace(0, 1, n_buckets + 1)
    bucket_indices = np.digitize(scores, bins, right=False) - 1
    buckets = [np.where(bucket_indices == i)[0] for i in range(n_buckets)]

    actual_samples_per_bucket = [0] * n_buckets
    
    initial_target = [N // n_buckets] * n_buckets
    remainder_N = N % n_buckets
    for i in range(n_buckets - remainder_N, n_buckets):
        initial_target[i] += 1

    for i in range(n_buckets):
        actual_samples_per_bucket[i] = min(initial_target[i], len(buckets[i]))

    samples_to_redistribute = N - sum(actual_samples_per_bucket)

    if samples_to_redistribute > 0:
        eligible_bucket_indices_list = [
            i for i in range(n_buckets) 
            if len(buckets[i]) > actual_samples_per_bucket[i]
        ]
        
        if not eligible_bucket_indices_list:
            print(f"Warning: Could not fulfill N={N} samples. Only {sum(actual_samples_per_bucket)} sampled as no more items available for redistribution.")
        else:
            bucket_cycler = cycle(eligible_bucket_indices_list)
            while samples_to_redistribute > 0:
                current_bucket_idx = next(bucket_cycler)
                if actual_samples_per_bucket[current_bucket_idx] < len(buckets[current_bucket_idx]):
                    actual_samples_per_bucket[current_bucket_idx] += 1
                    samples_to_redistribute -= 1
                elif all(actual_samples_per_bucket[idx] >= len(buckets[idx]) for idx in eligible_bucket_indices_list):
                    # If all eligible buckets are now full, break the loop
                    print(f"Warning: Could not fulfill N={N} samples. Reached capacity of all eligible buckets. Still needed {samples_to_redistribute} samples.")
                    break
    
    final_sampled_indices = []
    for i in range(n_buckets):
        num_to_sample = actual_samples_per_bucket[i]
        if num_to_sample > 0:
            final_sampled_indices.extend(np.random.choice(buckets[i], size=num_to_sample, replace=False))

    return np.array(final_sampled_indices)


def sample_balanced(scores, n_buckets=10, min_val=0):
    """
    Sample the same number of items from each bucket for each class (if multiclass),
    and return the union of indices without duplicates.

    Args:
        scores (np.array): Array of scores. Shape (N,) for single class, (N, C) for multiclass.
        n_buckets (int): Number of buckets to divide scores into.

    Returns:
        np.array: Unique indices of the sampled items.
    """
    if scores.ndim == 1:
        scores = scores[:, np.newaxis]  # Convert to (N, 1) for unified handling
        
    n_classes = scores.shape[1]
    all_indices = None
    all_indices = []

    for c in range(n_classes):
        class_scores = scores[:, c]

        # Create bins and assign scores to buckets
        bins = np.linspace(min_val, 1, n_buckets + 1)
        bucket_indices = np.digitize(class_scores, bins, right=False) - 1
        buckets = [np.where(bucket_indices == i)[0] for i in range(n_buckets)]
        min_bucket_size = min(len(b) for b in buckets if len(b) > 0)

        for b in buckets:
            if len(b) >= min_bucket_size:
                all_indices.extend(np.random.choice(b, size=min_bucket_size, replace=False))

    return np.array(sorted(set(all_indices)))