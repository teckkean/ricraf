#!/usr/bin/env python3
"""
Utility functions for RICRAF development.
Supports `notebooks/ricraf_development.ipynb` and other project components.
"""


import geopandas as gpd
import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from loguru import logger
from shapely.geometry import LineString, MultiLineString

# # Turn off future warnings to keep the logs tidy
# warnings.filterwarnings("ignore", category=FutureWarning)


def filter_road_network(input_file, filtered_output, removed_output):
    """
    Filter road network GeoJSON based on specified outlier criteria and save results.

    Parameters:
    input_file (str): Path to input GeoJSON file
    filtered_output (str): Path to save filtered GeoJSON
    removed_output (str): Path to save removed (outlier) GeoJSON
    """
    try:
        # Read the GeoJSON file
        gdf = gpd.read_file(input_file)
        logger.info(f"Initial number of features: {len(gdf)}")

        # Define outlier criteria
        outlier_condition = (
                (gdf['link_length'] < 10) |
                (gdf['ALLVEHS_AA'] < 100) |
                (gdf['TRUCKS_AAD'] < 10)
        )

        # Split into filtered (non-outliers) and removed (outliers) datasets
        gdf_filtered = gdf[~outlier_condition].copy()
        gdf_removed = gdf[outlier_condition].copy()

        # Log counts after filtering
        logger.info(f"Number of features after filtering (non-outliers): {len(gdf_filtered)}")
        logger.info(f"Number of features removed (outliers): {len(gdf_removed)}")

        # Save the filtered dataset
        gdf_filtered.to_file(filtered_output, driver='GeoJSON')
        logger.info(f"Filtered dataset saved to: {filtered_output}")

        # Save the removed dataset
        gdf_removed.to_file(removed_output, driver='GeoJSON')
        logger.info(f"Removed dataset saved to: {removed_output}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


def custom_weighted_smote(X, y, sample_weight, sampling_strategy='auto', k_neighbors=5, random_state=42):
    """
    Custom SMOTE implementation that incorporates sample_weight to influence minority sample selection.

    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Input features.
    y : array-like of shape (n_samples,)
        Target labels.
    sample_weight : array-like of shape (n_samples,)
        Weights for each sample.
    sampling_strategy : float, str, dict, or callable, default='auto'
        Sampling strategy as per imblearn SMOTE.
    k_neighbors : int, default=5
        Number of nearest neighbors to use for synthetic sample generation.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns:
    --------
    X_resampled : array-like of shape (n_samples_new, n_features)
        Resampled features.
    y_resampled : array-like of shape (n_samples_new,)
        Resampled labels.
    sample_weight_resampled : array-like of shape (n_samples_new,)
        Resampled weights.
    """
    np.random.seed(random_state)

    # Input validation
    X = np.asarray(X)
    y = np.asarray(y)
    sample_weight = np.asarray(sample_weight)
    if len(X) != len(y) or len(X) != len(sample_weight):
        raise ValueError("X, y, and sample_weight must have the same length.")

    # Compute class distribution
    class_counts = Counter(y)
    majority_class = max(class_counts, key=class_counts.get)
    n_majority = class_counts[majority_class]

    # Compute sampling strategy
    sampling_strategy_dict = {}
    if isinstance(sampling_strategy, (float, int)):
        if sampling_strategy <= 0:
            raise ValueError("sampling_strategy must be positive.")
        for cls, count in class_counts.items():
            if cls != majority_class:
                sampling_strategy_dict[cls] = int(n_majority * sampling_strategy) - count
    elif isinstance(sampling_strategy, str):
        if sampling_strategy == 'auto' or sampling_strategy == 'minority':
            for cls, count in class_counts.items():
                if cls != majority_class:
                    sampling_strategy_dict[cls] = n_majority - count
        elif sampling_strategy == 'all':
            for cls, count in class_counts.items():
                sampling_strategy_dict[cls] = n_majority - count
        else:
            raise ValueError(f"Unsupported sampling_strategy string: {sampling_strategy}")
    elif isinstance(sampling_strategy, dict):
        sampling_strategy_dict = {cls: max(0, n_samples - class_counts.get(cls, 0))
                                  for cls, n_samples in sampling_strategy.items()}
    elif callable(sampling_strategy):
        sampling_strategy_dict = sampling_strategy(y)
    else:
        raise ValueError("sampling_strategy must be float, str, dict, or callable.")

    X_resampled = X.copy()
    y_resampled = y.copy()
    sample_weight_resampled = sample_weight.copy()

    for target_class, n_samples in sampling_strategy_dict.items():
        if n_samples <= 0:
            continue

        # Select minority class samples
        mask = y == target_class
        X_minority = X[mask]
        sample_weight_minority = sample_weight[mask]
        n_minority = len(X_minority)

        if n_minority < k_neighbors + 1:
            raise ValueError(f"Class {target_class} has too few samples ({n_minority}) for k_neighbors={k_neighbors}.")

        # Normalize weights to probabilities
        weight_probs = sample_weight_minority / np.sum(sample_weight_minority)
        if np.any(np.isnan(weight_probs)) or np.any(weight_probs < 0):
            weight_probs = np.ones_like(weight_probs) / len(weight_probs)

        # Select samples to generate synthetic instances based on weights
        n_synthetic = n_samples
        indices = np.random.choice(n_minority, size=n_synthetic, p=weight_probs, replace=True)
        X_to_sample = X_minority[indices]
        sample_weight_synthetic = sample_weight_minority[indices]

        # Fit nearest neighbors on minority samples
        nn = NearestNeighbors(n_neighbors=k_neighbors + 1, n_jobs=-1)
        nn.fit(X_minority)
        distances, neighbors = nn.kneighbors(X_to_sample)
        neighbors = neighbors[:, 1:]  # Exclude self

        # Generate synthetic samples
        X_synthetic = []
        y_synthetic = []
        weight_synthetic = []

        for i in range(n_synthetic):
            x = X_to_sample[i]
            nn_idx = neighbors[i]
            nn_selected = np.random.choice(nn_idx)
            x_nn = X_minority[nn_selected]
            alpha = np.random.random()
            x_new = x + alpha * (x_nn - x)
            X_synthetic.append(x_new)
            y_synthetic.append(target_class)
            weight_synthetic.append((sample_weight_minority[indices[i]] + sample_weight_minority[nn_selected]) / 2)

        # Append synthetic samples
        X_resampled = np.vstack([X_resampled, X_synthetic])
        y_resampled = np.hstack([y_resampled, y_synthetic])
        sample_weight_resampled = np.hstack([sample_weight_resampled, weight_synthetic])

    return X_resampled, y_resampled, sample_weight_resampled


def offset_line(geometry, distance, objectid=None):
    """
    Offset geometry to the left by specified distance.

    Args:
        geometry: Shapely LineString or MultiLineString
        distance: Offset distance in meters
        objectid: The OBJECTID of the feature (optional)

    Returns:
        Offset geometry or original geometry if operation fails
    """
    global total_warnings
    try:
        if isinstance(geometry, LineString):
            offset_geom = geometry.parallel_offset(distance, 'left')
            return offset_geom if not offset_geom.is_empty else geometry

        elif isinstance(geometry, MultiLineString):
            offset_lines = [
                line.parallel_offset(distance, 'left')
                for line in geometry.geoms
                if not line.parallel_offset(distance, 'left').is_empty
            ]
            return MultiLineString(offset_lines) if offset_lines else geometry
        return geometry
    except Exception as e:
        objectid_str = f" for OBJECTID {objectid}" if objectid is not None else ""
        error_msg = f"Offset operation failed{objectid_str} - {str(e)}. Returning original geometry."
        logger.warning(error_msg)
        total_warnings += 1
        return geometry