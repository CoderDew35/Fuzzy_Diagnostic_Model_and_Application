import numpy as np
from sklearn.cluster import KMeans
import skfuzzy.membership as skm
from fuzzy.sets import DEFAULT_NUM_SETS, DEFAULT_SET_NAMES, get_generic_set_names

def auto_partition(data_series, num_sets=None, set_names=None, universe_size=500, random_state=42):
    """
    Automatically partitions a data series into fuzzy sets using k-means.

    Args:
        data_series (pd.Series): The input data series for a sensor.
        num_sets (int, optional): The number of fuzzy sets to create. Defaults to DEFAULT_NUM_SETS.
        set_names (list of str, optional): Names for the fuzzy sets. Auto-generated if None.
        universe_size (int, optional): Number of points in the universe of discourse. Defaults to 500.
        random_state (int, optional): Random state for KMeans for reproducibility. Defaults to 42.

    Returns:
        dict: A dictionary where 'universe' maps to the universe array,
              and each set_name maps to its membership function array.
    """
    if num_sets is None:
        num_sets = DEFAULT_NUM_SETS

    if set_names is None:
        if num_sets == len(DEFAULT_SET_NAMES) and num_sets == 3: # Be specific for default names
            set_names = DEFAULT_SET_NAMES
        else:
            set_names = get_generic_set_names(num_sets)
    elif len(set_names) != num_sets:
        raise ValueError("Length of set_names must be equal to num_sets")

    min_val = data_series.min()
    max_val = data_series.max()

    if min_val == max_val:
        # Handle constant data series: create a single, narrow triangular MF
        # Define a small, non-degenerate universe around the constant value
        abs_v = abs(min_val)
        # Heuristic for delta to create a small spread for the universe
        delta = 0.1 if abs_v < 1.0 else 0.01 * abs_v 
        delta = max(delta, 0.001) # Ensure delta is not zero

        # Create 3 distinct points for the universe basis
        # Use float for min_val to ensure floating point arithmetic for delta
        u_pts = sorted(list(set([float(min_val) - delta, float(min_val), float(min_val) + delta])))
        
        # If points are not distinct (e.g., min_val is very small, delta becomes 0 relative to it)
        if len(u_pts) < 3: 
            u_pts = [float(min_val) - 0.1, float(min_val), float(min_val) + 0.1] # Fallback
            # Final check if min_val itself is 0.0 and precision issues persist
            if u_pts[0] == u_pts[1] or u_pts[1] == u_pts[2]:
                 u_pts = [-0.1, 0.0, 0.1] # Absolute fallback for zero-like values

        x = np.linspace(u_pts[0], u_pts[-1], universe_size)
        
        # Create one fuzzy set centered at min_val
        # All other sets, if num_sets > 1, would be identical, so we simplify.
        mf = skm.trimf(x, [u_pts[0], float(min_val), u_pts[-1]])
        return {'universe': x, set_names[0]: mf}

    x = np.linspace(min_val, max_val, universe_size)
    data_reshaped = data_series.values.reshape(-1, 1)

    # KMeans clustering
    # n_init='auto' is for scikit-learn >= 1.2. Use n_init=10 for older versions if needed.
    kmeans = KMeans(n_clusters=num_sets, n_init='auto', random_state=random_state)
    kmeans.fit(data_reshaped)
    centers = np.sort(kmeans.cluster_centers_.flatten())

    # Ensure centers are unique; if k-means produces duplicate centers (e.g. few unique data points)
    # This can happen if data_series.nunique() < num_sets
    unique_centers = sorted(list(set(centers)))
    if len(unique_centers) < num_sets:
        # If not enough unique centers, adjust num_sets and set_names
        # This is a simplification; more sophisticated handling might be needed
        num_sets = len(unique_centers)
        set_names = set_names[:num_sets] 
        centers = np.array(unique_centers)
        if num_sets == 0 : # Should not happen if data_series is not empty and min_val != max_val
             return {'universe': x} # empty MFs

    fuzzy_output = {'universe': x}
    
    if num_sets == 0: # After potential adjustment
        return fuzzy_output
    elif num_sets == 1:
        fuzzy_output[set_names[0]] = skm.trimf(x, [min_val, centers[0], max_val])
    else:
        # First MF: from min_val, peak at centers[0], end at centers[1]
        fuzzy_output[set_names[0]] = skm.trimf(x, [min_val, centers[0], centers[1]])
        # Middle MFs
        for i in range(1, num_sets - 1):
            fuzzy_output[set_names[i]] = skm.trimf(x, [centers[i-1], centers[i], centers[i+1]])
        # Last MF: from centers[num_sets-2], peak at centers[num_sets-1], end at max_val
        fuzzy_output[set_names[num_sets-1]] = skm.trimf(x, [centers[num_sets-2], centers[num_sets-1], max_val])
        
    return fuzzy_output
