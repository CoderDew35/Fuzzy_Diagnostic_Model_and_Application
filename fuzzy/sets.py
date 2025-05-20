# Default number of fuzzy sets for auto-partitioning
DEFAULT_NUM_SETS = 3

# Default names for fuzzy sets when num_sets is 3
DEFAULT_SET_NAMES = ['low', 'medium', 'high']

# Generic names if specific names are not provided or num_sets doesn't match
def get_generic_set_names(num_sets):
    """Generates a list of generic set names, e.g., S0, S1, ..."""
    return [f'S{i}' for i in range(num_sets)]
