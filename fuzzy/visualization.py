import matplotlib.pyplot as plt
import numpy as np
import math
import skfuzzy.fuzzymath.fuzzy_ops as fuzzy_ops

def plot_fuzzy_sets_for_sensor(sensor_name, sensor_partition_data, ax=None, show_legend=True):
    """
    Plots the fuzzy sets for a single sensor on a given Matplotlib Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        # Not showing plot here, caller should handle it or it's part of a larger figure.

    universe = sensor_partition_data.get('universe')
    if universe is None or len(universe) == 0:
        ax.text(0.5, 0.5, 'No universe data to plot', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(f'Fuzzy Sets for {sensor_name} (No Universe)')
        return ax

    mf_plotted = False
    for set_name, mf_values in sensor_partition_data.items():
        if set_name == 'universe':
            continue
        if mf_values is not None and len(mf_values) == len(universe):
            ax.plot(universe, mf_values, label=set_name)
            mf_plotted = True
        else:
            print(f"Warning: MF data for '{set_name}' in sensor '{sensor_name}' is invalid or mismatched with universe. Skipping.")


    ax.set_title(f'Fuzzy Sets for {sensor_name}')
    ax.set_xlabel('Sensor Value')
    ax.set_ylabel('Membership Degree')
    if show_legend and mf_plotted:
        ax.legend()
    ax.grid(True)
    
    if not mf_plotted and (universe is not None and len(universe) > 0):
        ax.text(0.5, 0.5, 'No valid MFs to plot', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    return ax

def plot_all_sensor_partitions(all_fuzzy_sets, sensors_to_plot=None, figsize_per_plot=(6, 4), max_cols=3):
    """
    Plots the fuzzy set partitions for multiple sensors in a grid.
    """
    if not all_fuzzy_sets:
        print("No fuzzy sets data provided for plotting.")
        return

    if sensors_to_plot is None:
        sensors_to_plot = list(all_fuzzy_sets.keys())
    
    if not sensors_to_plot:
        print("No sensors selected for plotting.")
        return

    num_sensors = len(sensors_to_plot)
    ncols = min(max_cols, num_sensors)
    if ncols == 0: ncols = 1 # Ensure ncols is at least 1
    nrows = math.ceil(num_sensors / ncols)
    if nrows == 0: nrows = 1 # Ensure nrows is at least 1


    fig_width = ncols * figsize_per_plot[0]
    fig_height = nrows * figsize_per_plot[1]

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)
    axes_flat = axes.flatten()

    for i, sensor_name in enumerate(sensors_to_plot):
        if i >= len(axes_flat): # Should not happen if nrows, ncols calculated correctly
            break 
        ax = axes_flat[i]
        if sensor_name not in all_fuzzy_sets or not isinstance(all_fuzzy_sets[sensor_name], dict):
            print(f"Warning: Fuzzy sets for sensor '{sensor_name}' not found or invalid. Skipping plot.")
            ax.text(0.5, 0.5, f"Data for\n'{sensor_name}'\nnot found\nor invalid",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        sensor_data = all_fuzzy_sets[sensor_name]
        plot_fuzzy_sets_for_sensor(sensor_name, sensor_data, ax=ax)

    # Hide any unused subplots
    for j in range(num_sensors, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout()
    plt.suptitle("Fuzzy Set Partitions for Sensors", fontsize=16, y=1.02)
    plt.show()

def plot_input_membership_for_rule_antecedent(test_reading, all_fuzzy_sets, rule_info,
                                              figsize_per_plot=(7, 4), max_cols=3):
    """
    Visualizes how a test reading activates the antecedent of a specific rule.
    """
    if not rule_info or 'antecedent' not in rule_info or not rule_info['antecedent']:
        print("Rule information is missing or has no antecedents to visualize.")
        return

    rule_antecedent = rule_info['antecedent']
    rule_consequent = rule_info.get('consequent', 'N/A')
    firing_strength = rule_info.get('firing_strength', 0.0)

    antecedent_sensors = list(rule_antecedent.keys())
    if not antecedent_sensors:
        print("Rule has no antecedents to visualize.")
        return

    num_sensors = len(antecedent_sensors)
    ncols = min(max_cols, num_sensors)
    if ncols == 0: ncols = 1
    nrows = math.ceil(num_sensors / ncols)
    if nrows == 0: nrows = 1


    fig_width = ncols * figsize_per_plot[0]
    fig_height = nrows * figsize_per_plot[1]

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)
    axes_flat = axes.flatten()

    fig.suptitle(f"Activation for Rule -> {rule_consequent} (Firing Strength: {firing_strength:.4f})", fontsize=16)

    for i, sensor_name in enumerate(antecedent_sensors):
        if i >= len(axes_flat): break
        ax = axes_flat[i]
        
        if sensor_name not in all_fuzzy_sets or not isinstance(all_fuzzy_sets[sensor_name], dict):
            ax.set_title(f"{sensor_name}\n(Fuzzy set data missing)")
            ax.text(0.5, 0.5, "Fuzzy set data\nmissing", ha='center', va='center', transform=ax.transAxes)
            continue

        sensor_partitions = all_fuzzy_sets[sensor_name]
        plot_fuzzy_sets_for_sensor(sensor_name, sensor_partitions, ax=ax, show_legend=False)

        target_region_name = rule_antecedent[sensor_name]
        sensor_value = test_reading.get(sensor_name)

        if sensor_value is None:
            ax.set_title(f"{sensor_name}\n(Test value missing)")
            ax.text(0.5, 0.5, "Test value\nmissing", ha='center', va='center', transform=ax.transAxes)
            continue
        
        ax.axvline(sensor_value, color='r', linestyle='--', label=f'Test Value: {sensor_value:.2f}')

        if target_region_name in sensor_partitions and \
           sensor_partitions.get('universe') is not None and \
           target_region_name in sensor_partitions and \
           sensor_partitions.get(target_region_name) is not None:
            
            mf_universe = sensor_partitions['universe']
            mf_values = sensor_partitions[target_region_name]
            
            if len(mf_universe) == len(mf_values):
                membership_degree = fuzzy_ops.interp_membership(mf_universe, mf_values, sensor_value)
                ax.plot(mf_universe, mf_values, color='k', linewidth=2.5, label=f'{target_region_name} (Rule)')
                ax.fill_between(mf_universe, mf_values, alpha=0.2, color='k')
                ax.plot(sensor_value, membership_degree, 'ro', markersize=8)
                ax.text(sensor_value, membership_degree + 0.05, f'{membership_degree:.2f}', color='r', ha='center')
                ax.set_title(f"{sensor_name}: Val {sensor_value:.2f} in '{target_region_name}' (μ={membership_degree:.2f})")
            else:
                ax.set_title(f"{sensor_name}: Val {sensor_value:.2f} (MF data error for '{target_region_name}')")
        else:
            ax.set_title(f"{sensor_name}: Val {sensor_value:.2f} (Region '{target_region_name}' or universe not found)")
        
        ax.legend(loc='best')

    for j in range(num_sensors, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
