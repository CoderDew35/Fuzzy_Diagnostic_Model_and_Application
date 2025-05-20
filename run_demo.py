import pandas as pd
from fuzzy.partition import auto_partition
from fuzzy.rule_generator import generate_rules
from fuzzy.inference_engine import diagnose
from fuzzy.visualization import plot_all_sensor_partitions, plot_input_membership_for_rule_antecedent
import matplotlib.pyplot as plt

# 1. Load the Kaggle CSV
df = pd.read_csv("data/engine_failure_detection.csv")
# assume the last column is 'fault' and the rest are sensor readings
sensors = ['Temperature (°C)', 'RPM', 'Fuel_Efficiency', 'Vibration_X', 'Vibration_Y', 'Vibration_Z', 'Torque', 'Power_Output (kW)']
fault_column = 'Fault_Condition'

# 2. Split into (reading, label) pairs
data = list(zip(df[sensors].to_dict(orient='records'),
                df[fault_column].tolist()))

print(f"First data point (reading, label): {data[0]}")
print("Number of readings:", len(data))

# 3. Auto‐partition each sensor’s range into fuzzy sets
#    (e.g. 3 triangles per sensor, via k‐means on data[sensor])
fuzzy_sets = {s: auto_partition(df[s]) for s in sensors}

# Visualize fuzzy sets
plot_all_sensor_partitions(fuzzy_sets, sensors_to_plot=sensors)
plt.show() # Ensure plots are displayed

# 4. Generate rules and run one demo inference
rules = generate_rules(data, fuzzy_sets)
print(f"\nGenerated {len(rules)} rules. Top 5 rules:")
for i, rule in enumerate(rules[:5]):
    print(f"Rule {i+1}: IF {rule['antecedent']} THEN {rule['consequent']} (Conf: {rule['confidence']:.2f}, Supp: {rule['support']:.2f})")

test_reading = {s: df[s].iloc[0] for s in sensors}  # example row
print(f"\nTest reading: {test_reading}")
actual_fault = df[fault_column].iloc[0]
print(f"Actual fault for test reading: {actual_fault}")

print("\nDiagnosing fault...")
diagnosis_result = diagnose(test_reading, fuzzy_sets, rules)
diagnosed_fault = diagnosis_result['diagnosed_consequent']
activated_rule_info = diagnosis_result['activated_rule_info']

print(f"Diagnosed fault: {diagnosed_fault}")

# Visualize the activation of the best rule found
if activated_rule_info:
    print(f"\nVisualizing activation for the rule that led to diagnosis (or best candidate):")
    print(f"Rule: IF {activated_rule_info['antecedent']} THEN {activated_rule_info['consequent']}")
    print(f"Firing Strength: {activated_rule_info['firing_strength']:.4f}")
    plot_input_membership_for_rule_antecedent(test_reading, fuzzy_sets, activated_rule_info)
    plt.show() # Ensure plot is displayed
else:
    print("\nNo rule information available to visualize for activation.")
