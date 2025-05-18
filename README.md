## Fuzzy Logic Simulation

This Python project demonstrates a fuzzy logic system for diagnosing potential engine failures based on sensor data. It automates the process of creating fuzzy sets from data, inducing rules, and performing inference to predict fault conditions. The system is designed to handle uncertainty and imprecision inherent in real-world sensor readings.

Key features include:
-   Automatic fuzzy partitioning of sensor data using k-means clustering.
-   Generation of fuzzy association rules (IF-THEN rules) from historical data.
-   A priority-based fuzzy inference engine for diagnosing faults.
-   Visualization tools for inspecting fuzzy sets and rule activation.

```
fuzzy_diagnosis/
├── data/
│   └── engine_failure_detection.csv    # ← download from Kaggle and save here
├── fuzzy/
│   ├── sets.py                         # fuzzy‐set definitions and helpers
│   ├── partition.py                    # auto‐generate fuzzy partitions from data
│   ├── rule_generator.py               # fast rule‐induction
│   └── inference_engine.py             # priority‐based diagnosis
│   └── visualization.py                # plotting utilities
├── run_demo.py                         # main script to run the simulation
├── README.md                           # this file, instructions for acquiring Kaggle data
└── requirements.txt                    # project dependencies
```

## Project Workflow

The simulation follows these main steps:

1.  **Data Loading**: Sensor readings and corresponding fault conditions are loaded from a CSV file (e.g., `engine_failure_detection.csv`).
2.  **Fuzzy Partitioning**: For each sensor, its range of values is automatically divided into a set of fuzzy linguistic terms (e.g., "Low", "Medium", "High"). This is achieved by applying k-means clustering to the sensor data to find representative centers, which then define triangular membership functions.
3.  **Rule Generation**: Fuzzy association rules are generated from the historical data. These rules take the form:
    `IF (SensorA is RegionX) AND (SensorB is RegionY) ... THEN (Fault_Condition is Z)`
    Rules are generated based on the co-occurrence of fuzzy regions and fault labels, and are assigned support and confidence scores.
4.  **Fuzzy Inference**: Given a new set of sensor readings (a test case), the system diagnoses a potential fault. This involves:
    *   Calculating the membership degree of each sensor reading in the relevant fuzzy sets.
    *   Determining the firing strength of each rule based on its antecedents' membership degrees (typically using a t-norm like `min`).
    *   Selecting the rule with the highest firing strength (above a certain threshold) to determine the diagnosed fault. If multiple rules fire, priority (e.g., based on confidence) can be used.
5.  **Visualization**:
    *   The fuzzy sets (partitions) for each sensor are plotted to show how the sensor's range is divided.
    *   For a given test reading and the rule that led to the diagnosis, the activation of the rule's antecedents is visualized. This shows how the input sensor values map to the fuzzy sets involved in the activated rule.

## Setup

1.  **Acquire Data**: Download an appropriate `engine_failure_detection.csv` dataset. A common source is Kaggle. You will need a dataset with multiple sensor readings as features and a column indicating a fault condition or operational state. Place this file in the `data/` directory.
    *Example: Search Kaggle for "engine sensor data for predictive maintenance" or similar terms.*
2.  **Install Dependencies**: Create a `requirements.txt` file with the following content and install using `pip install -r requirements.txt`:

    ```txt
    pandas
    scikit-fuzzy
    scikit-learn
    numpy
    matplotlib
    ```
3.  **Run Demo**: Execute the demo script from the root directory of the project:
    ```bash
    python run_demo.py
    ```