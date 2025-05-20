import skfuzzy.fuzzymath.fuzzy_ops as fuzzy_ops


def diagnose(reading, fuzzy_sets, rules, threshold=0.4): # 0.01
    # precompute region memberships
    mems = {}
    for sensor, value in reading.items():
        if sensor not in fuzzy_sets or not isinstance(fuzzy_sets[sensor], dict) or 'universe' not in fuzzy_sets[sensor]:
            # print(f"Warning: Fuzzy set data for sensor '{sensor}' is incomplete or missing. Skipping.")
            continue # Skip this sensor if its fuzzy set data is problematic
        x = fuzzy_sets[sensor]['universe']
        mfs_for_sensor = {}
        for name, mf in fuzzy_sets[sensor].items():
            if name != 'universe':
                if mf is not None and len(mf) == len(x):
                     mfs_for_sensor[name] = fuzzy_ops.interp_membership(x, mf, value)
                # else:
                    # print(f"Warning: MF '{name}' for sensor '{sensor}' is invalid or mismatched. Skipping membership calculation.")
        mems[sensor] = mfs_for_sensor
    
    highest_firing_strength = -1.0
    candidate_rule_details = None

    for rule in rules: # rules are pre-sorted by priority
        strengths = []
        valid_antecedent = True
        for sensor, region in rule['antecedent'].items():
            if sensor not in mems: # Sensor data might have been skipped if fuzzy_sets were incomplete
                # print(f"Warning: Sensor '{sensor}' from rule antecedent not found in precomputed memberships. Rule may not fire correctly.")
                strengths.append(0.0) # Treat as zero membership if sensor data was bad
                valid_antecedent = False # Potentially mark rule as not fully evaluable
                break 
            if region not in mems[sensor]:
                # This can happen if a region name in a rule doesn't exist in fuzzy_sets[sensor]
                # or if the MF was invalid during mems calculation.
                # print(f"Warning: Region '{region}' for sensor '{sensor}' not found in memberships. Assigning 0 strength for this part.")
                strengths.append(0.0)
                continue
            membership_degree = mems[sensor].get(region, 0.0)
            strengths.append(membership_degree)
        
        if not valid_antecedent: # If a sensor in antecedent was missing from fuzzy_sets
            current_firing_strength = 0.0
        else:
            current_firing_strength = min(strengths) if strengths else 0.0
        
        if current_firing_strength > highest_firing_strength:
            highest_firing_strength = current_firing_strength
            candidate_rule_details = {
                'antecedent': rule['antecedent'], 
                'consequent': rule['consequent'], 
                'confidence': rule.get('confidence', 'N/A'),
                'support': rule.get('support', 'N/A'),
                'firing_strength': current_firing_strength,
                'original_rule_priority': rule.get('priority', 'N/A')
            }

    if candidate_rule_details is not None and highest_firing_strength >= threshold:
        return {
            'diagnosed_consequent': candidate_rule_details['consequent'],
            'activated_rule_info': candidate_rule_details,
            'status': 'threshold_met'
        }
    elif candidate_rule_details is not None:
        confidence_val = candidate_rule_details['confidence']
        support_val = candidate_rule_details['support']
        confidence_str = f"{confidence_val:.2f}" if isinstance(confidence_val, float) else str(confidence_val)
        support_str = f"{support_val:.2f}" if isinstance(support_val, float) else str(support_val)
        
        print(f"No rule fired above threshold {threshold}. "
              f"Max strength rule: IF {candidate_rule_details['antecedent']} THEN {candidate_rule_details['consequent']} "
              f"(RuleConf: {confidence_str}, Supp: {support_str}, "
              f"FiringStrength: {highest_firing_strength:.4f})")
        return {
            'diagnosed_consequent': None,
            'activated_rule_info': candidate_rule_details,
            'status': 'below_threshold'
        }
    else:
        print(f"No rule fired above threshold {threshold}. No rules were applicable or all had zero strength.")
        return {
            'diagnosed_consequent': None,
            'activated_rule_info': None,
            'status': 'no_applicable_rules'
        }
