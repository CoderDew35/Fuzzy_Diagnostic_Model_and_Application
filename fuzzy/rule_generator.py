from collections import Counter
import skfuzzy.fuzzymath.fuzzy_ops as fuzzy_ops


def generate_rules(data, fuzzy_sets):
    # data: list of (reading_dict, label)
    counts = Counter()
    label_counts = Counter()
    for reading, label in data:
        regions = {}
        for sensor, value in reading.items():
            x = fuzzy_sets[sensor]['universe']
            mfs = {name: fuzzy_ops.interp_membership(x, mf, value)
                    for name, mf in fuzzy_sets[sensor].items()
                    if name != 'universe'}
            # pick region with max membership
            regions[sensor] = max(mfs, key=mfs.get)
        rule_key = tuple(sorted(regions.items()))
        counts[(rule_key, label)] += 1
        label_counts[rule_key] += 1

    # build rule list
    rules = []
    for (rule_key, label), cnt in counts.items():
        support = cnt / sum(label_counts.values())
        confidence = cnt / label_counts[rule_key]
        rules.append({
            'antecedent': dict(rule_key),
            'consequent': label,
            'support': support,
            'confidence': confidence,
            'priority': confidence  # or some function of support/confidence
        })
    # sort descending by priority
    return sorted(rules, key=lambda r: r['priority'], reverse=True)
