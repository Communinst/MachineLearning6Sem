import numpy as np
from collections import Counter

def find_best_split(feature_vector, target_vector):
    feature_vector = np.asarray(feature_vector)
    target_vector = np.asarray(target_vector)
    num_samples = feature_vector.size

    sort_order = np.argsort(feature_vector)
    sorted_features = feature_vector[sort_order]
    sorted_targets = target_vector[sort_order]

    split_candidates = np.where(sorted_features[1:] != sorted_features[:-1])[0]
    if not split_candidates.size:
        return None, None, None, None

    samples_left = np.arange(1, num_samples)  # 1 to n-1
    class1_left = np.cumsum(sorted_targets)[:-1]
    class0_left = samples_left - class1_left

    total_class1 = class1_left[-1] + sorted_targets[-1]
    total_class0 = num_samples - total_class1
    
    samples_right = num_samples - samples_left
    class1_right = total_class1 - class1_left
    class0_right = total_class0 - class0_left

    gini_left = 1 - (class1_left/samples_left)**2 - (class0_left/samples_left)**2
    gini_right = 1 - (class1_right/samples_right)**2 - (class0_right/samples_right)**2

    thresholds = (sorted_features[1:] + sorted_features[:-1]) / 2
    gini_gain = -(samples_left/num_samples)*gini_left - (samples_right/num_samples)*gini_right

    valid_gains = gini_gain[split_candidates]
    valid_thresholds = thresholds[split_candidates]
    best_idx = np.argmax(valid_gains)

    return valid_thresholds, valid_gains, valid_thresholds[best_idx], valid_gains[best_idx]

class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if any(ft not in ("real", "categorical") for ft in feature_types):
            raise ValueError("Unknown feature type in feature_types")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y == sub_y[0]):
            self._make_terminal(node, sub_y[0])
            return

        depth = node.get("depth", 0)
        if (self._max_depth is not None and depth >= self._max_depth) or \
           (self._min_samples_split is not None and len(sub_y) < self._min_samples_split):
            self._make_terminal(node, Counter(sub_y).most_common(1)[0][0])
            return

        best_feature, best_threshold, best_gini, best_split = None, None, None, None
        
        for feature_idx, feature_type in enumerate(self._feature_types):
            feature_vector = sub_X[:, feature_idx]
            
            if feature_type == "real":
                pass  
            elif feature_type == "categorical":
                categories, counts = np.unique(feature_vector, return_counts=True)
                click_rates = [np.sum(sub_y[feature_vector == cat]) / cnt 
                             for cat, cnt in zip(categories, counts)]
                
                sorted_order = np.argsort(click_rates)
                cat_to_num = {cat: i for i, cat in enumerate(categories[sorted_order])}
                feature_vector = np.array([cat_to_num[cat] for cat in feature_vector])
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            
            if threshold is None:
                continue

            if best_gini is None or gini > best_gini:
                best_feature = feature_idx
                best_gini = gini
                current_split = feature_vector < threshold

                if feature_type == "real":
                    best_threshold = threshold
                else:  
                    best_threshold = [cat for cat in cat_to_num 
                                     if cat_to_num[cat] < threshold]


        if best_feature is None:
            self._make_terminal(node, Counter(sub_y).most_common(1)[0][0])
            return

        node.update({
            "type": "nonterminal",
            "feature_split": best_feature,
            "left_child": {},
            "right_child": {}
        })
        
        if self._feature_types[best_feature] == "real":
            node["threshold"] = best_threshold
        else:
            node["categories_split"] = best_threshold

        left_mask = current_split
        right_mask = ~current_split

        if (self._min_samples_leaf is not None and 
            (np.sum(left_mask) < self._min_samples_leaf or 
             np.sum(right_mask) < self._min_samples_leaf)):
            self._make_terminal(node, Counter(sub_y).most_common(1)[0][0])
            return

        new_depth = depth + 1
        node["left_child"]["depth"] = new_depth
        node["right_child"]["depth"] = new_depth
        
        self._fit_node(sub_X[left_mask], sub_y[left_mask], node["left_child"])
        self._fit_node(sub_X[right_mask], sub_y[right_mask], node["right_child"])

    def _make_terminal(self, node, class_label):
        """Helper to convert node to terminal"""
        node.clear()
        node.update({
            "type": "terminal",
            "class": class_label
        })

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature_idx = node["feature_split"]
        feature_type = self._feature_types[feature_idx]
        feature_value = x[feature_idx]

        if feature_type == "real":
            child = "left_child" if feature_value < node["threshold"] else "right_child"
        else:  
            child = "left_child" if feature_value in node["categories_split"] else "right_child"

        return self._predict_node(x, node[child])

    def fit(self, X, y):
        self._tree = {} 
        self._fit_node(np.array(X), np.array(y), self._tree)

    def predict(self, X):
        return np.array([self._predict_node(x, self._tree) for x in np.array(X)])