import pandas as pd
from collections import Counter

def gini_impurity(labels):
    impurity = 1
    label_counts = Counter(labels)
    for label in label_counts:
        prob_of_label = label_counts[label] / len(labels)
        impurity -= prob_of_label ** 2
    return impurity


# split dataset
def split_dataset(data, labels, column, value):
    left_data, right_data, left_labels, right_labels = [], [], [], []
    for row, label in zip(data, labels):
        if row[column] < value:
            left_data.append(row)
            left_labels.append(label)
        else:
            right_data.append(row)
            right_labels.append(label)
    return left_data, right_data, left_labels, right_labels



class DecisionNode:
    def __init__(self, column, value, true_branch, false_branch):
        self.column = column
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch

class LeafNode:
    def __init__(self, labels):
        self.predictions = Counter(labels)


# build the tree
def build_tree(data, labels, depth=1, max_depth=5):
    if len(set(labels)) == 1:
        return LeafNode(labels)
    if depth == max_depth:
        return LeafNode(labels)

    best_gini = 1
    best_split = None
    current_gini = gini_impurity(labels)

    n_features = len(data[0])
    for col in range(n_features):
        values = set([row[col] for row in data])
        for value in values:
            left_data, right_data, left_labels, right_labels = split_dataset(data, labels, col, value)
            if len(left_data) == 0 or len(right_data) == 0:
                continue

            p_left = len(left_data) / len(data)
            gini = (p_left*gini_impurity(left_labels)) + ((1 - p_left) * gini_impurity(right_labels))

            if gini < best_gini:
                best_gini = gini
                best_split = (col, value, left_data, right_data, left_labels, right_labels)

    if best_gini == current_gini:
        return LeafNode(labels)

    true_branch = build_tree(best_split[2], best_split[4], depth + 1, max_depth)
    false_branch = build_tree(best_split[3], best_split[5], depth + 1, max_depth)

    return DecisionNode(best_split[0], best_split[1], true_branch, false_branch)


# predictions
def classify(row, node):
    if isinstance(node, LeafNode):
        return node.predictions.most_common(1)[0][0]

    if row[node.column] < node.value:
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


# Load
dataset = pd.read_csv("extracted_data.csv")

# Convert to lists
data_list = dataset.drop(columns=['label']).values.tolist()
labels_list = dataset['label'].tolist()

# Train/test split
train_data,test_data,train_labels,test_labels =data_list[:240], data_list[240:], labels_list[:240], labels_list[240:]

# Train the tree
tree = build_tree(train_data, train_labels)

# Test the tree
correct = 0
for row, label in zip(test_data, test_labels):
    prediction = classify(row, tree)
    if prediction == label:
        correct += 1

accuracy = correct / len(test_data)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(correct)
