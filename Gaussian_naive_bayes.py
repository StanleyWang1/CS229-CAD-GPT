import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd
class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Calculate the mean, variance and priors for each class
        self.mean = np.zeros((n_classes,X.shape[1]), dtype=np.float64)
        self.var = np.zeros((n_classes, X.shape[1]), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for idx, label in enumerate(self.classes):
            X_class = X[y == label]
            self.mean[idx, :] = X_class.mean(axis=0)
            self.var[idx, :] =X_class.var(axis=0)
            self.priors[idx] = X_class.shape[0] / float(X.shape[0])

    def predict(self, X):
        y_pred = [self._predict(sample) for sample in X]
        return np.array(y_pred)

    def _predict(self, sample):
        posteriors = []

        # Compute posterior probability for each class
        for idx, label in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            class_conditional = np.sum(np.log(self.pdf(idx, sample)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx] + 1e-10  # Adding a small constant to avoid division by zero
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


def k_fold_cross_validation(X, y, k=5):
    # Shuffle the data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    fold_size = len(X) // k
    accuracies = []

    for i in range(k):
        # Create train and validation splits
        validation_indices = indices[i * fold_size:(i + 1) * fold_size]
        train_indices = np.setdiff1d(indices, validation_indices)

        X_train, X_val = X[train_indices], X[validation_indices]
        y_train, y_val = y[train_indices], y[validation_indices]

        # Train the classifier
        clf = GaussianNaiveBayes()
        clf.fit(X_train, y_train)

        # Validate the classifier
        y_pred = clf.predict(X_val)
        accuracy =np.mean(y_pred == y_val)
        accuracies.append(accuracy)

    return accuracies

# Load the data

#data = pd.read_csv("extracted_data.csv")
#X = data.drop("label", axis=1).values
#y = data["label"].map({"nut": 0, "bolt": 1, "washer": 2}).values

start_time = time.time()

data = pd.read_csv("training_x_600.csv")
X = data.drop(columns=["y"]).values
y = data["y"].values


# Splitting data
train_size = int(0.9 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train the classifier
gnb = GaussianNaiveBayes()
gnb.fit(X_train, y_train)
predictions = gnb.predict(X_test)

accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

accuracies = k_fold_cross_validation(X, y, k=5)
average_accuracy = np.mean(accuracies)

print(f"Accuracies for each fold: {accuracies}")
print(f"Average Accuracy: {average_accuracy:.4f}")

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Runtime: {elapsed_time:.2f} seconds")


def plot_feature_distribution(X, y, feature_index, class_labels):
    colors = ['r', 'g', 'b']
    plt.figure(figsize=(10, 5))

    all_data = X[:, feature_index]
    mean = np.mean(all_data)
    std = np.std(all_data)

    # Calculate the bounds for 80% of the data (based on Z-table, roughly 1.28 standard deviations for 80%)
    lower_bound = mean - 1.28 * std
    upper_bound = mean + 1.28 * std

    for label, color in zip(class_labels, colors):
        data = X[y == label][:, feature_index]
        data = data[~np.isnan(data) & ~np.isinf(data) & ~np.isneginf(
            data)]  # Check for NaN, positive infinity, and negative infinity

        # Only consider data within the 80% bound
        data = data[(data >= lower_bound) & (data <= upper_bound)]

        if data.size == 0:
            continue
        sns.histplot(data, color=color, label=f'{label}', kde=True, bins=30)  # Specifying the bins
        plt.title(f"Feature {feature_index + 1} Distribution")
        plt.xlabel(f"Feature {feature_index + 1}")
        plt.ylabel("Density")
        plt.legend()
    plt.show()


for i in range(X_train.shape[1]):
    plot_feature_distribution(X_train, y_train, i, [0, 1, 2])
