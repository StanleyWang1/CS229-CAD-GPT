import util
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

training_path = './MCMASTER_DATA/training_x_300.csv'
[x_train, y_train] = util.load_dataset(training_path, label_col='y', add_intercept=True)

# Feature Engineering
f1 = np.divide(x_train[:,0], x_train[:,5]) # V / BBV ratio
f2 = np.divide(x_train[:,2], x_train[:,4]) # X / Z ratio
f3 = np.divide(x_train[:,3], x_train[:,4]) # Y / Z ratio
f4 = np.divide(x_train[:,6], x_train[:,7]) # NV / NE ratio
f5 = np.divide(x_train[:,6], x_train[:,8]) # NV / NF ratio
f6 = np.divide(x_train[:,7], x_train[:,8]) # NE / NF ratio
x_train2 = np.vstack((f1, f2, f3, f4, f5, f6)).T
x_train3 = np.concatenate((x_train, x_train2), axis=1)

BBdims = np.vstack((x_train[:,2], x_train[:,3], x_train[:,4])).T         
max_values = np.max(BBdims, axis=1)
max_values = max_values.reshape(-1, 1)
min_values = np.min(BBdims, axis=1)
min_values = min_values.reshape(-1, 1)
aspect_ratio = np.divide(min_values, max_values)

packing_density = np.divide(x_train[:,0], x_train[:,5])
packing_density = np.reshape(packing_density, (300, 1))

x_train2D = np.concatenate((aspect_ratio[y_train != 2], packing_density[y_train != 2]), axis=1)
y_train2D = y_train[y_train != 2]
# x_train2D = np.concatenate((min_values[y_train != 2], max_values[y_train != 2]), axis=1)
# y_train2D = y_train[y_train != 2]

print(np.shape(x_train2D))
print(np.shape(y_train2D))
# # scaler = StandardScaler().fit(x_train3)
# # x_scaled = scaler.transform(x_train3)

model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
model.fit(x_train2D, y_train2D)
y_pred = model.predict(x_train2D)
accuracy = accuracy_score(y_train2D, y_pred)
print(f"Accuracy: {accuracy}")

# plt.figure()
# plt.scatter(np.reshape(x_train2D[:,0], (200, 1)), np.reshape(x_train2D[:,1], (200, 1)))
# plt.show()

print(np.shape(x_train2D[y_train2D==0][:, 0]))
print(np.shape(x_train2D[y_train2D==0][:, 1]))

plt.figure()
plt.scatter(x_train2D[y_train2D==0][:, 0], x_train2D[y_train2D==0][:, 1], color='blue')
plt.scatter(x_train2D[y_train2D==1][:, 0], x_train2D[y_train2D==1][:, 1], color='red')
plt.xlabel('Aspect Ratio')
plt.ylabel('Packing Density')
plt.show()

# util.plot_decision_boundary(model, x_train2D, y_train2D, feature_names=['Aspect Ratio', 'Packing Density'])