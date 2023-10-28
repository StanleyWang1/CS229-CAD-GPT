import util
import numpy as np

training_path = './MCMASTER_DATA/training_x_600.csv'
[x_train, y_train] = util.load_dataset(training_path, label_col='y', add_intercept=True)

# Given Features (CADQuery)
f1 = x_train[:,1] # volume
f2 = x_train[:,2] # surface area
f3 = x_train[:,3] # bounding box x length
f4 = x_train[:,4] # bounding box y length
f5 = x_train[:,5] # bounding box z length
f6 = x_train[:,6] # bounding box volume 
f7 = x_train[:,7] # bounding box center x
f8 = x_train[:,8] # bounding box center y
f9 = x_train[:,9] # bounding box center z
f10 = x_train[:,10] # COM x
f11 = x_train[:,11] # COM y
f12 = x_train[:,12] # COM z
f13 = x_train[:,13] # number of vertices
f14 = x_train[:,14] # number of edges
f15 = x_train[:,15] # number of faces

# Engineered Features
a1 = np.divide(f3, f4) # nondimensional aspect ratio x/y
a2 = np.divide(f4, f5) # nondimensional aspect ratio y/z
a3 = np.divide(f5, f3) # nondimensional aspect ratio z/x
a4 = np.divide(f1, f6) # volumetric packing ratio
a5 = np.divide(f2, f3*f4 + f4*f5 + f5*f3) # area ratio
a6 = np.divide(f13, f14) # nondimensional aspect ratio NV/NE
a7 = np.divide(f14, f15) # nondimensional aspect ratio NE/NF
a8 = np.divide(f15, f13) # nondimensional aspect ratio NF/NV

X_NEW = np.column_stack((f1, f2, f3, f4, f5, f6, f7, f8, f8, f10, f11, f12, f13, f14, f15, a1, a2, a3, a4, a5, a6, a7, a8))
EXPORT = np.hstack((X_NEW, y_train[:, np.newaxis]))
file_path = "./MCMASTER_DATA/training_x_600_23_features.csv"
np.savetxt(file_path, EXPORT, delimiter=',', fmt='%f')
