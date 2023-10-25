# @STANLEY WANG 2023
# Demo Code for Using CAD_interface class (built off cadquery to import STEP file and extract key features)

from CAD_interface import StepReader
import os
import csv

data_dir = "./MCMASTER_DATA/training_x_300/"
output_file = "./MCMASTER_DATA/training_x_300.csv"
labels = sorted([l for l in os.listdir(data_dir) if not l.startswith(".")])

with open(output_file, mode="a", newline="") as csv_data:
    for ind, label in enumerate(labels):  # iterate through labeled folders in data
        sub_dir = data_dir + label + "/"
        filenames = [f for f in os.listdir(sub_dir) if not f.startswith(".")]
        for file in filenames:  # iterate through files w/ given label
            S = StepReader(sub_dir + file)
            data = [S.V, S.SA, S.BBX, S.BBY, S.BBZ, S.BBV, S.NV, S.NE, S.NF, ind]
            writer = csv.writer(csv_data)
            writer.writerow(data)
            print("Successfully parsed " + file.split("_")[1])
            