# @STANLEY WANG 2023
# Demo Code for Using CAD_interface class (built off cadquery to import STEP file and extract key features)

from CAD_interface import StepReader

filename = "STEP_import_demo/91251A542_Black-Oxide Alloy Steel Socket Head Screw.STEP"
# filename = './STEP_import_demo/stanley_box.step'

S_test = StepReader(filename)
BB = S_test.data.val().BoundingBox()
COM = S_test.data.solids().val().centerOfMass
# print(S_test.data.val().centerOfMass)
# print(dir(S_test.data.val().mesh))
print(S_test.data.val().Center().x)
print(S_test.data.val().Center().y)
print(S_test.data.val().Center().z)
