import cadquery as cq


class StepReader:
    def __init__(self, filename):
        try:
            self.data = cq.importers.importStep(filename)
        except Exception as e:
            print(f"An unexpected error occurred with STEP import:\n{e}")
       
        # BASIC GEOMETRIC PROPERTIES
        self.V = self.data.val().Volume()  # volume
        self.SA = self.data.val().Area()  # surface area
       
        # BOUNDING BOX CALCULATION
        bounding_box = self.data.val().BoundingBox()
        self.BBX = bounding_box.xlen  # x dim of bounding box
        self.BBY = bounding_box.ylen  # x dim of bounding box
        self.BBZ = bounding_box.zlen  # x dim of bounding box
        self.BBV = (
            bounding_box.xlen * bounding_box.ylen * bounding_box.zlen
        )  # volume of bounding box
        self.BBCX = self.data.val().CenterOfBoundBox().x # center of bounding box
        self.BBCY = self.data.val().CenterOfBoundBox().y # center of bounding box
        self.BBCZ = self.data.val().CenterOfBoundBox().z # center of bounding box
       
        # CENTER OF MASS
        self.CX = self.data.val().Center().x
        self.CY = self.data.val().Center().y
        self.CZ = self.data.val().Center().z
       
        # ADVANCED GEOMETRIC PROPERTIES
        self.NV = len(self.data.val().Vertices())  # number of vertices
        self.NE = len(self.data.val().Edges())  # number of edges
        self.NF = len(self.data.val().Faces())  # number of faces
