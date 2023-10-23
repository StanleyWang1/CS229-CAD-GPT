import cadquery as cq
import sys

def main():
    if len(sys.argv) != 5:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")
    x_dim = sys.argv[1]
    y_dim = sys.argv[2]
    z_dim = sys.argv[3]
    filename = sys.argv[4]
    box = cq.Workplane().box(int(x_dim), int(y_dim), int(z_dim))
    cq.exporters.export(box, filename)

if __name__ == '__main__':
    main()