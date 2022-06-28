import sys
import netgen.geom2d as ngeo
from netgen.libngpy._geom2d import Solid2d

from opencmp.config_functions import expanded_config_parser
import netgen.geom2d as ngeo
import netgen

geo = ngeo.SplineGeometry()
config_file = sys.argv[1]
config = expanded_config_parser.ConfigParser(config_file)

# Get config params
x_0 = config.get_item(['RECTANGLE', 'x_0'], float)
y_0 = config.get_item(['RECTANGLE', 'y_0'], float)
x_n = config.get_item(['RECTANGLE', 'x_n'], float)
y_n = config.get_item(['RECTANGLE', 'y_n'], float)

print(x_0, y_0, x_n, y_n)
obj_type = config.get_item(['IMMERSED_OBJ', 'type'], str)
cx = config.get_item(['IMMERSED_OBJ', 'cx'], float)
cy = config.get_item(['IMMERSED_OBJ', 'cy'], float)
save_file_vol = config.get_item(['FILE', 'save_file_vol'], str)
#save_file_msh = config.get_item(['FILE', 'save_file_msh'], str)
maxh_walls = config.get_item(['MESH', 'maxh_walls'], float)
maxh_obj = config.get_item(['MESH', 'maxh_obj'], float)
maxh_global = config.get_item(['MESH', 'maxh_global'], float)

def MakeEllipse (geo, c, r, **args):
    cx,cy = c
    rx,ry = r
    pts = [geo.AppendPoint(*p) for p in [(cx,cy-ry), (cx+rx,cy-ry), (cx+rx,cy), (cx+rx,cy+ry),
                                         (cx,cy+ry), (cx-rx,cy+ry), (cx-rx,cy), (cx-rx,cy-ry)]]
    for p1,p2,p3 in [(0,1,2), (2,3,4), (4, 5, 6), (6, 7, 0)]:
        geo.Append( ["spline3", pts[p1], pts[p2], pts[p3]], **args)

if obj_type == 'cylinder':
    geo.AddRectangle((x_0, y_0), (x_n, y_n), bcs=("wall", "outlet", "wall", "inlet"), maxh=maxh_walls)
    cyl_radius = config.get_item(['IMMERSED_OBJ', 'radius'], float)
    geo.AddCircle((cx,cy), r=cyl_radius, leftdomain=0, rightdomain=1, bc="cyl", maxh=maxh_obj)
    #geo.AddCircle((cx+0.10, cy), r=cyl_radius, leftdomain=0, rightdomain=1, bc="cyl", maxh=maxh_obj)

if obj_type == 'ellipse':
    geo.AddRectangle((x_0, y_0), (x_n, y_n), bcs=("wall", "outlet", "wall", "inlet"), maxh=maxh_walls)
    #geo.AddRectangle((0.0, 0.4), (4.0, 0.6), bcs=("wall", "outlet", "wall", "inlet"), maxh=0.02)
    #cyl_radius = config.get_item(['IMMERSED_OBJ', 'radius'], float)
    #Make_Circle(geo, (0.5,0.5), n0.1, leftdomain=0, rightdomain=1,bc="cyl")
    MakeEllipse(geo, (cx, cy), (0.22, 0.121),leftdomain=0, rightdomain=1,bc="cyl",maxh=maxh_obj)
    #p1, p2, p3, p4 = [geo.AppendPoint(*p) for p in [(0,0.55), (4.0,0.55), (0, 0.40), (4.0, 0.40)]]
    #geo.Append(["line", p1,p2],leftdomain=0, rightdomain=1, bc="cyl")
    # geo.Append(["line", p3,p4],leftdomain=0, rightdomain=1,bc="cyl",maxh=0.04)

mesh = geo.GenerateMesh(maxh=maxh_global)
mesh.Save(save_file_vol)
#mesh.Export(save_file_msh,"Gmsh2 Format")