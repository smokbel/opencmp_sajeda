
import netgen.gui
from netgen.geom2d import unit_square
from ngsolve import *
import ngsolve as ngs
#from config_functions.sol_to_data import create_gfu
from opencmp.config_functions import BCFunctions
from opencmp.config_functions import ConfigParser
import sys

config_file_path = '../examples/ins_sajeda/config'
# config = ConfigParser(config_file_path)
#
# run_dir = config.get_item(['OTHER', 'run_dir'], str)
# element = config.get_dict(['FINITE ELEMENT SPACE', 'elements'], run_dir, None, all_str=True)
#
# interp_ord = config.get_item(['FINITE ELEMENT SPACE', 'interpolant_order'], int)
# mesh_file = config.get_item(['MESH', 'filename'], str)
# mesh = Mesh(mesh_file)

mesh = Mesh('../examples/ins_sajeda/2d_ell_55.vol')
fe_u = HDiv(mesh=mesh, order=2)
fe_p = L2(mesh=mesh, order=1)
fes = ngs.comp.FESpace([fe_u, fe_p])
n = 64
m = 128
gfu = ngs.GridFunction(fes)
gfu.Load('predicted_gfu.sol')
save_name = 'predict'
if len(gfu.components) > 0:
        coefs = [component for component in gfu.components]

    # Write to .vtk
VTKOutput(ma=mesh,
    coefs=coefs,
              names=['u', 'p'],
              filename='predict',
              subdivision=3).Do()

#gfu = create_gfu(poly=interp_ord, mesh=mesh, element=element, sol_path_str='predicted_gfu.sol')


