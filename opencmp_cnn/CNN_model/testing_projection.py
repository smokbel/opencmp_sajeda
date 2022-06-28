import sys, json, time
from typing import Dict, Optional, cast
import os.path
from opencmp.models import get_model_class
from opencmp.solvers import get_solver_class
from opencmp.helpers.error_analysis import h_convergence, p_convergence
from opencmp.helpers.error import calc_error
from opencmp.config_functions import ConfigParser
from opencmp.helpers import error
from opencmp_cnn.config_functions import nn_to_gfu, sol_to_data
import pyngcore as ngcore
from ngsolve import *
from ngsolve import ngsglobals, Mesh
from ngsolve.comp import GridFunction
from opencmp.helpers.post_processing import sol_to_vtu, PhaseFieldModelMimic
from autoencode import NeuralNet
import torch
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
from opencmp.helpers.saving import SolutionFileSaver
from opencmp.config_functions import BCFunctions

config_file_path = sys.argv[1]
config = ConfigParser(config_file_path)

run_dir = config.get_item(['OTHER', 'run_dir'], str)
element = config.get_dict(['FINITE ELEMENT SPACE', 'elements'], run_dir, None, all_str=True)
#print(element)

interp_ord = config.get_item(['FINITE ELEMENT SPACE', 'interpolant_order'], int)
mesh_file = config.get_item(['MESH', 'filename'], str)
mesh = Mesh(mesh_file)

n = 64
m = 256


#print(meshel.vertices)


# Load the CNN and its parameters.
# model = NeuralNet()
# model = model.double()
# param_file = 'CNN_model/param.json'

sol_init_str = '../../examples/ins_sajeda_test/output/sol/ins_10.sol'
sol_final_str = '../../examples/ins_sajeda_test/output/sol/ins_298.sol'

#mesh2 = Mesh('../../examples/ins_sajeda/2d_ell_r.vol')

gfu_init = sol_to_data.create_gfu(poly=interp_ord, mesh=mesh, element=element, sol_path_str=sol_init_str)
gfu_fin = sol_to_data.create_gfu(poly=interp_ord, mesh=mesh, element=element, sol_path_str=sol_final_str)
np_data = sol_to_data.create_np_data(gfu=gfu_fin, mesh=mesh, n=n, m=m, mask=False, interp_ord=3)
point_map_u = np_data[1]
point_map_p = np_data[2]
gfuir_coor_u = np_data[3]
gfuir_coor_p = np_data[4]

print("Length irs ", len(gfuir_coor_u.components[0].vec))
print("Length irs ", len(gfuir_coor_u.components[1].vec))
print("Length irs ", len(gfuir_coor_p.vec))
# lele = np_data
#
# new = np_data[0]
#
# vis = sns.heatmap(new[0][0], vmin=np.min(new[0][0]), vmax=np.max(new[0][0]))
# plt.show()
#
# vis = sns.heatmap(new[0][1], vmin=np.min(new[0][1]), vmax=np.max(new[0][1]))
# plt.show()
#
# vis = sns.heatmap(new[0][2], vmin=np.min(new[0][2]), vmax=np.max(new[0][2]))
# plt.show()
#
#
# gfu_from_nn = nn_to_gfu.numpy_to_gfu(np_data[0][0], config_file_path, gfu_init, mesh, n, m, point_map_u, point_map_p,
#                                      gfuir_coor_u, gfuir_coor_p, gfu_fin)
# gfu_from_nn.Save("../../examples/ins_sajeda_test/output/sol/ins_350.sol")
# i = 0
#
# fes_err = H1(mesh, order=1)
# gfu_err = GridFunction(fes_err)
# i = 0

# def mag(input):
#     return math.sqrt((input[0]+input[1])**2)
#
# for p in mesh.vertices:
#     diff = mag(gfu_from_nn.components[0](mesh(*p.point))) - mag(gfu_fin.components[0](mesh(*p.point)))
#     #print("Projected: ", gfu_from_nn.components[0](mesh(*p.point))[0], "Actual: ", gfu_fin.components[0](mesh(*p.point))[0])
#     gfu_err.vec[i] = abs(diff)
#     i += 1
#     # if abs(diff) >= 0.00001:
#     #     print("Too high of a difference for these points: ", p.point)
#
# #err = error.norm("l2_norm",gfu_from_nn.components[0], gfu_from_nn.components[0], mesh, element, average=False)
# # vtk = VTKOutput(ma=mesh,
# #                 coefs=[gfu_err],
# #                 names = ["diff"],
# #                 filename="result",
# #                 subdivision=3)
#
# vtk = VTKOutput(ma=mesh,
#                 coefs=[gfu_from_nn.components[0], gfu_from_nn.components[1]],
#                 names = ["u", "p"],
#                 filename="projected_gfu1",
#                 subdivision=3)
# vtk.Do()





