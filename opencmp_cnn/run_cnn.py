########################################################################################################################
# Copyright 2021 the authors (see AUTHORS file for full list).                                                         #
#                                                                                                                      #
# This file is part of OpenCMP.                                                                                        #
#                                                                                                                      #
# OpenCMP is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public  #
# License as published by the Free Software Foundation, either version 2.1 of the License, or (at your option) any     #
# later version.                                                                                                       #
#                                                                                                                      #
# OpenCMP is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied        #
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more  #
# details.                                                                                                             #
#                                                                                                                      #
# You should have received a copy of the GNU Lesser General Public License along with OpenCMP. If not, see             #
# <https://www.gnu.org/licenses/>.                                                                                     #
########################################################################################################################

import sys, json, time
from typing import Dict, Optional, cast
import os.path
from opencmp.models import get_model_class
from opencmp.solvers import get_solver_class
from opencmp.helpers.error_analysis import h_convergence, p_convergence
from opencmp.helpers.error import calc_error
from opencmp.config_functions import ConfigParser
from config_functions import nn_to_gfu, sol_to_data
import pyngcore as ngcore
from ngsolve import ngsglobals, Mesh
from ngsolve.comp import GridFunction
from opencmp.helpers.post_processing import sol_to_vtu, PhaseFieldModelMimic
from CNN_model.autoencode import NeuralNet
import torch
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from opencmp.helpers.saving import SolutionFileSaver
from opencmp.config_functions import BCFunctions
import fileinput


config_file_path = sys.argv[1]
config = ConfigParser(config_file_path)

run_dir = config.get_item(['OTHER', 'run_dir'], str)
element = config.get_dict(['FINITE ELEMENT SPACE', 'elements'], run_dir, None, all_str=True)

interp_ord = config.get_item(['FINITE ELEMENT SPACE', 'interpolant_order'], int)
mesh_file = config.get_item(['MESH', 'filename'], str)
mesh = Mesh(mesh_file)

n = 64
m = 256

# Load the CNN and its parameters.
model = NeuralNet()
model = model.double()
param_file = 'CNN_model/param.json'

with open(param_file) as json_file:
    param = json.load(json_file)

optimizer = optim.RMSprop(model.parameters(), lr=param['learning_rate'])
model_save = 'CNN_model/' + param['model_save']

model.load_state_dict(torch.load(str(model_save)))
model.eval()
model = model.double()

saver = SolutionFileSaver


def run_cnn(config_file_path: str, tol: list, initial_condition: Optional[GridFunction] = None, config_parser: Optional[ConfigParser] = None) -> None:
    """
    Main function that runs OpenCMP. Same as the main run.py function, except returns the Gridfunction.

    Args:
        config_file_path: Filename of the config file to load.
        config_parser: Optionally provide the ConfigParser if running tests.

    Returns:
        Gridfunction storing the solution.
    """
    # Load the config_parser file.
    if config_parser is None:
        config_parser = ConfigParser(config_file_path)
    else:
        assert config_parser is not None
        config_parser = cast(ConfigParser, config_parser)

    # Load run parameters from the config_parser file.
    num_threads = config_parser.get_item(['OTHER', 'num_threads'], int)
    msg_level = config_parser.get_item(['OTHER', 'messaging_level'], int, quiet=True)
    model_name = config_parser.get_item(['OTHER', 'model'], str)

    # Load error analysis parameters from the config_parser file.
    check_error = config_parser.get_item(['ERROR ANALYSIS', 'check_error'], bool)

    # Set parameters for ngsolve
    ngcore.SetNumThreads(num_threads)
    ngsglobals.msg_level = msg_level

    # Run the model.
    with ngcore.TaskManager():

        model_class = get_model_class(model_name)
        solver_class = get_solver_class(config_parser)
        solver = solver_class(model_class, config_parser)

        # The tolerance will be passed as an adjustabale parameter.
        solver.model.rel_nonlinear_tolerance = tol[0]
        solver.model.abs_nonlinear_tolerance = tol[1]

        if initial_condition is not None:
            solver.model.IC = initial_condition

        sol = solver.solve()
        #solver.model


        if check_error:
            calc_error(config_parser, solver.model, sol)

        # Suppressing the warning about using the default value for convergence_test.
        convergence_test: Dict[str, str] = config_parser.get_dict(['ERROR ANALYSIS', 'convergence_test'],
                                                                  None, quiet=True)
        for key, var_lst in convergence_test.items():
            if key == 'h' and var_lst:
                for var in var_lst:
                    h_convergence(config_parser, solver, sol, var)
            elif key == 'p' and var_lst:
                for var in var_lst:
                    p_convergence(config_parser, solver, sol, var)

    save_output = config_parser.get_item(['VISUALIZATION', 'save_to_file'], bool, quiet=True)
    if save_output:
        save_type = config_parser.get_item(['VISUALIZATION', 'save_type'], str, quiet=True)

        # Run the post-processor to convert the .sol to .vtu
        if save_type == '.vtu':
            print('Converting saved output to VTU.')

            # Path where output is stored
            output_dir_path = config_file_path.replace('config', '') + '/output/'

            #output_dir_path = config_parser.get_item(['OTHER', 'run_dir'], str) + '/output/'

            # Run the conversion
            sol_to_vtu(config_parser, output_dir_path, solver.model)

            # Repeat for the saved phase field .sol files if using the diffuse interface method.
            if solver.model.DIM:
                print('Converting saved phase fields to VTU.')

                # Construct a mimic of the Model class appropriate for the phase field (mainly contains the correct
                # finite element space).
                phi_model = PhaseFieldModelMimic(solver.model)

                # Path where the output is stored
                output_dir_phi_path = config_parser.get_item(['OTHER', 'run_dir'], str) + '/output_phi/'

                # Run the conversion.
                # Note: The normal main simulation ConfigParse can be used since it is only used
                # to get a value for subdivision.
                sol_to_vtu(config_parser, output_dir_phi_path, phi_model)

    return sol

def full_cnn_run(config_file_path: str, config_parser: Optional[ConfigParser] = None):
    # Start the timer.
    time_start = time.perf_counter()

    # Run the simulation for high tolerance, and obtain the intermediate Gridfunction.
    gfu_init = run_cnn(config_file_path=config_file_path, tol=[0.05, 0.05])

    #Convert the intermediate gridfunction to numpy array.
    #gfu_init_np = sol_to_data.create_np_data(gfu_init, mesh, n, m, True)

    # Get the min and max values.

    # init_min_uy, init_max_uy = np.amin(gfu_init_np[0,1]), np.amax(gfu_init_np[0,1])
    # init_min_p, init_max_p = np.amin(gfu_init_np[0,2]), np.amax(gfu_init_np[0,2])

    # Normalize the data, and put it through the neural network.

    # gfu_init_nn = sol_to_data.normalize(gfu_init_np, True, gfu_np_fill, False, 1, 0)
    # gfu_init_nn[0,0,:,:] = sol_to_data.apply_mask(gfu_init_nn[0,1,:,:])

    gfu_init_cn = sol_to_data.create_nn_data(gfu=None, mesh=mesh, poly=interp_ord, element=element, sample_n=5, n=64, m=256, target_file=None, sol_dir='../examples/ins_sajeda/output/sol/')
    gfu_init_np = gfu_init_cn
    winit_min_ux, winit_max_ux = np.amin(gfu_init_np[0, 0]), np.amax(gfu_init_np[0, 0])
    winit_min_uy, winit_max_uy = np.amin(gfu_init_np[0, 1]), np.amax(gfu_init_np[0, 1])
    winit_min_p, winit_max_p = np.amin(gfu_init_np[0, 2]), np.amax(gfu_init_np[0, 2])
    init_min_ux, init_max_ux = -0.659199759129811, 3.0583608657642967
    init_min_uy, init_max_uy = -1.4528969066355684, 1.4528969066355684
    init_min_p, init_max_p = -1.2052483770849363, 3.9305380672066534
    gfu_np_fill = np.empty((len(gfu_init_cn), 4, n, m))
    gfu_init_nn = sol_to_data.normalize(gfu_init_cn, True, gfu_np_fill, False, 1, 0)
    np.save('from_sim', gfu_init_nn)
    i = 0
    while i < 1:
        vis = sns.heatmap(gfu_init_nn[i][0], vmin=np.min(gfu_init_nn[i][0]), vmax=np.max(gfu_init_nn[i][0]))
        plt.show()

        vis = sns.heatmap(gfu_init_nn[i][1], vmin=np.min(gfu_init_nn[i][1]), vmax=np.max(gfu_init_nn[i][1]))
        plt.show()

        vis = sns.heatmap(gfu_init_nn[i][2], vmin=np.min(gfu_init_nn[i][2]), vmax=np.max(gfu_init_nn[i][2]))
        plt.show()

        vis = sns.heatmap(gfu_init_nn[i][3], vmin=np.min(gfu_init_nn[i][3]), vmax=np.max(gfu_init_nn[i][3]))
        plt.show()

        i += 1

    input_nn = torch.from_numpy(gfu_init_nn[0:1,:,:,:])
    #data = np.load('data/data_apr20.npy')
    #in_value = data[8:9, :, :, :]
    #in_model = torch.from_numpy(input_nn)
    # Put the np array through the CNN model and get an output.
    prediction_in = model.forward(input_nn)
    prediction = prediction_in.detach().numpy()

    vis = sns.heatmap(prediction[0][0], vmin=np.min(prediction[0][0]), vmax=np.max(prediction[0][0]))
    plt.show()

    vis = sns.heatmap(prediction[0][1], vmin=np.min(prediction[0][1]), vmax=np.max(prediction[0][1]))
    plt.show()

    vis = sns.heatmap(prediction[0][2], vmin=np.min(prediction[0][2]), vmax=np.max(prediction[0][2]))
    plt.show()


    # Unnormalize the data.
    #gfu_np_fill = np.empty(np.shape(gfu_init_np))
    #gfu_to_solver = sol_to_data.normalize(prediction, run_dir, True, gfu_np_fill, True)
    prediction[0,0] = sol_to_data.unnormalize(prediction[0,0], init_min_ux, init_max_ux)
    prediction[0,1] = sol_to_data.unnormalize(prediction[0,1], init_min_uy, init_max_uy)
    prediction[0,2] = sol_to_data.unnormalize(prediction[0,2], init_min_p, init_max_p)


    # vis = sns.heatmap(prediction[0][0],vmin=np.min(prediction[0][0]),vmax=np.max(prediction[0][0]))
    # plt.show()
    #
    # vis = sns.heatmap(prediction[0][1],vmin=np.min(prediction[0][1]),vmax=np.max(prediction[0][1]))
    # plt.show()
    #
    # vis = sns.heatmap(prediction[0][2],vmin=np.min(prediction[0][2]),vmax=np.max(prediction[0][2]))
    # plt.show()
    time_end = time.perf_counter()
    print("Time", time_end - time_start)

    # Convert back to a gridfunction, and put it back into the solver as the initial condition.
    gfu_from_nn = nn_to_gfu.numpy_to_gfu(prediction[0], config_file_path, gfu_init, mesh, n, m)
    #run_cnn(config_file_path=config_file_path, tol=[0.000001, 0.000001], initial_condition=gfu_from_nn)
    gfu_from_nn.Save('predicted_gfu.sol')

    # Rerun the solver, with new gfu as initial condition, with low tolerance.
    # gfu_fin = run_cnn(config_file_path, [0.000001, 0.000001], gfu_from_nn)
    #
    time_end = time.perf_counter()
    print("Time", time_end - time_start)
    print("Min max should be", init_min_uy, init_max_uy)
    print("Min max is", winit_min_uy, winit_max_uy)


#Run the simulation, and obtain thse intermediate Gridfunction.
# sajeda@sajeda: This will be fixed this week to account for only one config file.


# Get numpy data from gfu.
# data_for_nn = sol_to_data.create_np_data(gfu=gfu_init, mesh=mesh, n=n, m=m)
#
# # Put the numpy array into the CNN model.
# model = NeuralNet()
# model = model.double()
# param_file = sys.argv[1]
#
# with open(param_file) as json_file:
#     param = json.load(json_file)
#
# optimizer = optim.RMSprop(model.parameters(), lr=param['learning_rate'])
# model_save = param['model_save']
#
# model.load_state_dict(torch.load(str(model_save)))
# model.eval()
# model = model.double()
#
# prediction = model.forward(data_for_nn)
# prediction = prediction.detach().numpy()
#
# # Back to opencmp to check error.
# gfu_fin = nn_to_gfu.numpy_to_gfu(np_data=prediction, config_file=config_file_path,
#                                  gfu_init=gfu_init, mesh=mesh, n=n, m=m)

# use solverclass object and reintitialize the initial condition



if __name__ == '__main__':

    if len(sys.argv) == 1:
        print("ERROR: Provide configuration file path.")
        exit(0)

    full_cnn_run(config_file_path)

    # time_start = time.perf_counter()
    # # Run the simulation, and obtain thse intermediate Gridfunction.
    # # sajeda@sajeda: This will be fixed this week to account for only one config file.
    # gfu_init = run_cnn(config_file_path, [0.01, 0.01])
    # time_end = time.perf_counter()
    # print("Time", time_end - time_start)
# gfu_fin = run_cnn(config_file_path)
# #gfu_init = sol_to_data.create_gfu(poly=interp_ord, mesh=mesh, element=element,
#                                   #sol_path_str='examples/tutorial_4/output/sol/stokes_0.1.sol')
#
# Get dimensions.

#
# # Put into trained nn model and get output.
# ## This step to complete....
#
# #gfu_fin = sol_to_data.create_gfu(poly=3, mesh=mesh, element=element, sol_path_str='examples/tutorial_4/output/sol/stokes_0.2.sol')
# data_from_nn = sol_to_data.create_np_data(gfu=gfu_fin, mesh=mesh, n=n, m=m)
#
# # Put back as a Gridfunction.
# gfu_fin = nn_to_gfu.numpy_to_gfu(np_data=data_from_nn, config_file=config_file_path,
#                                  gfu_init=gfu_init, mesh=mesh, n=n, m=m)
#
# gfu_init = sol_to_data.create_gfu(poly=interp_ord, mesh=mesh, element=element, sol_path_str='examples/tutorial_4/output/sol/stokes_1.8.sol')
#
#
#
# data = sol_to_data.create_np_data(gfu_init, mesh, n, m)
#
# data3 = sol_to_data.create_np_data(gfu_fin, mesh, n, m)
# ux_gfu = data3[0]
# uy_gfu = data3[1]
# p_gfu = data3[2]
#
# vis =sns.heatmap(p_gfu,vmin=np.min(p_gfu),vmax=np.max(p_gfu))
# plt.show()
# vis =sns.heatmap(data[2],vmin=np.min(data[2]),vmax=np.max(data[2]))
# plt.show()
#
# vis =sns.heatmap(ux_gfu,vmin=np.min(ux_gfu),vmax=np.max(ux_gfu))
# plt.show()
# vis =sns.heatmap(data[0],vmin=np.min(data[0]),vmax=np.max(data[0]))
# plt.show()
#
# vis =sns.heatmap(uy_gfu,vmin=np.min(uy_gfu),vmax=np.max(uy_gfu))
# plt.show()
# vis =sns.heatmap(data[1],vmin=np.min(data[1]),vmax=np.max(data[1]))
# plt.show()
#
#
#
# run
