from ngsolve import *
from ngsolve.comp import IntegrationRuleSpace
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ngsolve as ngs
import os
from pathlib import Path
import json
from opencmp.config_functions import expanded_config_parser

def data_gen(data_folder: str, n: int):
    """
        Function to create gridfunction from neural network output.

        Arguments:
            np_data: The neural network output, which is in the form of a numpy array.
            config_file: The config file for the simulation.
            gfu_init: The initial gridfunction where the simulation last left off.
    """

    dir_path = '../../examples/ins_sajeda_test'
    config_file = dir_path + '/config'
    #config_file = '../examples/ins_sajeda_test/config'
    config = expanded_config_parser.ConfigParser(config_file)

    # Get the run directory and the model.
    run_dir = config.get_item(['OTHER', 'run_dir'], str)


    # Load the finite element space parameters.
    element = config.get_dict(['FINITE ELEMENT SPACE', 'elements'], run_dir, None, all_str=True)
    interp_ord = config.get_item(['FINITE ELEMENT SPACE', 'interpolant_order'], int)

    # Create IntegrationRuleSpaces for each element.
    # Each element will need two IRS: One to interpolate the Gridfunction data to, another to get the coordinates from.


    idx = 0 # Index in Gridfunction components.

    m = 8
    np_arr = np.empty((1170,m,n,1))

    f = 0
    p = Path(data_folder)
    subdirectories = [x for x in p.iterdir() if x.is_dir()]
    for case in subdirectories:
        jsonfile = str(case) + '/info.json'
        with open(jsonfile) as json_file:
            param = json.load(json_file)

        # Get the parameters needed to make data for specific case.
        target_file = param["target_sol"]
        sol_file_target = os.path.join(case, target_file)
        total_it = int(param["num_it"])
        uref = float(param["uref"])
        mesh_file = param["mesh_file"]
        mesh_path = dir_path + "/" + mesh_file
        mesh = Mesh(mesh_path)

        fes_l = []
        for field in element:
            V = getattr(ngs, element[field])
            if field == 'p':
                interp_ord_p = interp_ord - 1
                fes_e = V(mesh=mesh, order=interp_ord_p, dirichlet="wall|cyl|inlet")
            else:
                fes_e = V(mesh=mesh, order=interp_ord, dirichlet="wall|cyl|inlet")
            fes_l.append(fes_e)

        fes = FESpace(fes_l)

        for filename in os.listdir(case):
            if filename not in ['info.json', target_file]:
                sol_file = os.path.join(case, filename)

                # Load sol file as a Gridfunction.
                gfu_init = GridFunction(fes)
                gfu_init.Load(sol_file)
                print("Size of weights vector: ", len(gfu_init.vec))
                print("Values ", gfu_init.vec[0:10])

                gfu_final = GridFunction(fes)
                gfu_final.Load(sol_file_target)
                print("Size of weights vector final: ", len(gfu_final.vec))
                print("Values ", gfu_final.vec[0:10])



                dim = gfu_init.components[idx].dim

                # Working with 2d data - the fesir coordinate data always has dimension 2.
                fesir_coor = IntegrationRuleSpace(mesh=mesh, order=interp_ord, dirichlet="wall|inlet|cyl") ** 2

                # Create IRS Gridfunction for data and for coordinates.
                gfuir_coor = GridFunction(fesir_coor)
                gfuir_coor.Interpolate(CF((x, y)))

                # Coordinates from IntegrationRuleSpace.
                coord_x = gfuir_coor.components[0]
                coord_y = gfuir_coor.components[1]

                # Create intermediate and final (target) data.
                z = []
                for i in range(n):
                    p1 = coord_x.vec[i]
                    p2 = coord_y.vec[i]
                    np_arr[f, 0, i, 0] = gfu_init.components[idx](mesh(p1,p2))[0] / uref # Ux
                    z.append(gfu_init.components[idx](mesh(p1,p2))[0] / uref)
                    np_arr[f, 1, i, 0] = gfu_init.components[idx](mesh(p1, p2))[1] / uref # Uy
                    np_arr[f, 2, i, 0] = p1 # x coordinate
                    np_arr[f, 3, i, 0] = p2 # y coordinate
                    np_arr[f, 4, i, 0] = gfu_final.components[idx](mesh(p1, p2))[0] / uref# Ux
                    np_arr[f, 5, i, 0] = gfu_final.components[idx](mesh(p1, p2))[1] / uref# Uy
                    np_arr[f, 6, i, 0] = p1  # x coordinate
                    np_arr[f, 7, i, 0] = p2  # y coordinate
                f += 1
                print(f)
        tot = len(coord_x.vec)
        print(tot, mesh_path)
        print("Intermediate: ", np_arr[10,0,45:55,0])
        print("Final: ", np_arr[10, 4, 45:55, 0])
        plt.scatter(coord_x.vec[0:tot], coord_y.vec[0:tot], s=0.0001, c="g", marker="*")
        plt.show()
    return np_arr



#
np_data = data_gen('../../../../cnn_data_case1', 12000)
np.save('./data_fc_test', np_data)

#data = np.load('./data_fc.npy')
#
# print(len(data[0][0]))













