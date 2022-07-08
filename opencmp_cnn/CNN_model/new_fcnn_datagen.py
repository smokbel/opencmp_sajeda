from ngsolve import *
from ngsolve.comp import IntegrationRuleSpace
import numpy as np
import matplotlib.pyplot as plt
import ngsolve as ngs
import os
from pathlib import Path
import json
from opencmp.config_functions import expanded_config_parser
import matplotlib.pyplot as plt


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

    print("Interpolating order: ", interp_ord)
    print("Element: ", element)



    # Create IntegrationRuleSpaces for each element.
    # Each element will need two IRS: One to interpolate the Gridfunction data to, another to get the coordinates from.


    idx = 0 # Index in Gridfunction components.

    m = 8
    np_arr = np.empty((85, m, 22224, 1))


    f = 0
    p = Path(data_folder)
    subdirectories = [x for x in p.iterdir() if x.is_dir()]
    print(subdirectories)

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

        print("Mesh file: ", mesh_file)

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
            print("LENGTH OF ", len(os.listdir(case)), os.listdir(case))
            print("MESH PATH ", mesh_path)
            mesh_path_dir = os.path.join(case, mesh_file)
            print(mesh_path_dir)
            if filename not in ['info.json', target_file, mesh_path, 'channel_ell_055.vol']:
                sol_file = os.path.join(case, filename)

                print("Working on file: ", sol_file)
                # Load sol file as a Gridfunction.
                gfu_init = GridFunction(fes)
                gfu_init.Load(sol_file)

                gfu_final = GridFunction(fes)
                gfu_final.Load(sol_file_target)

                #dim = gfu_init.components[idx].dim

                # Working with 2d data - the fesir coordinate data always has dimension 2.
                fesir_coor = IntegrationRuleSpace(mesh=mesh, order=interp_ord, dirichlet="wall|inlet|cyl") ** 2

                # Create IRS Gridfunction for data and for coordinates.
                gfuir_coor = GridFunction(fesir_coor)
                gfuir_coor.Interpolate(CF((x, y)))

                # Coordinates from IntegrationRuleSpace.
                coord_x = gfuir_coor.components[0].vec.FV().NumPy()
                coord_y = gfuir_coor.components[1].vec.FV().NumPy()

                # Create intermediate and final (target) data.
                # First, order points so the data makes sense spatially.
                # sorted_coord_x = coord_x.sort()
                # sorted_coord_y = coord_y.sort()

                coords = [(coord_x[i], coord_y[i]) for i in range(len(coord_x))]
                coords = np.array(coords)
                ind = np.lexsort((coords[:, 1], coords[:, 0]))
                sorted_points = coords[ind]

                z = []

                for i in range(n):
                    #p1, p2 = sorted_coord_x[i], sorted_coord_y[i]
                    p1,p2 = sorted_points[i,0], sorted_points[i,1]
                    np_arr[f, 0, i, 0] = gfu_init.components[0](mesh(p1, p2))[0]  # Ux
                    z.append(gfu_init.components[0](mesh(p1, p2))[0] )
                    np_arr[f, 1, i, 0] = gfu_init.components[0](mesh(p1, p2))[1]  # Uy
                    #print("Val u is: ", gfu_init.components[idx](mesh(p1, p2)) )
                    np_arr[f, 2, i, 0] = p1  # x coordinate
                    np_arr[f, 3, i, 0] = p2  # y coordinate
                    np_arr[f, 4, i, 0] = gfu_final.components[idx](mesh(p1, p2))[0]   # Ux
                    np_arr[f, 5, i, 0] = gfu_final.components[idx](mesh(p1, p2))[1]   # Uy
                    np_arr[f, 6, i, 0] = p1  # x coordinate
                    np_arr[f, 7, i, 0] = p2  # y coordinate



                if filename == 'ins_20.sol':
                    max_ux = np.max(np_arr[f, 0, :, 0])
                    max_uy = np.max(np_arr[f,1,:,0])
                    print("Max ux: ", max_ux)
                    print("Max uy: ", max_uy)
                    # print("Arr representing ux ", np_arr[f, 0, :, 0])
                    # print("Arr representing uy ", np_arr[f,1,:,0])
                f += 1
        print("Length of sorted points is: ", len(sorted_points))
        print("NP max before dividing: ", np.max(np_arr[0:80,0,:,0]))
        print(" NP max for ux: ", max_ux)
        np_arr[0:80,0,:,0] = np_arr[0:80,0,:,0] / max_ux
        np_arr[0:80,1,:,0] = np_arr[0:80,1,:,0] / max_uy
        np_arr[0:80, 4, :, 0] = np_arr[0:80, 4, :, 0] / max_ux
        np_arr[0:80, 5, :, 0] = np_arr[0:80, 5, :, 0] / max_uy
        print("Np max is: ", np.max(np_arr[0:80,0,:,0]))
        plt.scatter(sorted_points[0:n, 0], sorted_points[0:n, 1], marker="*", s=0.05, alpha=0.8, c=z,
                    cmap='plasma')
        plt.show()



    return np_arr






#
np_data = data_gen('./cases', 20000)
np.save('./data_fc_test', np_data)

#data = np.load('./data_fc.npy')
#
# print(len(data[0][0]))













