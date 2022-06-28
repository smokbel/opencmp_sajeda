import sys, os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import ngsolve as ngs
from typing import Optional
import re
from ngsolve import *
from ngsolve.comp import IntegrationRuleSpace
from opencmp.config_functions import expanded_config_parser
from opencmp.config_functions import ConfigParser

def create_gfu(poly: int, mesh: ngs.Mesh, element: dict, sol_path_str: Optional[str] = None):
    """
        Function to create Gridfunction from simulation data.

        Args:
            poly: The interpolant order, obtained from config file.
            mesh: The mesh used for the simulation.
            element: The finite element space elements obtained from config file.

        Returns:
            Gridfunction.
    """

    # Generate Finite Element Space (FES) from the Finite Elements defined in simulation.
    fe_elements = []

    for field in element:
        V = getattr(ngs, element[field])
        if field == 'p':
            # For pressure space, it will be one polynomial degree less than V.
            poly -= 1
        fe = V(mesh=mesh, order=poly)
        fe_elements.append(fe)

    if len(fe_elements) == 1:
        # If finite element space is not mixed, and only contains one element, pass element directly.
        fes = ngs.comp.FESpace(fe_elements[0])
    else:
        fes = ngs.comp.FESpace(fe_elements)

    # Create Gridfunction from Finite Element Space.
    gfu = ngs.GridFunction(fes)

    # Load .sol data into Gridfunction.
    if sol_path_str is not None:
        gfu.Load(sol_path_str)

    return gfu

def create_np_data(gfu: ngs.GridFunction, mesh: ngs.Mesh, n: int, m: int, interp_ord: int, mask: Optional[bool]):
    """
        Function to create numpy array from Gridfunction data.

        Args:
            gfu_prof_components: The original gridfunction to load the values into.
            mesh: The mesh used in this simulation.
            n: Desired number of discrete cells (elements) in the uniform grid (numpy array)
               in the x direction.
            n: Desired number of discrete cells (elements) in the uniform grid (numpy array)
               in the y direction.
            sol_path_str: Optional directory string for the .sol file.

        Returns:
            Numpy array of field data Ux, Uy, P.
    """

    # Obtain mesh dimensions from vertices.
    # nasser@sajeda: Improve in future
    vertices = mesh.vertices
    v1, v3 = vertices[0], vertices[2]
    x0, xn, y0, yn = v1.point[0], v3.point[0], v1.point[1], v3.point[1]

    # Initialize numpy arrays for field data.
    output_Ux = np.zeros((n,m))
    output_Uy = np.zeros((n,m))
    output_P = np.zeros((n,m))

    output_fields = np.empty((1, 3, n, m))

    # Create uniform grid points based on mesh dimensions.
    x_interp = np.linspace(x0,xn,m)
    y_interp = np.linspace(y0,yn,n)

    # Uniform indices.
    x_ind = np.arange(m)
    y_ind = np.arange(n)

    gfu_comp = gfu.components

    fesir_coor = IntegrationRuleSpace(mesh=mesh, order=interp_ord, dirichlet="wall|inlet|cyl") ** 2
    fesir_coor_p = IntegrationRuleSpace(mesh=mesh, order=interp_ord-1, dirichlet="wall|inlet|cyl") ** 2

    # Create IRS Gridfunction for data and for coordinates.
    gfuir_coor = ngs.GridFunction(fesir_coor)
    gfuir_coor.Interpolate(CF((x, y)))

    gfuir_coor_p = ngs.GridFunction(fesir_coor_p)

    coord_x_p = gfuir_coor_p.components[0]
    coord_y_p = gfuir_coor_p.components[1]

    point_map = []
    point_map_p = []

    # Coordinates from IntegrationRuleSpace.
    coord_x_u = gfuir_coor.components[0]
    coord_y_u = gfuir_coor.components[1]

    # x_idx = 0
    # for p1 in x_interp:
    #     y_idx = 0
    #     for p2 in y_interp:
    #         try:
    #             val_Ux = gfu_comp[0](mesh(p1, p2))[0]
    #             val_Uy = gfu_comp[0](mesh(p1, p2))[1]
    #             val_P = gfu_comp[1](mesh(p1, p2))
    #             output_P[y_idx, x_idx] = val_P
    #             output_Ux[y_idx, x_idx] = val_Ux
    #             output_Uy[y_idx, x_idx] = val_Uy
    #             # output_P[x_idx, y_idx] = val_P
    #             # output_Ux[x_idx, y_idx] = val_Ux
    #             # output_Uy[x_idx, y_idx] = val_Uy
    #         except Exception as ex:
    #             print(ex)
    #             pass
    #         y_idx += 1
    #     x_idx += 1

    ##########################
    for i in range(len(coord_x_u.vec)):
        p1 = coord_x_u.vec[i]
        p2 = coord_y_u.vec[i]

        # Get corresponding indices.
        x_idx = int(np.interp(p1, x_interp, x_ind))
        y_idx = int(np.interp(p2, y_interp, y_ind))

        print(p1, p2, " corresponding indices: ", x_idx, y_idx)

        try:
            val_Ux = gfu_comp[0](mesh(p1, p2))[0]
            #print(val_Ux, gfu_comp[0](mesh(p1, p2))[0])
            val_Uy = gfu_comp[0](mesh(p1, p2))[1]
            output_Ux[y_idx, x_idx] = val_Ux
            output_Uy[y_idx, x_idx] = val_Uy
            point_map.append([y_idx,x_idx])

        except Exception as ex:
            print(ex)
            pass

    for i in range(len(coord_x_p.vec)):
        p1 = coord_x_p.vec[i]
        p2 = coord_y_p.vec[i]

        # Get corresponding indices.
        x_idx = int(np.interp(p1, x_interp, x_ind))
        y_idx = int(np.interp(p2, y_interp, y_ind))

        try:
            val_P = gfu_comp[1](mesh(p1, p2))
            output_P[y_idx, x_idx] = val_P
            point_map_p.append([y_idx,x_idx])

        except Exception as ex:
            print(ex)
            pass

    ###########################3
    if mask is True:
        output_fields = np.empty((1, 4, n, m))
        output_fields[0, 0], output_fields[0, 1], output_fields[0, 2], output_fields[0,3] = apply_mask(output_Ux), output_Ux, output_Uy, output_P
        output_fields[0,0] = apply_mask(output_fields[0,1])
    else:
        output_fields[0,0], output_fields[0,1], output_fields[0,2] = output_Ux, output_Uy, output_P

    return output_fields, point_map, point_map_p, gfuir_coor, gfuir_coor_p


####################################################################################################################
# Generating CNN data format
####################################################################################################################


def apply_mask(arr: np.array):
    """
        Function that creates a geometry mask for the simulation.

        Args:
            arr:

        Returns:
            Numpy array of geometry mask, with the same dimensions as
            the input grid.
    """
    mask = np.zeros_like(arr)

    for m, n in np.ndindex(arr.shape):
        if arr[m, n] != 0.0:
            mask[m, n] = 1.0
    return mask

def normalize(data: np.array, input: bool, new_data: np.array, undo: bool,b, a):
    """

    Non-dimensionalizes values by using reference parameters based on characteristic values.
    This is a common practice in Fluid Mechanics.

    Args:
        data_arr: The multi-dimensional array of field values to be normalized.
        idx: The z index to normalize.

    Returns:
        Normalized field array.
    """

    # config_bc_dir = run_dir + '/bc_dir/bc_config'
    # config_bc = ConfigParser(config_bc_dir)
    # Get the inlet velocity - this will be the characteristic velocity.
    #u_inlet_x = '2*y*(1-y)/(1*1)'
        #config_bc.get_list(['CHARACTERISTIC', 'inlet_cnn'], str)[0]
    # inlet_u = bc_u['inlet']

    # Velocity reference for Ux and Uy, which are indices 1,2 and 4,5
    if input is True:
        vel_indices = [1,2]
        p_indices = [3]
    else:
        vel_indices = [1,2,4,5]
        p_indices = [3,6]

    # vertices = mesh.vertices
    # v1, v3 = vertices[0], vertices[2]
    # x0, xn, y0, yn = v1.point[0], v3.point[0], v1.point[1], v3.point[1]
    #
    # y_interp = np.linspace(y0, yn, n)
    # y_ind = np.arange(n)

    #y = 0.5
    #u_ref = eval(u_inlet_x)
    #u_ref = '3.75'
    #print(u_ref, u_inlet_x)
    #new_data = np.empty(np.shape(data))


    if undo is False:
        for i in range(len(data)):
            for idx in vel_indices:
                max_val = np.amax(data[i,idx,:,:])
                print(max_val)
                min_val = np.amin(data[i,idx,:,:])
                print(min_val)
                for m, n in np.ndindex(data[i,idx].shape):
                    new_data[i,idx,m,n] = ((b - a) * (data[i,idx,m,n] - min_val) / (max_val - min_val)) + a
            for idx in p_indices:
                max_val = np.amax(data[i, idx, :, :])
                print(max_val)
                min_val = np.amin(data[i, idx, :, :])
                print(min_val)
                for m, n in np.ndindex(data[i,idx].shape):
                    new_data[i, idx, m, n] = ((b - a) * (data[i,idx,m,n] - min_val) / (max_val - min_val)) + a
        new_data[0, 0, :, :] = apply_mask(new_data[0, 1, :, :])


        # for i in range(len(data)):
        #     print(u_ref)
        #     for idx in vel_indices:
        #         new_data[:,idx, :,:] = data[:,idx, :,:] / (float(u_ref))
        #     for idx in p_indices:
        #         new_data[:,idx, :, :] = data[:,idx, :, :] / (float(u_ref**2))

    # if undo is True:
    #     for i in range(len(data)):
    #         print(u_ref)
    #         for idx in vel_indices:
    #             new_data[:,idx, :,:] = data[:,idx, :,:] * (float(u_ref))
    #         for idx in p_indices:
    #             new_data[:,idx, :, :] = data[:,idx, :, :] * (float(u_ref**2))

        # n is the physical 'y', as it is the row number.
        # for n,m in np.ndindex(data.shape[-2:]):
        #     print(data.shape[-2:])
        #     y = 0.5
        #         #float(np.interp(n, y_ind, y_interp))
        #     print(y, n, m, u_inlet_x)
        #     u_ref = eval(u_inlet_x)
        #     print("uref", u_ref)
        #     if idx in vel_indices:
        #         data[i, idx, n, m] = data[i, idx, n,m] / u_ref
        #     elif idx in p_indices:
        #         data[i, idx, n, m] = data[i, idx, n, m] / u_ref**2

    return new_data

def unnormalize(data, min_val, max_val):
        """
        Unnormalizes values from 0 to 1, to be between original min and max of field values.
        This function takes the output of the neural network fields to go back into OpenFOAM.
        Args:
            val: The field value at a point in space. Example: Ux[i][j]
            min_val: The
            max_val:
        Returns:
            Unnormalized field array.
        """
        return ((max_val - min_val) * data) + min_val

def normalize_ref(data: np.array, input: bool, uref: float):
    def normalize(data: np.array, input: bool, new_data: np.array, undo: bool, b, a):
        """

        Non-dimensionalizes values by using reference parameters based on characteristic values.
        This is a common practice in Fluid Mechanics.

        Args:
            data_arr: The multi-dimensional array of field values to be normalized.
            idx: The z index to normalize.

        Returns:
            Normalized field array.
        """

        # Velocity reference for Ux and Uy, which are indices 1,2 and 4,5
        if input is True:
            vel_indices = [1, 2]
            p_indices = [3]
        else:
            vel_indices = [1, 2, 4, 5]
            p_indices = [3, 6]

        if undo is False:
            for i in range(len(data)):
                for idx in vel_indices:
                    max_val = np.amax(data[i, idx, :, :])
                    print(max_val)
                    min_val = np.amin(data[i, idx, :, :])
                    print(min_val)
                    for m, n in np.ndindex(data[i, idx].shape):
                        new_data[i, idx, m, n] = ((b - a) * (data[i, idx, m, n] - min_val) / (max_val - min_val)) + a
                for idx in p_indices:
                    max_val = np.amax(data[i, idx, :, :])
                    print(max_val)
                    min_val = np.amin(data[i, idx, :, :])
                    print(min_val)
                    for m, n in np.ndindex(data[i, idx].shape):
                        new_data[i, idx, m, n] = ((b - a) * (data[i, idx, m, n] - min_val) / (max_val - min_val)) + a
            new_data[0, 0, :, :] = apply_mask(new_data[0, 1, :, :])

def create_nn_data(gfu: Optional[ngs.GridFunction], mesh: ngs.Mesh, poly: Optional[int],
                   element: Optional[dict], sample_n: int, n: int, m: int, target_file: Optional[str],
                   sol_dir: Optional[str]):
    """
            Function to create convolutional neural network data using the above functions.

            Args:
                sim_list: List of string paths to the simulation folders to create data from.
                          Example -> ['./INS_1', './INS_2']
                sample_n: The number of samples to grab from each simulation folder.
                n: Desired number of discrete cells (elements) in the uniform grid (numpy array)
                   in the x direction.
                m: Desired number of discrete cells (elements) in the uniform grid (numpy array)
                   in the y direction.

            Returns:
                Numpy array of size [len(sim_list)*sample_n,3,n,m] containing len(sim_list)*sample_n
                samples of 3 elements (channels) Ux, Uy, P of size (resolution) nxm.
    """

    # Initiate numpy array.
    #cnn_data = np.empty((len(sim_list)*sample_n, 7, n, m))

    # Swap
    cnn_data = np.empty((sample_n, 7, n, m))

    # Continue populating cnn_data until desired number of samples is reached.
    i = 0
    l = 0
    act_files = []
    # Obtain directory with .sol files.
    # sol_dir = run_dir + '/output/sol/'
    if sol_dir is not None:
        all_sol_files = os.listdir(sol_dir)
        sol_samples = all_sol_files
        #[:sample_n]

        for file in sol_samples:

            if int(re.findall(r'\d+', file)[0]) <= (sample_n-3): \
                    #and int(re.findall(r'\d+', file)[0]) > 0:
                file_dir = sol_dir + '/' + file
                act_files.append(file_dir)

            #file_dir = sol_dir + '/' + 'stokes_0.0.sol'
            #if gfu is None:
                gfu = create_gfu(poly=poly, mesh=mesh,element=element,sol_path_str=file_dir)
                data = create_np_data(gfu=gfu, mesh=mesh, n=n, m=m, mask=False)
                print("file_dir:", file)
                print(np.shape(cnn_data), np.shape(data))
                cnn_data[i, 1, :, :] = data[0,0]  # Ux
                cnn_data[i, 2, :, :] = data[0,1]  # Uy
                cnn_data[i, 3, :, :] = data[0,2]  # P
                cnn_data[i, 0, :, :] = apply_mask(data[0,0])  # Geometry mask
                l+=1

                # Obtain the target .sol file.
                if target_file is not None:
                    gfu_target = create_gfu(poly=poly, mesh=mesh, element=element, sol_path_str=target_file)
                    target_data = create_np_data(gfu=gfu_target, mesh=mesh, n=n, m=m, mask=False)
                    print(np.shape(cnn_data[i, 4, :, :]), np.shape(target_data[0]))

                    cnn_data[i, 4, :, :] = target_data[0,0]  # Ux
                    cnn_data[i, 5, :, :] = target_data[0,1]  # Uy
                    cnn_data[i, 6, :, :] = target_data[0,2]  # P
                i += 1


    else:
        data = create_np_data(gfu=gfu, mesh=mesh, n=n, m=m)
        cnn_data[i, 1, :, :] = data[0]  # Ux
        cnn_data[i, 2, :, :] = data[1]  # Uy
        cnn_data[i, 3, :, :] = data[2]  # P
        cnn_data[i, 0, :, :] = apply_mask(data[0])  # Geometry mask
    print("sol files", sol_samples)
    print("N is", l)
    print("Actual files", act_files)


    return cnn_data
