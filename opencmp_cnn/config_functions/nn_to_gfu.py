from ngsolve import *
from ngsolve.comp import IntegrationRuleSpace
import numpy as np
import ngsolve as ngs
from opencmp.config_functions import expanded_config_parser
from scipy import interpolate

def numpy_to_gfu(np_data: np.array, config_file: str, gfu_init: GridFunction, mesh: Mesh, n: int, m: int, point_map_u: np.array,
                 point_map_p: np.array, gfuir_coor_u: GridFunction, gfuir_coor_p: GridFunction, gfu_final: GridFunction):
    """
        Function to create gridfunction from neural network output.

        Arguments:
            np_data: The neural network output, which is in the form of a numpy array.
            config_file: The config file for the simulation.
            gfu_init: The initial gridfunction where the simulation last left off.
    """

    #dir_path = 'examples/tutorial_6'
    #config_file = dir_path + '/config'
    config = expanded_config_parser.ConfigParser(config_file)

    # Get the run directory and the model.
    run_dir = config.get_item(['OTHER', 'run_dir'], str)
    model = config.get_item(['OTHER', 'model'], str)

    # Load the finite element space parameters.
    element = config.get_dict(['FINITE ELEMENT SPACE', 'elements'], run_dir, None, all_str=True)
    interp_ord = config.get_item(['FINITE ELEMENT SPACE', 'interpolant_order'], int)

    # Create IntegrationRuleSpaces for each element.
    # Each element will need two IRS: One to interpolate the Gridfunction data to, another to get the coordinates from.

    idx = 0 # Index in Gridfunction components.
    w = 0
    for field in element:
        # Get FES.
        V = getattr(ngs, element[field])
        gfuir_coor = gfuir_coor_u
        if field == 'p':
            # For pressure space, it will be one polynomial degree less than V.
            interp_ord -= 1
            gfuir_coor = gfuir_coor_p

        # Corresponding Gridfunction component if Mixed FiniteElementSpace was used.
        if len(gfu_init.components) > 0:
            # Get dimension of field.
            dim = gfu_init.components[idx].dim
        else:
            dim = gfu_init.dim

        if dim >= 2:
            fesir = IntegrationRuleSpace(mesh=mesh, order=interp_ord, dirichlet="wall|inlet|cyl") ** dim
            fesir_irs = IntegrationRuleSpace(mesh=mesh, order=interp_ord, dirichlet="wall|inlet|cyl")
        else:
            fesir = IntegrationRuleSpace(mesh=mesh, order=interp_ord)
            fesir_irs = fesir

        # Create IRS Gridfunction for data and for coordinates.
        gfuir = GridFunction(fesir)
        #gfuir_coor = GridFunction(fesir_coor)

        # Interpolate point-values in integration points.
        # if len(gfu_init.components) > 0:
        #     gfuir.Interpolate(gfu_init.components[idx])
        # else:
        #     gfuir.Interpolate(gfu_init)

        # if len(gfu_init.components) > 0:
        #     gfuir.Interpolate(gfu_init.components[idx])
        # else:
        #     gfuir.Interpolate(gfu_init)

        vertices = mesh.vertices
        v1, v3 = vertices[0], vertices[2]
        x0, xn, y0, yn = v1.point[0], v3.point[0], v1.point[1], v3.point[1]

        ########################################################
        # Reverse interpolate                                  #
        ########################################################

        # Create uniform grid points based on mesh dimensions.
        x_interp = np.linspace(x0, xn, m)
        y_interp = np.linspace(y0, yn, n)

        # Uniform indices.
        x_ind = np.arange(m)
        y_ind = np.arange(n)

        # Coordinates from IntegrationRuleSpace.
        coord_x = gfuir_coor.components[0]
        coord_y = gfuir_coor.components[1]

        # Create interpolant from numpy data:
        np_interpolant_ux = interpolate.interp2d(x_interp, y_interp, np_data[0],  kind='linear')
        np_interpolant_uy = interpolate.interp2d(x_interp, y_interp, np_data[1],  kind='linear')
        np_interpolant_p = interpolate.interp2d(x_interp, y_interp, np_data[2],  kind='linear')

        for i in range(0,len(coord_x.vec)):
            p1 = coord_x.vec[i]
            p2 = coord_y.vec[i]

            # Get corresponding indices.
            # x_idx = int(np.interp(p1, x_interp, x_ind))
            # y_idx = int(np.interp(p2, y_interp, y_ind))



            # Fill gfuir data with numpy data.
            if idx == 0:
                x_idx = point_map_u[i][1]
                y_idx = point_map_u[i][0]
                gfuir.components[0].vec[i] = np_data[0][y_idx, x_idx]
                gfuir.components[1].vec[i] = np_data[1][y_idx, x_idx]
                # gfuir.components[0].vec[i] = gfu_final.components[0](mesh(p1,p2))[0]
                # gfuir.components[1].vec[i] =gfu_final.components[0](mesh(p1, p2))[1]
            else:
                x_idx = point_map_p[i][1]
                y_idx = point_map_p[i][0]

                gfuir.vec[i] = np_data[2][y_idx, x_idx]
                gfuir.vec[i] = gfu_final.components[1](mesh(p1, p2))
                # # Since this is for 2D problems, we are only considering 2 components.


                # gfuir.components[0].vec[i] = gfu_final.components[0](mesh(p1,p2))[0]
                # gfuir.components[1].vec[i] =gfu_final.components[0](mesh(p1, p2))[1]

                # Trying cubic interpolation...
                # ux_val = np_interpolant_ux(p1, p2)
                # uy_val = np_interpolant_uy(p1, p2)
                #
                # gfuir.components[0].vec[i] = ux_val
                # gfuir.components[1].vec[i] = uy_val

                # print("Points ", p1, p2, " GFU final value: ", gfu_final.components[0](mesh(p1,p2))[0],
                #       "Interpolant value: ",np_data[0][y_idx, x_idx])
                #
                # print("Points ", p1, p2, " GFU final value: ", gfu_final.components[0](mesh(p1, p2))[1],
                #       "Interpolant value: ", np_data[1][y_idx, x_idx])

            if np_data[0][y_idx, x_idx] != gfu_final.components[0](mesh(p1, p2))[0]:
                w += 1
                print("points", p1, p2, "indices", x_idx, y_idx, "Numpy val: ", np_data[0][y_idx, x_idx], "Actual ", gfu_final.components[0](mesh(p1, p2))[0])
                print(w)
                # gfuir.components[0].vec[i] = gfu_final.components[0](mesh(p1,p2))[0]
                # gfuir.components[1].vec[i] = gfu_final.components[0](mesh(p1,p2))[1]





                # p_val = np_interpolant_p(p1, p2)
                # gfuir.vec[i] = p_val
                # gfuir.vec[i] = gfu_final.components[1](mesh(p1,p2))


        irs = fesir_irs.GetIntegrationRules()

        fes = V(mesh=mesh, order=interp_ord, dirichlet="wall|inlet|cyl")
        p, q = fes.TnT()
        mass = BilinearForm(p * q * dx).Assemble().mat
        invmass = mass.Inverse(inverse="sparsecholesky")

        rhs = LinearForm(gfuir * q * dx(intrules=irs))
        rhs.Assemble()
        # commenting this out
        gfu_init.components[idx].vec.data = invmass * rhs.vec

        idx += 1
    print(len(gfuir_coor_u.components[0].vec))
    print(len(point_map_u))
    print(len(gfuir_coor_p.components[0].vec))
    print(len(point_map_p))
    print(w)
    return gfu_init

























def numpy_to_gfu2(np_data: np.array, config_file: str, gfu_init: GridFunction, mesh: Mesh, n: int, m: int):
    """
        Function to create gridfunction from neural network output2.

        Arguments:
            np_data: The neural network output2, which is in the form of a numpy array.
            config_file: The config file for the simulation.
            gfu_init: The initial gridfunction where the simulation last left off.
    """

    #dir_path = 'examples/tutorial_6'
    #config_file = dir_path + '/config'
    config = expanded_config_parser.ConfigParser(config_file)

    # Get the run directory and the model.
    run_dir = config.get_item(['OTHER', 'run_dir'], str)
    model = config.get_item(['OTHER', 'model'], str)

    # Load the finite element space parameters.
    element = config.get_dict(['FINITE ELEMENT SPACE', 'elements'], run_dir, None, all_str=True)
    interp_ord = config.get_item(['FINITE ELEMENT SPACE', 'interpolant_order'], int)

    # Create IntegrationRuleSpaces for each element.
    # Each element will need two IRS: One to interpolate the Gridfunction data to, another to get the coordinates from.

    idx = 0 # Index in Gridfunction components.
    # Full Mass Matrix for debugging.
    V = VectorH1(mesh=mesh, order=interp_ord, dirichlet="wall|inlet|cyl")
    Q = H1(mesh=mesh, order=interp_ord-1, dirichlet="wall|inlet|cyl")
    # V = VectorH1(mesh=mesh, order=interp_ord, dirichlet="wall|inlet|cyl")
    # Q = H1(mesh=mesh, order=interp_ord - 1)
    fes = FESpace([V, Q])

    u, p = fes.TrialFunction()
    v, q = fes.TestFunction()
    print("made it to this point")

    mass = BilinearForm(p*q*dx + u*v*dx).Assemble().mat
    invmass = mass.Inverse(inverse="sparsecholesky")

    # Corresponding Gridfunction component if Mixed FiniteElementSpace was used.
    if len(gfu_init.components) > 0:
        # Get dimension of field.
        dim = gfu_init.components[idx].dim
    else:
        dim = gfu_init.dim

    if dim >= 2:
        fesir_u = IntegrationRuleSpace(mesh=mesh, order=interp_ord, dirichlet="wall|inlet|cyl") ** dim
        fesir_irs_u = IntegrationRuleSpace(mesh=mesh, order=interp_ord, dirichlet="wall|inlet|cyl")
        fesir_p = IntegrationRuleSpace(mesh=mesh, order=interp_ord-1, dirichlet="wall|inlet|cyl")
        fesir_irs_p = IntegrationRuleSpace(mesh=mesh, order=interp_ord-1, dirichlet="wall|inlet|cyl")
    else:
        fesir = IntegrationRuleSpace(mesh=mesh, order=interp_ord, dirichlet="wall|inlet|cyl")
        fesir_irs = fesir



    # Working with 2d data - the fesir coordinate data always has dimension 2.
    fesir_coor_u = IntegrationRuleSpace(mesh=mesh, order=interp_ord) ** 2
    fesir_coor_p = IntegrationRuleSpace(mesh=mesh, order=interp_ord-1) ** 2

    # Create IRS Gridfunction for data and for coordinates.
    gfuir_u = GridFunction(fesir_u)
    gfuir_p = GridFunction(fesir_p)
    gfuir_coor_u = GridFunction(fesir_coor_u)
    gfuir_coor_p = GridFunction(fesir_coor_p)

    # Interpolate point-values in integration points.

    gfuir_u.Interpolate(gfu_init.components[0])
    gfuir_p.Interpolate(gfu_init.components[1])


    gfuir_coor_u.Interpolate(CF((x, y)))
    gfuir_coor_p.Interpolate(CF((x, y)))

    vertices = mesh.vertices
    v1, v3 = vertices[0], vertices[2]
    x0, xn, y0, yn = v1.point[0], v3.point[0], v1.point[1], v3.point[1]

    ########################################################
    # Reverse interpolate                                  #
    ########################################################

    # Create uniform grid points based on mesh dimensions.
    x_interp = np.linspace(x0, xn, m)
    y_interp = np.linspace(y0, yn, n)

    # Uniform indices.
    x_ind = np.arange(m)
    y_ind = np.arange(n)

    # Coordinates from IntegrationRuleSpace.
    coord_x_u = gfuir_coor_u.components[0]
    coord_y_u = gfuir_coor_u.components[1]

    coord_x_p = gfuir_coor_p.components[0]
    coord_y_p = gfuir_coor_p.components[1]

    for i in range(len(coord_x_u.vec)):
        print("made it to this point")
        p1 = coord_x_u.vec[i]
        p2 = coord_y_u.vec[i]

        # Get corresponding indices.
        x_idx = int(np.interp(p1, x_interp, x_ind))
        y_idx = int(np.interp(p2, y_interp, y_ind))

        # Fill gfuir data with numpy data.

        # Since this is for 2D problems, we are only considering 2 components.
        if np_data[0][y_idx, x_idx] != -10 and np_data[1][y_idx, x_idx] != -10:
            print("ux is ", np_data[0][y_idx, x_idx], "uy is ", np_data[1][y_idx, x_idx])
            gfuir_u.components[0].vec[i] = np_data[0][y_idx, x_idx]
            gfuir_u.components[1].vec[i] = np_data[1][y_idx, x_idx]

    for i in range(len(coord_x_p.vec)):
        print("made it to this point")
        p1 = coord_x_p.vec[i]
        p2 = coord_y_p.vec[i]

        # Get corresponding indices.
        # x_idx = int(np.interp(p1, x_interp, x_ind))
        # y_idx = int(np.interp(p2, y_interp, y_ind))

        # Fill gfuir data with numpy data.

        # Since this is for 2D problems, we are only considering 2 components.
        if np_data[2][y_idx, x_idx] != -10:
            print("Here p")
            gfuir_p.vec[i] = np_data[2][y_idx, x_idx]

    irs_u = fesir_irs_u.GetIntegrationRules()
    irs_p = fesir_irs_p.GetIntegrationRules()


    # fes = V(mesh=mesh, order=interp_ord)
     # p, q = fes.TnT()
    # mass = BilinearForm(p * q * dx).Assemble().mat
    # invmass = mass.Inverse(inverse="sparsecholesky")

    rhs_u = LinearForm(gfuir_u * v * dx(intrules=irs_u))
    rhs_u.Assemble()
        #
    #gfu_init.components[idx].vec.data = invmass * rhs.vec
    gfu_init.vec.data = invmass * rhs_u.vec

    # rhs_p = LinearForm(gfuir_p * q * dx(intrules=irs_p))
    # rhs_p.Assemble()
    #
    # gfu_init.vec.data = invmass * rhs_p.vec




    return gfu_init
# For each element, create IntegrationRuleSpaces;






