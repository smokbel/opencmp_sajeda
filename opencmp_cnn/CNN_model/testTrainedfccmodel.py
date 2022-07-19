import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.optim as optim
import sys
from timeit import default_timer as timer
from ngsolve import *
from ngsolve.comp import IntegrationRuleSpace
import numpy as np
import ngsolve as ngs
from opencmp.config_functions import expanded_config_parser
from scipy import interpolate
from fcnn import NeuralNet

start = timer()
# Load the CNN and its parameters.
model = NeuralNet()
model = model.double()
param_file = 'param.json'

with open(param_file) as json_file:
    param = json.load(json_file)

with open('./cases/case1/info.json') as json_file_2:
    param2 = json.load(json_file_2)

optimizer = optim.RMSprop(model.parameters(), lr=param['learning_rate'])
#model_save = param['model_save']

model.load_state_dict(torch.load(str('model_jul7.pth')))
model.eval()
model = model.double()


data = np.load('./data_fc_test.npy')
in_value = data[8:9,0:4,:,0]
in_model = torch.from_numpy(in_value)

prediction = model.forward(in_model)
prediction = prediction.detach().numpy()

sol_file = './cases/case1/ins_14.sol'
sol_file_target = './cases/case1/ins_80.sol'

dir_path = '../../examples/ins_sajeda_test'
config_file = dir_path + '/config'

mesh_file = param2["mesh_file"]
mesh_path = dir_path + "/" + mesh_file
mesh = Mesh(mesh_path)




def numpy_to_gfu(config_file: str, mesh: Mesh, sol_file, sol_file_target):
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
    gfu_init = GridFunction(fes)
    gfu_init.Load(sol_file)

    gfu_final = GridFunction(fes)
    gfu_final.Load(sol_file_target)

    # Corresponding Gridfunction component if Mixed FiniteElementSpace was used.

    fesir_coor = IntegrationRuleSpace(mesh=mesh, order=3, dirichlet="wall|inlet|cyl") ** 2
    fesir_gfu = IntegrationRuleSpace(mesh=mesh, order=3, dirichlet="wall|inlet|cyl") ** 2
    fesir_irs = IntegrationRuleSpace(mesh=mesh, order=3, dirichlet="wall|inlet|cyl")

    # Create IRS Gridfunction for data and for coordinates.
    gfuir = GridFunction(fesir_gfu)
    gfuir_coor = GridFunction(fesir_coor)
    gfuir_coor.Interpolate(CF((x, y)))

    ########################################################
    # Reverse interpolate                                  #
    ########################################################

    # Coordinates from IntegrationRuleSpace.
    coord_x = gfuir_coor.components[0]
    coord_y = gfuir_coor.components[1]
    # Create interpolant from numpy data:
    u = 0
    for i in range(len(coord_x.vec)):
        p1 = coord_x.vec[i]
        p2 = coord_y.vec[i]

        resultx = np.where(in_value[0,2] == p1)
        resulty = np.where(in_value[0,3] == p2)

        for l in range(len(resultx[0])):
            for m in range(len(resulty[0])):
                print(resultx[0][l], resulty[0][m])
                if resultx[0][l] == resulty[0][m]:
                    gfuir.components[0].vec[i] = prediction[0,0,resultx[0][l]] * ux_max
                    gfuir.components[1].vec[i] = prediction[0,1,resultx[0][l]] * uy_max
                    print("Prediction value, ", prediction[0,0,resultx[0][l]] * ux_max)
                    print("Actual value ,",  gfu_final.components[0](mesh(p1,p2))[0])
                    print("Previous value, ", gfu_init.components[0](mesh(p1,p2))[0])
                else:
                    gfuir.components[0].vec[i] = gfu_init.components[0](mesh(p1,p2))[0]
                    gfuir.components[1].vec[i] = gfu_init.components[0](mesh(p1,p2))[1]




        # print(in_value[0,2,resultx[0]])
        # print(in_value[0,3,resulty[0]])
        # print(p1,p2)
        # print("X ind ", resultx[0])
        # print("Y ind ", resulty[0])
        # print(prediction[0,0,10:20])
        # print(data[0,4,10:20])




        # Get corresponding indices.
        # x_idx = int(np.interp(p1, x_interp, x_ind))
        # y_idx = int(np.interp(p2, y_interp, y_ind))
        # Fill gfuir data with numpy data.
    #     if idx == 0:
    #         x_idx = point_map_u[i][1]
    #         y_idx = point_map_u[i][0]
    #         gfuir.components[0].vec[i] = np_data[0][y_idx, x_idx]
    #         gfuir.components[1].vec[i] = np_data[1][y_idx, x_idx]
    #         # gfuir.components[0].vec[i] = gfu_final.components[0](mesh(p1,p2))[0]
    #         # gfuir.components[1].vec[i] =gfu_final.components[0](mesh(p1, p2))[1]
    #     else:
    #         x_idx = point_map_p[i][1]
    #         y_idx = point_map_p[i][0]
    #         gfuir.vec[i] = np_data[2][y_idx, x_idx]
    #         gfuir.vec[i] = gfu_final.components[1](mesh(p1, p2))
    #         # # Since this is for 2D problems, we are only considering 2 components.
    #         # gfuir.components[0].vec[i] = gfu_final.components[0](mesh(p1,p2))[0]
    #         # gfuir.components[1].vec[i] =gfu_final.components[0](mesh(p1, p2))[1]
    #         # Trying cubic interpolation...
    #         # ux_val = np_interpolant_ux(p1, p2)
    #         # uy_val = np_interpolant_uy(p1, p2)
    #         #
    #         # gfuir.components[0].vec[i] = ux_val
    #         # gfuir.components[1].vec[i] = uy_val
    #         # print("Points ", p1, p2, " GFU final value: ", gfu_final.components[0](mesh(p1,p2))[0],
    #         #       "Interpolant value: ",np_data[0][y_idx, x_idx])
    #         #
    #         # print("Points ", p1, p2, " GFU final value: ", gfu_final.components[0](mesh(p1, p2))[1],
    #         #       "Interpolant value: ", np_data[1][y_idx, x_idx])
    #     if np_data[0][y_idx, x_idx] != gfu_final.components[0](mesh(p1, p2))[0]:
    #         w += 1
    #         print("points", p1, p2, "indices", x_idx, y_idx, "Numpy val: ", np_data[0][y_idx, x_idx], "Actual ", gfu_final.components[0](mesh(p1, p2))[0])
    #         print(w)
    #         # gfuir.components[0].vec[i] = gfu_final.components[0](mesh(p1,p2))[0]
    #         # gfuir.components[1].vec[i] = gfu_final.components[0](mesh(p1,p2))[1]
    #         # p_val = np_interpolant_p(p1, p2)
    #         # gfuir.vec[i] = p_val
    #         # gfuir.vec[i] = gfu_final.components[1](mesh(p1,p2))
    #
    #
    # irs = fesir_irs.GetIntegrationRules()
    # fes = V(mesh=mesh, order=interp_ord, dirichlet="wall|inlet|cyl")
    # p, q = fes.TnT()
    # mass = BilinearForm(p * q * dx).Assemble().mat
    # invmass = mass.Inverse(inverse="sparsecholesky")
    # rhs = LinearForm(gfuir * q * dx(intrules=irs))
    # rhs.Assemble()
    # # commenting this out
    # gfu_init.components[idx].vec.data = invmass * rhs.vec
    # idx += 1
    #print(coord_x.vec)
    idx = 0
    irs = fesir_irs.GetIntegrationRules()

    fes = VectorH1(mesh=mesh, order=3, dirichlet="wall|inlet|cyl")
    p, q = fes.TnT()
    mass = BilinearForm(p * q * dx).Assemble().mat
    invmass = mass.Inverse(inverse="sparsecholesky")

    rhs = LinearForm(gfuir * q * dx(intrules=irs))
    rhs.Assemble()
    # commenting this out
    gfu_init.components[idx].vec.data = invmass * rhs.vec
    print(gfu_final.vec[0:10])
    return gfu_init


gfu_from_nn = numpy_to_gfu(config_file, mesh, sol_file, sol_file_target)
print(gfu_from_nn.vec[0:10])
gfu_from_nn.Save('../../examples/ins_sajeda_test/initial_test.sol')
vtk = VTKOutput(ma=mesh,
                coefs=[gfu_from_nn.components[0], gfu_from_nn.components[1]],
                names = ["u", "p"],
                filename="projected_gfu1",
                subdivision=3)
vtk.Do()
