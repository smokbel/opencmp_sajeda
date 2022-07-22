import sys
import numpy as np
import vtk
import meshio
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate

n = 100
m = 60

openfoam_dir = sys.argv[1]
final_t = sys.argv[2] 

sim_dir = openfoam_dir + '/' + final_t + '/'

# The points are stored in the 'C' files in the latestTime openfoam simulation directory after running
# postProcess -func writeCellCentres -latestTime. The x coords are located in Cx and the y coords
# are located in Cy.

points_file = sim_dir + 'C'

# The U and P data are stored in the 'U' and 'P' file in the openfoam simulation directory.
U_file = sim_dir + 'U'
p_file = sim_dir + 'p'

# Open and read coordinate and field files.
points = open(points_file,'r')
velocity = open(U_file, 'r')
pressure = open(p_file, 'r')

lines_points = points.readlines()
lines_vel = velocity.readlines()
lines_pres = pressure.readlines()

# HARDCODED sajeda@sajeda: might need to fix, not sure how to.
# In Openfoam 9, the number of points is given on the 20th line of the 'C' file.
num_points = int(lines_points[20])

# In order to obtain the max and min coordinates, a VTK file is read with meshio.
case_name = openfoam_dir.split("/")[-1]
vtk_file = openfoam_dir + '/VTK/{}_{}.vtk'.format(case_name,final_t)

# Read mesh file and obtain maximum and minimum coordinates to develop gridspace.
mesh = meshio.read(vtk_file)
x_coor = np.array(mesh.points[:,0])
y_coor = np.array(mesh.points[:,1])

x_interp = np.linspace(np.min(x_coor), np.max(x_coor), num=n)
y_interp = np.linspace(np.min(y_coor), np.max(y_coor), num=m)

x_coords_map = np.arange(n)
y_coords_map = np.arange(m)


# Initialize numpy arrays for field data.
output_Ux = np.zeros((n,m))
output_Uy = np.zeros((n,m))
output_P = np.zeros((n,m))

# HARDCODED sajeda@sajeda: Find a better way to locate lines with point values if possible.
# The point coordinates start on line 22, and there are num_points amount of them.

# First, grab all x and y coordinates and store them in an array.
x_coords = []
y_coords = []
Ux_vals = []
Uy_vals = []
P_vals = []

def numpy_to_scatter(arr, n, m, mesh, x_coords, y_coords, Ux_vals):

    x_coor = np.array(mesh.points[:, 0])
    y_coor = np.array(mesh.points[:, 1])

    x_interp = np.linspace(np.min(x_coor), np.max(x_coor), num=n)
    y_interp = np.linspace(np.min(y_coor), np.max(y_coor), num=m)

    fcn = interpolate.RectBivariateSpline(x_interp,y_interp,arr)
    for i in range(len(x_coords)):
        new_Ux = fcn.ev(x_coords[i], y_coords[i])
        print(new_Ux, Ux_vals[i])


for i in range(22,22+num_points):
    coords_xyz = lines_points[i].replace('(', '').replace(')', '').split(' ')
    U_data = lines_vel[i].replace('(', '').replace(')', '').split(' ')
    P_data = lines_pres[i]

    Xval = coords_xyz[0]
    Yval = coords_xyz[1]

    Ux_val = U_data[0]
    Uy_val = U_data[1]

    x_coords.append(float(Xval))
    y_coords.append(float(Yval))

    Ux_vals.append(float(Ux_val))
    Uy_vals.append(float(Uy_val))
    P_vals.append(float(P_data))

#    Create interpolant function for Ux values.
# f_ux = interpolate.interp2d(x_coords, y_coords, Ux_vals, kind='linear')
# print(x_coords, y_coords)

output_Ux = np.load('./output_Ux.npy')
numpy_to_scatter(output_Ux, n, m, mesh, x_coords, y_coords, Ux_vals)


# i = 0
# for i in range(len(x_coords)):
#     Xval = x_coords[i]
#     Yval = y_coords[i]
#
#     idx_x = int(np.interp(Xval, x_interp, x_coords_map))
#     idx_y = int(np.interp(Yval, y_interp, y_coords_map))
#
#     if output_Ux[idx_x,idx_y] == 0:
#         output_Ux[idx_x,idx_y] = Ux_vals[i]
#     elif idx_x == 58 or idx_x == 0:
#         print("here")
#         output_Ux[idx_x, idx_y] = 0




    # else:
    #     # Find the next closest point
    #     t = 1
    #     q= 2
    #
    #     if output_Ux[idx_x+t,idx_y] == 0:
    #         output_Ux[idx_x + t, idx_y] =  Ux_vals[i]
    #     elif output_Ux[idx_x-t,idx_y] == 0:
    #         output_Ux[idx_x - t, idx_y] = Ux_vals[i]
    #     elif  output_Ux[idx_x,idx_y+t] == 0:
    #         output_Ux[idx_x, idx_y + t] = Ux_vals[i]
    #     elif output_Ux[idx_x, idx_y - t] == 0:
    #         output_Ux[idx_x, idx_y - t] = Ux_vals[i]
    #     elif output_Ux[idx_x-t, idx_y - t] == 0:
    #         output_Ux[idx_x-t, idx_y - t] = Ux_vals[i]
    #     elif output_Ux[idx_x+t, idx_y+t] == 0:
    #         output_Ux[idx_x+t, idx_y+t] = Ux_vals[i]











# Given arrays of [x,y,u,p] we can use an interpolating function to determine values on a uniform grid.

#for i in range(22,22+num_points):


    # Get corresponding Ux values for new, uniform points.
# df = pd.DataFrame(data=z_ux, columns=x_interp, index=y_interp)
# sns.heatmap(df, square=False)
# plt.show()



    # Obtain coordinates from the line.
    # coords_xyz = lines_points[i].replace('(', '').replace(')', '').split(' ')
    # U_data = lines_vel[i].replace('(', '').replace(')', '').split(' ')
    # P_data = lines_pres[i]
    #
    # Xval = coords_xyz[0]
    # Yval = coords_xyz[1]
    #

    #
    # # Grab corresponding Ux, Uy, and P data for point.
    # output_Ux[idx_x, idx_y] = float(U_data[0])
    # output_Uy[idx_x, idx_y] = float(U_data[1])
    # output_P[idx_x, idx_y] = float(P_data)


# vis =sns.heatmap(output_Ux,vmin=np.min(output_Ux),vmax=np.max(output_Ux))
# plt.show()
# #
# vis =sns.heatmap(output_Uy,vmin=np.min(output_Uy),vmax=np.max(output_Uy))
# plt.show()
#
# vis =sns.heatmap(output_P,vmin=np.min(output_P),vmax=np.max(output_P))
# plt.show()