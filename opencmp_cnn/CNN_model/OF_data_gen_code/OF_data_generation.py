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

print(len(x_coords))
print(len(y_coords))
print(len(Ux_vals))
print(x_coords_map)
print(max(x_coords), max(y_coords), max(Ux_vals))
print(min(x_coords), min(y_coords), min(Ux_vals))
# Given arrays of [x,y,u,p] we can use an interpolating function to determine values on a uniform grid.

#for i in range(22,22+num_points):
#    Create interpolant function for Ux values.


f_ux = interpolate.interp2d(x_coords, y_coords, Ux_vals, kind='cubic')

    # Get corresponding Ux values for new, uniform points.
z_ux = f_ux(x_interp, y_interp)
df = pd.DataFrame(data=z_ux, columns=x_interp, index=y_interp)
sns.heatmap(df, square=False)
plt.show()



    # Obtain coordinates from the line.
    # coords_xyz = lines_points[i].replace('(', '').replace(')', '').split(' ')
    # U_data = lines_vel[i].replace('(', '').replace(')', '').split(' ')
    # P_data = lines_pres[i]
    #
    # Xval = coords_xyz[0]
    # Yval = coords_xyz[1]
    #
    # idx_x = int(np.interp(Xval, x_interp, x_coords_map))
    # idx_y = int(np.interp(Yval, y_interp, y_coords_map))
    #
    # # Grab corresponding Ux, Uy, and P data for point.
    # output_Ux[idx_x, idx_y] = float(U_data[0])
    # output_Uy[idx_x, idx_y] = float(U_data[1])
    # output_P[idx_x, idx_y] = float(P_data)


# vis =sns.heatmap(output_Ux,vmin=np.min(output_Ux),vmax=np.max(output_Ux))
# plt.show()
#
# vis =sns.heatmap(output_Uy,vmin=np.min(output_Uy),vmax=np.max(output_Uy))
# plt.show()
#
# vis =sns.heatmap(output_P,vmin=np.min(output_P),vmax=np.max(output_P))
# plt.show()