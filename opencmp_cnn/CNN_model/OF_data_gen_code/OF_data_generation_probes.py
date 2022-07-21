import sys, re
import numpy as np
import vtk
import meshio
import seaborn as sns
import matplotlib.pyplot as plt

n = 100
m = 60

openfoam_dir = sys.argv[1]
final_t = sys.argv[2] 

str_chars = '()' 

# Probes file to get data from.
file_dir = openfoam_dir + '/postProcessing/probes/%s/' % final_t 

# Initialize numpy array for field data.
field_arr = np.zeros((n,m))

def map_idx(point, type,interp_valx,x_map,interp_valy,y_map):
    """
    Mapping (interpolation) function. Given a field value, this function maps it from the 
    range min(field) -> max(field), to the index of the 2D (100,60) array. Can also be used
    vise versa. 
    Args:
        point: The field value at a point in space. Example: Ux[i][j]
        type: For 2D array, can be either 'x' or 'y'
        interp_valx: 
        x_map: corresponding 
    Returns:
        Unnormalized field array.
    """
    if type == 'x':
        mp = np.interp(point,interp_valx, x_map)
        return int(mp)
    mp = np.interp(point,interp_valy,y_map)
    return int(mp)

def file_handle(file_name: str,n: int,m: int, field: str):

    """
        Function to create numpy array of interpolated field data.

        Args:
            file_name: The location of the probes version of the field file. 
            n: Desired number of discrete cells (elements) in the uniform grid (numpy array)
               in the x direction.
            n: Desired number of discrete cells (elements) in the uniform grid (numpy array)
               in the y direction.
            field: Desired field to create numpy array from. Can be 'Ux', 'Uy', or 'P'.

        Returns:
            Numpy array of projected field data.
    """

    data_file = open(file_name, 'r') 

    if field == 'Ux':

        field_arr = re.split('  +', list(data_file)[-1])
        field_arr = field_arr[2:]

        for i in range(len(field_arr)):
            field_arr[i] = field_arr[i].strip(str_chars)
            field_arr[i] = re.search(r'-?[\d.]+(?:e-?\d+)?', field_arr[i]).group()
        field_arr = [float(x) for x in field_arr]
        field_arr = np.array(field_arr)
    
    elif field == 'Uy': 
        field_arr = re.split('  +', list(data_file)[-1])
        field_arr = field_arr[2:]
        field_arr = [float(y.split()[1]) for y in field_arr]
  
    elif field == 'p': 
        p = re.split('  +', list(data_file)[-1])
        p = p[2:]
        def flatten(li):
            return sum(([x] if not isinstance(x, list) else flatten(x) for x in li), [])
        field_arr = []
        for i in p:
            field_arr.append(i.split(' '))
        field_arr = flatten(field_arr)
    
    data_file2 = open(file_name, 'r') 
    lines = data_file2.readlines()
    spt = np.empty(n*m, dtype=object)
    i = 0

    for line in lines[0:n*m]:
        spt[i] = re.search(r'\((.*?)\)', line).group().strip(str_chars).strip()
        i += 1
        
    spt = np.array([np.fromstring(i, sep=" ") for i in spt])    
    x_coor = np.unique(np.array(spt[:,0]))
    y_coor = np.unique(np.array(spt[:,1]))
    x_map = np.arange(n)
    y_map = np.arange(m)

    output = np.zeros((n,m))
    for i in range(m*n): 
        Xval = spt[i,0] 
        xidx = map_idx(Xval,'x',interp_valx=x_coor, x_map=x_map, interp_valy=y_coor, y_map=y_map)
        Yval = spt[i,1] 
        yidx = map_idx(Yval,'y',interp_valx=x_coor, x_map=x_map, interp_valy=y_coor, y_map=y_map)
        output[xidx, yidx] = field_arr[i]

    return output

def remove_inf(data_arr, m, n):
    for x1 in range(m):
        for x2 in range(n):
            if np.abs(data_arr[x2,x1]) > 1.7:
                data_arr[x2,x1] = 0
    return data_arr

output_Ux = file_handle(file_name=file_dir + 'U', n=n, m=m, field='Ux')
output_Uy = file_handle(file_name=file_dir + 'U', n=n, m=m, field='Uy')
output_P = file_handle(file_name=file_dir + 'P', n=n, m=m, field='p')

output_Ux = remove_inf(output_Ux, m, n)
output_Uy = remove_inf(output_Uy, m, n)
output_P = remove_inf(output_P, m, n)

vis =sns.heatmap(output_Ux,vmin=np.min(output_Ux),vmax=np.max(output_Ux))
plt.show()

vis =sns.heatmap(output_Uy,vmin=np.min(output_Uy),vmax=np.max(output_Uy))
plt.show()

vis =sns.heatmap(output_P,vmin=np.min(output_P),vmax=np.max(output_P))
plt.show()