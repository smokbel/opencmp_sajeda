from config_functions import sol_to_data
from ngsolve import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from opencmp.config_functions import ConfigParser

n = 64
m = 256
#
config_dir = '../examples/ins_sajeda/config'
mesh_file = '../examples/ins_sajeda/2d_ell_55.vol'
mesh = Mesh(mesh_file)
#
config = ConfigParser(config_dir)

run_dir = config.get_item(['OTHER', 'run_dir'], str)
element = config.get_dict(['FINITE ELEMENT SPACE', 'elements'], run_dir, None, all_str=True)

interp_ord = config.get_item(['FINITE ELEMENT SPACE', 'interpolant_order'], int)

# target_file = '../examples/ins_sajeda/output/sol/ins_25.sol'
# sol_dir = '../examples/ins_sajeda/output/sol/'
# data = sol_to_data.create_nn_data(gfu=None, mesh=mesh, poly=interp_ord, element=element, sample_n=20, n=64, m=256, target_file=target_file, sol_dir=sol_dir)
#
# data_2 = np.empty((len(data),7,n,m))
# train = sol_to_data.normalize(data, False, data_2, False, 0.0, 1, 0)
# #22
#np.save('./data/ncase_058', train)

#
#
# print(len(train))
# # test = np.load('./data/case_r1.npy')
#
case_025 = np.load('./data/case_055.npy')
# ncase_05 = np.load('./data/ncase_05.npy')
# ncase_052 = np.load('./data/ncase_052.npy')
# ncase_053 = np.load('./data/ncase_053.npy')
# ncase_054 = np.load('./data/ncase_054.npy')
# ncase_056 = np.load('./data/ncase_056.npy')
# ncase_057 = np.load('./data/ncase_057.npy')
# ncase_058 = np.load('./data/ncase_058.npy')
# #
# print(len(case_025))
# print(len(ncase_05))
# print(len(ncase_052))
# print(len(ncase_053))
# print(len(ncase_054))
# print(len(ncase_056))
# print(len(ncase_057))
# print(len(ncase_058))

#
#train = np.empty((295, 7, n, m))
# # # case025 = np.empty((len(case_025), 7, n, m))
# # # case035 = np.empty((len(case_035), 7, n, m))
# # # case055 = np.empty((len(case_055), 7, n, m))
# # # case07 = np.empty((len(case_07), 7, n, m))
# # # case01 = np.empty((len(case_01), 7, n, m))
# # # caser1 = np.empty((len(case_r1), 7, n, m))
# # #
# train[0:160,:,:,:] = case_025
# train[160:176,:,:,:] = ncase_05
# train[176:188,:,:,:] = ncase_052
# train[188:202,:,:,:] = ncase_053
# train[202:215,:,:,:] =  ncase_054
# train[215:230,:,:,:] =  ncase_056
# train[230:275,:,:,:] =  ncase_057
# train[275:295,:,:,:] = ncase_058
# train[len(case_025)+len(case_035):len(case_025)+len(case_035)+len(case_055), :, :, :] = case_055
# train[len(case_025)+len(case_035)+len(case_055):len(case_07)+len(case_025)+len(case_035)+len(case_055), :, :, :] \
#      = case_07
# #
# train[len(case_07)+len(case_025)+len(case_035)+len(case_055):len(case_07)+len(case_025)+len(case_035)+len(case_055)+len(case_01),:,:,:] = \
#      case_01
# #
# train[len(case_07)+len(case_025)+len(case_035)+len(case_055)+len(case_01):len(case_07)+len(case_025)+len(case_035)+len(case_055)+len(case_01)+len(case_r1),:,:,:] =\
#  case_r1
#
# train[len(case_07)+len(case_025)+len(case_035)+len(case_055)+len(case_01)+len(case_r1):135,:,:,:] =\
#  case_r2
#
# train[135:160,:,:,:] = case_r3
# # #
# for i in range(len(train)):
#     for idx in range(len(train[i])):
#         train[i,idx,:,:] = train[i,0,:,:] * train[i,idx,:,:]
# # # # print(len(case_025), len(case_035), len(case_055), len(case_07), len(case_01), len(case_r1))
# # #
# # # #np.save('./data/data_apr11', train)
# # #
# # #
# order = np.random.permutation(np.arange(len(train)))
# train = train[order]
#
# np.save('./data/data_apr20', train)
# i = 0
# #
# np.save('./data/train_apr12_2', train)
i = 0
test = case_025
while i < 1:
    print("data point: ", i)
    vis = sns.heatmap(test[i, 0], vmin=np.min(test[i, 0]), vmax=np.max(test[i, 0]))
    plt.show()
    vis =sns.heatmap(test[i,1],vmin=np.min(test[i,1]),vmax=np.max(test[i,1]))
    plt.show()
    vis = sns.heatmap(test[i, 4], vmin=np.min(test[i, 4]), vmax=np.max(test[i, 4]))
    plt.show()
    vis = sns.heatmap(test[i, 2], vmin=np.min(test[i, 2]), vmax=np.max(test[i, 2]))
    plt.show()
    vis = sns.heatmap(test[i, 5], vmin=np.min(test[i, 5]), vmax=np.max(test[i, 5]))
    plt.show()
    vis = sns.heatmap(test[i, 3], vmin=np.min(test[i, 3]), vmax=np.max(test[i, 3]))
    plt.show()
    vis = sns.heatmap(test[i, 6], vmin=np.min(test[i, 6]), vmax=np.max(test[i, 6]))
    plt.show()
    i += 1
#
