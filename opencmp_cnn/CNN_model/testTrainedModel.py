import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.optim as optim
import sys
from timeit import default_timer as timer

from autoencode import NeuralNet

start = timer()
# Load the CNN and its parameters.
model = NeuralNet()
model = model.double()
param_file = 'param.json'

with open(param_file) as json_file:
    param = json.load(json_file)

optimizer = optim.RMSprop(model.parameters(), lr=param['learning_rate'])
#model_save = param['model_save']

model.load_state_dict(torch.load(str('model_apr21.pth')))
model.eval()
model = model.double()


data = np.load('../data/data_apr20.npy')
in_value = data[8:9,:,:,:]
in_model = torch.from_numpy(in_value)

prediction_recurrent = model.forward(in_model)
prediction_recurrent = prediction_recurrent.detach().numpy()
end = timer()
print("TIME", end - start)
test = prediction_recurrent

for k in range(1):
    vis = sns.heatmap(in_value[0,0], vmin=np.min(in_value[0,0]), vmax=np.max(in_value[0,0]))
    plt.show()

    vis =sns.heatmap(in_value[0,1],vmin=np.min(in_value[0,1]),vmax=np.max(in_value[0,1]))
    plt.show()
    vis = sns.heatmap(test[0, 0], vmin=np.min(test[0, 0]), vmax=np.max(test[0, 0]))
    plt.show()
    vis = sns.heatmap(in_value[0,2], vmin=np.min(in_value[0,2]), vmax=np.max(in_value[0,2]))
    plt.show()
    vis = sns.heatmap(test[0,1], vmin=np.min(test[0,1]), vmax=np.max(test[0, 1]))
    plt.show()
    vis = sns.heatmap(in_value[0,3], vmin=np.min(in_value[0,3]), vmax=np.max(in_value[0,3]))
    plt.show()
    vis = sns.heatmap(test[0, 2], vmin=np.min(test[0, 2]), vmax=np.max(test[0, 2]))
    plt.show()
in_value = torch.from_numpy(prediction_recurrent)

prediction_recurrent = model.forward(in_value)
prediction_recurrent = prediction_recurrent.detach().numpy()
end = timer()
print("TIME", end - start)
test = prediction_recurrent
i=0
for k in range(1):
    vis = sns.heatmap(test[i, 0], vmin=np.min(test[i, 0]), vmax=np.max(test[i, 0]))
    plt.show()
    vis =sns.heatmap(test[i,1],vmin=np.min(test[i,1]),vmax=np.max(test[i,1]))
    plt.show()
    vis = sns.heatmap(test[i, 2], vmin=np.min(test[i, 2]), vmax=np.max(test[i, 2]))
    plt.show()



# prediction_recurrent = x_mse

# def divergence(f):
#     """
#     Computes the divergence of the vector field f, corresponding to dFx/dx + dFy/dy + ...
#     :param f: List of ndarrays, where every item of the list is one dimension of the vector field
#     :return: Single ndarray of the same shape as each of the items in f, which corresponds to a scalar field
#     """
#     num_dims = len(f)
#     return np.ufunc.reduce(np.add, [np.gradient(f[num_dims - i - 1], axis=i) for i in range(num_dims)])


# prediction_recurrent = x_mse
# vis =sns.heatmap(prediction_recurrent[0][1],vmin=0,vmax=1,cmap="mako")
# plt.show()
# lala = np.zeros((1,4,100,60), dtype=float)
# lala[0][0] = x_mse[0][0]
# lala[0,1:4] = prediction_recurrent[0,0:3]
# lala = torch.from_numpy(lala)
# vis =sns.heatmap(model.forward(lala)[0][1],vmin=0,vmax=1,cmap="mako")
# plt.show()
# vis =sns.heatmap(x_mse[0][5],vmin=0,vmax=1,cmap="mako")
# plt.show()
# error = np.zeros((100,60), dtype=float)
# for m in range(100):
#     for n in range(60):
#         error[m,n] = np.abs(x_mse[0][5][m,n]-prediction_recurrent[0][1][m,n])

# vis =sns.heatmap(error,vmin=0,vmax=0.4, cmap="mako")

# plt.show()
# for i in range(20):
#     print(np.shape(prediction_recurrent[0][0][i]))
#     test = prediction_recurrent[0][0][i]
#     vis =sns.heatmap(test,vmin=np.min(prediction_recurrent[0][0][i]),vmax=np.max(prediction_recurrent[0][0][i]))
#     plt.show()

# vis =sns.heatmap(prediction_recurrent[i][1],vmin=0,vmax=1)
# plt.show()

# vis =sns.heatmap(prediction_recurrent[i][2],vmin=0,vmax=1)
# plt.show()

# vis =sns.heatmap(prediction_recurrent[i][3],vmin=0,vmax=1)
# plt.show()S

# vis =sns.heatmap(prediction_recurrent[i][4],vmin=0,vmax=1)
# plt.show()

# vis =sns.heatmap(prediction_recurrent[i][5],vmin=0,vmax=1)
# plt.show()

# vis =sns.heatmap(prediction_recurrent[i][6],vmin=0,vmax=1)
# plt.show()

# vis =sns.heatmap(prediction_recurrent[i][7],vmin=0,vmax=1)
# plt.show()

# vis =sns.heatmap(x_mse[i][5],vmin=0,vmax=1)
# plt.show()

# vis =sns.heatmap(x_mse[i][6],vmin=0,vmax=1)
# plt.show()

# vis =sns.heatmap(x_mse[i][7],vmin=0,vmax=1)
# plt.show()

# vis =sns.heatmap(x_mse[i][5],vmin=0,vmax=1)
# plt.show()

# vis =sns.heatmap(x_mse[i][6],vmin=0,vmax=1)
# plt.show()

# vis =sns.heatmap(x_mse[i][7],vmin=0,vmax=1)
# plt.show()
# vis =sns.heatmap(x_mse[i][1],vmin=0,vmax=1)
# plt.show()

# vis =sns.heatmap(x_mse[i][2],vmin=0,vmax=1)
# plt.show()

# vis =sns.heatmap(x_mse[i][3],vmin=0,vmax=1)
# plt.show()

# vis =sns.heatmap(prediction_recurrent[i][6],vmin=0,vmax=1)
# plt.show()

# vis =sns.heatmap(prediction_recurrent[i][7],vmin=0,vmax=1)
# plt.show()

# lala = np.load('./results/prediction_all.npy')
# vis =sns.heatmap(prediction_recurrent[1700][0],vmin=0,vmax=1)
# plt.show()
# for i in range(300):
#     for epoch in range(4):

#         train_val= model.backprop(x, y, loss, epoch, optimizer)
#         obj_vals.append(train_val)
#         test_val= model.test(x, y, loss, epoch)
#         cross_vals.append(test_val)
#         epoch += 1

#         if not ((epoch + 1) % param['display_epochs']):
#             print('Epoch [{}/{}]'.format(epoch+1,

#             )+\
#             '\tTraining Loss: {:.4f}'.format(train_val)+\
#             '\tTest Loss: {:.4f}'.format(test_val))

#     x = torch.from_numpy(x)
#     #in_val = np.load('./results/prediction_iter_no_train_first_p.npy')
#     prediction_n = model.forward(x)
#     prediction_n = prediction_n.detach().numpy()
#     np.save('./results/prediction_all', prediction_n)
#     x = np.load('./results/prediction_all.npy')
#     print("got the new x", print(x[0]))
#     i += 1

# #Final training loss
# print('Final training loss: {:.4f}'.format(obj_vals[-1]))
# print('Final test loss: {:.4f}'.format(cross_vals[-1]))

# project_path = 'C:/Users/Owner/Desktop/CFDNet'
# path = os.path.join(project_path, 'modelRecurrent_300.pth')
# torch.save(model.state_dict(), path)