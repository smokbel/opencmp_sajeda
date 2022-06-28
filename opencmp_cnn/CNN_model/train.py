import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import sys
import os
from numpy import asarray
from numpy import savetxt
from autoencode import NeuralNet
import time


param_file = sys.argv[1]

time_start = time.perf_counter()
with open(param_file) as json_file:
    param = json.load(json_file)

# Construct model and dataset
model = NeuralNet()
# model.load_state_dict(torch.load('./goodmodels/modelmay15_5.pth'))
# model.eval()
model = model.double()
data = np.load('../data/data_apr20.npy')
data = data[0:100,:,:,:]

# Define an optimizer and the loss function
optimizer = optim.RMSprop(model.parameters(), lr=param['learning_rate'])
# loss= torch.nn.MSELoss()
loss = torch.nn.MSELoss()

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("TOTAL PARAMS:", pytorch_total_params)

obj_vals = []
cross_vals = []
num_epochs = int(param['num_epochs'])
batch = int(param['batch'])

model = model.double()

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

data_size = len(data)
i = 0
while i + batch < data_size:

    # train_x = x.reshape(25,1,100,128,64)
    # target_x = y.reshape(25,1,100,128,64)
    # train_x = x.reshape(2500,1,128,64)
    # target_x = y.reshape(2500,1,128,64)

    # x_t = train_x[i:i+batch,0:1,:,:]
    # y_t = target_x[i+batch:i+batch+batch,0:1,:,:]
    x_t = data[i:i + batch, :, :, :]
    y_t = data[i:i + batch, 4:7, :, :]

    # print(np.shape(x))
    # print(np.shape(y))
    print("Testing:", i, "to", i + batch)
    # print("Target:",i+batch, "to", i+batch+batch)
    # print("Batch:", i+batch)
    for epoch in range(1, num_epochs + 1):

        train_val = model.backprop(x_t, y_t, loss, epoch, optimizer)
        # test_val= model.test(x_test, y_test, loss, epoch)
        # cross_vals.append(test_val)
        obj_vals.append(train_val)
        if not ((epoch + 1) % param['display_epochs']):
            print('Epoch [{}/{}]'.format(epoch + 1, num_epochs) + \
                  '\tTraining Loss: {:.4f}'.format(train_val) + \
                  '\tTest Loss: {:.4f}'.format(train_val))
    i += batch

# project_path = '/Users/sajedamokbel/Desktop/code/of_opencmp/opencmp_cnn/CNN_model'
# path = os.path.join(project_path, 'model_apr20.pth')
torch.save(model.state_dict(), 'model_apr21.pth')


# Final training loss
print('Final training loss: {:.4f}'.format(obj_vals[-1]))
print('Final test loss: {:.4f}'.format(cross_vals[-1]))
train_loss = []
test_loss = []
i = 0
print(len(obj_vals))
while i + 100 < len(obj_vals) - 1:
    train_loss.append(obj_vals[100 + i - 1])
    i += 100
i = 0
while i + 100 < len(cross_vals) - 1:
    test_loss.append(cross_vals[100 + i - 1])
    i += 100

train_dropout04 = asarray(train_loss)
test_dropout04 = asarray(test_loss)
# save to csv file
savetxt('train_bt50.csv', train_dropout04, delimiter=',')
savetxt('test_bt50.csv', test_dropout04, delimiter=',')
plt.plot(train_loss, label="Training loss", color="rebeccapurple")
plt.plot(test_loss, label="Test loss", color="lightseagreen")
plt.legend()
plt.savefig('results-20.png')

time_elapsed = (time.perf_counter() - time_start)
print(time_elapsed)