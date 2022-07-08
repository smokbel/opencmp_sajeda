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
from fcnn import NeuralNet
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

param_file = sys.argv[1]

time_start = time.perf_counter()
with open(param_file) as json_file:
    param = json.load(json_file)

# Construct model and dataset
model = NeuralNet()
# model.load_state_dict(torch.load('./goodmodels/modelmay15_5.pth'))
# model.eval()
model = model.double()
model.to(device)

data = np.load('data_fc_test.npy')

# Define an optimizer and the loss function
#optimizer = optim.RMSprop(model.parameters(), lr=param['learning_rate'])
optimizer = optim.Adagrad(model.parameters(), lr=param['learning_rate'], lr_decay=1)
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

data_size = 60
i = 0
new_data = data[:,:,:,0]
print("New data shape: ", np.shape(new_data))
while i + batch < data_size:

    # train_x = x.reshape(25,1,100,128,64)
    # target_x = y.reshape(25,1,100,128,64)
    # train_x = x.reshape(2500,1,128,64)
    # target_x = y.reshape(2500,1,128,64)

    # x_t = train_x[i:i+batch,0:1,:,:]
    # y_t = target_x[i+batch:i+batch+batch,0:1,:,:]
    x_t = new_data[i:i + batch, :, :]
    y_t = new_data[i:i + batch, 4:6, :]

    x_t = torch.from_numpy(x_t)
    y_t = torch.from_numpy(y_t)

    x_t = x_t.to(device)
    y_t = y_t.to(device)

    # inputs = torch.from_numpy(x)
    # targets = torch.from_numpy(y)
    # inputs = inputs.to(device)

    # print(np.shape(x))
    # print(np.shape(y))
    # print("Target:",i+batch, "to", i+batch+batch)
    # print("Batch:", i+batch)
    for epoch in range(1, num_epochs + 1):
        #y_t = y_t.reshape(2, length)
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
torch.save(model.state_dict(), 'model_jul7_2.pth')


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