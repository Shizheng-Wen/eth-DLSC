import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
import pandas as pd


def activation(name):
    if name in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif name in ['relu', 'ReLU']:
        return nn.ReLU(inplace=True)
    elif name in ['lrelu', 'LReLU']:
        return nn.LeakyReLU(inplace=True)
    elif name in ['sigmoid', 'Sigmoid']:
        return nn.Sigmoid()
    elif name in ['softplus', 'Softplus']:
        return nn.Softplus(beta=4)
    elif name in ['celu', 'CeLU']:
        return nn.CELU()
    elif name in ['elu']:
        return nn.ELU()
    elif name in ['mish']:
        return nn.Mish()
    else:
        raise ValueError('Unknown activation function')


################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    ######################### TO DO ####################################

    def forward(self, x):
        batchsize = x.shape[0]
        # x.shape == [batch_size, in_channels, number of grid points]
        # hint: use torch.fft library torch.fft.rfft
        # use DFT to approximate the fourier transform

        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


####################################################################


class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 1  # pad the domain if input is non-periodic
        self.linear_p = nn.Linear(2, self.width)  # input channel is 2: (u0(x), x)

        self.spect1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.spect2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.spect3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.lin0 = nn.Conv1d(self.width, self.width, 1)
        self.lin1 = nn.Conv1d(self.width, self.width, 1)
        self.lin2 = nn.Conv1d(self.width, self.width, 1)

        self.linear_q = nn.Linear(self.width, 32)
        self.output_layer = nn.Linear(32, 2)

        self.activation = torch.nn.Tanh()

    def fourier_layer(self, x, spectral_layer, conv_layer):
        return self.activation(spectral_layer(x) + conv_layer(x))

    def linear_layer(self, x, linear_transformation):
        return self.activation(linear_transformation(x))

    def forward(self, x):
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)
        x = self.linear_p(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic

        x = self.fourier_layer(x, self.spect1, self.lin0)
        x = self.fourier_layer(x, self.spect2, self.lin1)
        x = self.fourier_layer(x, self.spect3, self.lin2)

        # x = x[..., :-self.padding]  # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)

        x = self.linear_layer(x, self.linear_q)
        x = self.output_layer(x)
        return x


torch.manual_seed(0)
np.random.seed(0)

n_train = 100

data = pd.read_table('TrainingData.txt', sep=',').values
window_size = 35

x = torch.zeros([5, window_size, 2], dtype=torch.float)
y = torch.zeros([5, window_size, 2], dtype=torch.float)

x_fluid = torch.tensor(data[:, 0:3], dtype=torch.float)

min_vals, _ = torch.min(x_fluid, dim=0, keepdim=True)
max_vals, _ = torch.max(x_fluid, dim=0, keepdim=True)

x_fluid = (x_fluid - min_vals) / (max_vals - min_vals)

# Data assemble
for i in range(5):
    x[i, :, 0:2] = x_fluid[i * window_size: (i + 1) * window_size, 1:3]
    y[i, :, 0:2] = x_fluid[(i + 1) * window_size: (i + 2) * window_size, 1:3]

batch_size = 2

training_set = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

learning_rate = 0.01

epochs = 1000
#epochs = 2
step_size = 50
gamma = 0.5

modes = 16
width = 64

# model

fno = FNO1d(modes, width)

optimizer = Adam(fno.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

l = torch.nn.MSELoss()
freq_print = 1
for epoch in range(epochs):
    train_mse = 0.0
    for step, (input_batch, output_batch) in enumerate(training_set):
        optimizer.zero_grad()
        output_pred_batch = fno(input_batch).squeeze(2)
        loss_f = l(output_pred_batch, output_batch)
        loss_f.backward()
        optimizer.step()
        train_mse += loss_f.item()
    train_mse /= len(training_set)

    scheduler.step()
    if epoch % freq_print == 0: print("######### Epoch:", epoch, " ######### Train Loss:", train_mse)

input_function_test_n = torch.zeros([1, 35, 2],dtype = float)
input_function_test_n[0,:,0:1] = x_fluid[0:35, 0:1]
input_function_test_n[0,:,1:2] = x_fluid[5 *35: 6*35, 1:2]
output_function_test_pred_n = fno(input_function_test_n.float())
test_data = pd.read_table('TestingData.txt', sep=',').values
print(output_function_test_pred_n.shape)

time = torch.tensor(data[:, 0:1], dtype=torch.float)
p_time = torch.tensor(test_data[:, 0:1], dtype=torch.float)
Tf = x_fluid[:, 1:2]
plt.figure(dpi=100)
plt.grid(True, which="both", ls=":")
plt.plot(time, x_fluid[:,1:2], label="Fluid", c="C0", lw=1.5)
plt.plot(time, x_fluid[:,2:3], label="Fluid", c="C1", lw=1.5)
plt.plot(p_time, output_function_test_pred_n[0,:,0].detach(), label="Solid", c="C0", lw=1.5)
plt.plot(p_time, output_function_test_pred_n[0,:,1].detach(), label="Solid", c="C1", lw=1.5)
# #plt.scatter(input_function_test_n[0,:,1].detach(), output_function_test_pred_n[0].detach(), label="Approximate Solution", s=8, c="C0")

# plt.legend()
print(test_data.shape)

