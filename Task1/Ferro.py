#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from Common import NeuralNet, MultiVariatePoly
import time
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(128)


# In[2]:


class Pinns:
    def __init__(self, n_int_, n_sb_, n_tb_, save_dir_, pre_model_save_path_):
        
        self.pre_model_save_path = pre_model_save_path_
        self.save_dir = save_dir_
        
        self.n_int = n_int_
        self.n_sb = n_sb_
        self.n_tb = n_tb_
        
        self.U_0 = 0
        # Extrema of the solution domain (t,x,y) in [0,1]x[0,50]x[0.50]
        self.domain_extrema = torch.tensor([[0, 1],  # Time dimension
                                            [0, 50],  # x dimension
                                            [0, 50]])  # y dimension
        # Number of space dimensions
        self.space_dimensions = 2
        
        # F Dense NN to approximate the solution of the underlying heat equation
        self.approximate_solution = NeuralNet(input_dimension=self.domain_extrema.shape[0], output_dimension=5,
                                              n_hidden_layers=4,
                                              neurons=20,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=42)
        '''self.approximate_solution = MultiVariatePoly(self.domain_extrema.shape[0], 3)'''
        if pre_model_save_path_:
            self.load_checkpoint()

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        # Training sets S_sb, S_tb, S_int as torch dataloader
        self.training_set_sb, self.training_set_tb, self.training_set_int = self.assemble_datasets()

    ################################################################################################
    # Function to linearly transform a tensor whose value are between 0 and 1
    # to a tensor whose values are between the domain extrema
    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]
    # Initial condition
    def initial_condition(self, x):
        return torch.full([len(x),5], self.U_0)
    
    ################################################################################################
    # Function returning the input-output tensor required to assemble the training set S_tb corresponding to the temporal boundary
    def add_temporal_boundary_points(self):
        t0 = self.domain_extrema[0, 0]
        input_tb = self.convert(self.soboleng.draw(self.n_tb))
        input_tb[:, 0] = torch.full(input_tb[:, 0].shape, t0)
        output_tb = self.initial_condition(input_tb[:, 1:])

        return input_tb, output_tb

# Function returning the input-output tensor required to assemble the training set S_sb corresponding to the spatial boundary
    def add_spatial_boundary_points(self):
        x0 = self.domain_extrema[1, 0]
        xL = self.domain_extrema[1, 1]
        y0 = self.domain_extrema[2, 0]
        yL = self.domain_extrema[2, 1]

        input_sb = self.convert(self.soboleng.draw(self.n_sb))

        input_sb_L = torch.clone(input_sb)
        input_sb_L[:, 1] = torch.full(input_sb_L[:, 1].shape, x0)
        input_sb_D = torch.clone(input_sb)
        input_sb_D[:, 2] = torch.full(input_sb_D[:, 2].shape, y0)
        input_sb_R = torch.clone(input_sb)
        input_sb_R[:, 1] = torch.full(input_sb_R[:, 1].shape, xL)
        input_sb_U = torch.clone(input_sb)
        input_sb_U[:, 2] = torch.full(input_sb_U[:, 2].shape, yL)

        output_sb_U = torch.zeros((input_sb.shape[0], 1))
        output_sb_D = torch.zeros((input_sb.shape[0], 1))
        output_sb_L = torch.zeros((input_sb.shape[0], 1))
        output_sb_R = torch.zeros((input_sb.shape[0], 1))
        
        return torch.cat([input_sb_U, input_sb_D, input_sb_L, input_sb_R], 0), torch.cat([output_sb_U, output_sb_D, output_sb_L, output_sb_R], 0)
    
    def add_interior_points(self):
        input_int = self.convert(self.soboleng.draw(self.n_int))
        output_int = torch.zeros((input_int.shape[0], 1))
        return input_int, output_int
    
    def assemble_datasets(self):
        input_sb, output_sb = self.add_spatial_boundary_points()   # S_sb
        input_tb, output_tb = self.add_temporal_boundary_points()  # S_tb
        input_int, output_int = self.add_interior_points()         # S_int

        training_set_sb = DataLoader(torch.utils.data.TensorDataset(input_sb, output_sb), batch_size=2*self.space_dimensions*self.n_sb, shuffle=False)
        training_set_tb = DataLoader(torch.utils.data.TensorDataset(input_tb, output_tb), batch_size=self.n_tb, shuffle=False)
        training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int, output_int), batch_size=self.n_int, shuffle=False)

        return training_set_sb, training_set_tb, training_set_int

    def compute_initial_condition(self, input_tb):
        u = self.approximate_solution(input_tb)
        u1 = u[:,0].reshape(-1,1)
        u2 = u[:,1].reshape(-1,1)
        varphi = u[:,2].reshape(-1,1)
        P1 = u[:,3].reshape(-1,1)
        P2 = u[:,4].reshape(-1,1) 
        
        residual_u1 = u1
        residual_u2 = u2
        residual_varphi = varphi 
        residual_P = P1**2+P2**2-0.04
        
        return residual_u1.reshape(-1, ), residual_u2.reshape(-1, ), residual_varphi.reshape(-1, ), residual_P.reshape(-1, )

    ##def compute_data_loss(self, input_data)

    
    # Function to compute the PDE residuals 
    def compute_pde_residual(self, input_int):
        input_int.requires_grad = True
        u = self.approximate_solution(input_int)
        u1 = u[:,0].reshape(-1,1)
        u2 = u[:,1].reshape(-1,1)
        varphi = u[:,2].reshape(-1,1)
        P1 = u[:,3].reshape(-1,1)
        P2 = u[:,4].reshape(-1,1)
        
        grad_u1 = torch.autograd.grad(u1.sum(), input_int, create_graph=True)[0]
        u1_t, u1_1, u1_2 = grad_u1[:, 0], grad_u1[:, 1], grad_u1[:, 2]
        
        grad_u1_1 = torch.autograd.grad(u1_1.sum(), input_int, create_graph=True)[0]
        u1_11, u1_12 = grad_u1_1[:, 1], grad_u1_1[:, 2]        
        grad_u1_2 = torch.autograd.grad(u1_2.sum(), input_int, create_graph=True)[0]
        u1_21, u1_22 = grad_u1_2[:, 1], grad_u1_2[:, 2]  
        
        grad_u2 = torch.autograd.grad(u2.sum(), input_int, create_graph=True)[0]
        u2_t, u2_1, u2_2 = grad_u2[:, 0], grad_u2[:, 1], grad_u2[:, 2]

        grad_u2_1 = torch.autograd.grad(u2_1.sum(), input_int, create_graph=True)[0]
        u2_11, u2_12 = grad_u2_1[:, 1], grad_u2_1[:, 2]        
        grad_u2_2 = torch.autograd.grad(u1_2.sum(), input_int, create_graph=True)[0]
        u2_21, u2_22 = grad_u2_2[:, 1], grad_u2_2[:, 2]
        
        grad_varphi = torch.autograd.grad(varphi.sum(), input_int, create_graph=True)[0]
        varphi_1, varphi_2 = grad_varphi[:, 1], grad_varphi[:, 2]
        grad_varphi_1 = torch.autograd.grad(varphi_1.sum(), input_int, create_graph=True)[0]
        varphi_11 = grad_varphi_1[:, 1]
        grad_varphi_2 = torch.autograd.grad(varphi_2.sum(), input_int, create_graph=True)[0]
        varphi_22 = grad_varphi_1[:, 2]        
        
        grad_P1 = torch.autograd.grad(P1.sum(), input_int, create_graph=True)[0]
        P1_t, P1_1, P1_2 = grad_P1[:, 0], grad_P1[:, 1], grad_P1[:, 2]

        grad_P1_1 = torch.autograd.grad(P1_1.sum(), input_int, create_graph=True)[0]
        P1_11, P1_12 = grad_P1_1[:, 1], grad_P1_1[:, 2]        
        grad_P1_2 = torch.autograd.grad(P1_2.sum(), input_int, create_graph=True)[0]
        P1_21, P1_22 = grad_P1_2[:, 1], grad_P1_2[:, 2] 
        
        grad_P2 = torch.autograd.grad(P2.sum(), input_int, create_graph=True)[0]
        P2_t, P2_1, P2_2 = grad_P2[:, 0], grad_P2[:, 1], grad_P2[:, 2]        

        grad_P2_1 = torch.autograd.grad(P2_1.sum(), input_int, create_graph=True)[0]
        P2_11, P2_12 = grad_P2_1[:, 1], grad_P2_1[:, 2]        
        grad_P2_2 = torch.autograd.grad(P1_2.sum(), input_int, create_graph=True)[0]
        P2_21, P2_22 = grad_P2_2[:, 1], grad_P2_2[:, 2]

        residual_PDE_1 = 174.6*u1_11-2*0.089*P1*P1_1+0.5*5*P1_11+111.1*(u1_22+u2_22)-2*0.032*(P1*P2_2+P2*P1_2)+0.5*5*P1_11
        residual_PDE_2 = 174.6*u2_22-2*0.089*P2*P2_2+0.5*5*P2_22+111.1*(u1_21+u2_21)-2*0.032*(P1*P2_1+P2*P1_1)+0.5*5*P2_22
        residual_PDE_3 = -0.5841*varphi_11-0.5841*varphi_22+P1+P2
        residual_PDE_4 = P1_t-(-2*0.148*P1-4*0.031*P1**3+2*0.63*P1+P2**2+6*0.25*P1**5+0.97*(2*P1*P2**4+4*P1**3*P2**2))\
        +0.15*P1_11-0.15*P2_21+0.15*(P2_12+P1_22)+0.089*u1_1*P1-0.026*u2_2*P1+0.032*(u1_2+u2_1)*P1+5*u1_11-varphi_1
        residual_PDE_5 = P2_t-(-2*0.148*P2-4*0.031*P2**3+2*0.63*P2*P1**2+6*0.25*P2**5+0.97*(2*P2*P1**4+4*P2**3*P1**2))\
        +0.15*P2_11+0.15*P2_22+0.032*(u1_2+u2_1)*P1-0.026*u1_1*P2+0.089*u2_2*P2+5*u2_22-varphi_2
        
        return residual_PDE_1.reshape(-1, ), residual_PDE_2.reshape(-1, ), residual_PDE_3.reshape(-1, ), residual_PDE_4.reshape(-1, ), residual_PDE_5.reshape(-1, )
    
    def compute_bc_residual(self, input_bc):
        input_bc.requires_grad = True
        
        u = self.approximate_solution(input_bc)
        u1 = u[:,0].reshape(-1,1)
        u2 = u[:,1].reshape(-1,1)
        varphi = u[:,2].reshape(-1,1)
        
        residual_u1 = u1
        residual_u2 = u2
        residual_varphi = varphi
#        residual = torch.abs(residual_Tf) + torch.abs(residual_Ts)
        
        return residual_u1.reshape(-1, ), residual_u2.reshape(-1, ), residual_varphi.reshape(-1, )

    # Function to compute the total loss (weighted sum of spatial boundary loss, temporal boundary loss and interior loss)
    def compute_loss(self, inp_train_sb, u_train_sb, inp_train_tb, u_train_tb, inp_train_int, verbose=True):
        #u_pred_tb = self.apply_initial_condition(inp_train_tb)

        #assert (u_pred_tb.shape[1] == u_train_tb.shape[1])

        r_int_1, r_int_2, r_int_3, r_int_4, r_int_5 = self.compute_pde_residual(inp_train_int)
        r_sb_u1, r_sb_u2, r_sb_varphi  = self.compute_bc_residual(inp_train_sb)
        r_tb_u1, r_tb_u2, r_tb_varphi, r_tb_P = self.compute_initial_condition(inp_train_tb)

        loss_sb = torch.mean(abs(r_sb_u1) ** 2) + torch.mean(abs(r_sb_u2) ** 2) + torch.mean(abs(r_sb_varphi) ** 2)
        loss_tb = torch.mean(abs(r_tb_u1) ** 2) + torch.mean(abs(r_tb_u2) ** 2) + torch.mean(abs(r_tb_varphi) ** 2) + torch.mean(abs(r_tb_P) ** 2)
        loss_int = torch.mean(abs(r_int_1) ** 2) + torch.mean(abs(r_int_2) ** 2) + torch.mean(abs(r_int_3) ** 2) + torch.mean(abs(r_int_4) ** 2) + torch.mean(abs(r_int_5) ** 2)
        
#         r_int = self.compute_pde_residual(inp_train_int)
#         r_tb = u_train_tb - u_pred_tb
#         r_sb_0 = self.compute_bc0_residual(inp_train_sb[0:self.n_sb,:])
#         r_sb_L = self.compute_bcL_residual(inp_train_sb[self.n_sb:,:])
        
#         loss_tb = torch.mean(abs(r_tb) ** 2)
#         loss_int = torch.mean(abs(r_int) ** 2)
#         loss_sb = torch.mean(abs(r_sb_0) ** 2) + torch.mean(abs(r_sb_L) ** 2)

        loss_u = loss_sb + loss_tb

        loss = torch.log10(loss_sb + loss_tb + loss_int)
        if verbose: print("Total loss: ", round(loss.item(), 4), "| PDE Loss: ", round(torch.log10(loss_u).item(), 4), "| Function Loss: ", round(torch.log10(loss_int).item(), 4))

        return loss

    ################################################################################################
    def fit(self, num_epochs, optimizer, verbose=True):
        history = list()

        # Loop over epochs
        for epoch in range(num_epochs):
            if verbose: print("################################ ", epoch, " ################################")

            for j, ((inp_train_sb, u_train_sb), (inp_train_tb, u_train_tb), (inp_train_int, u_train_int)) in enumerate(zip(self.training_set_sb, self.training_set_tb, self.training_set_int)):
                def closure():
                    optimizer.zero_grad()
                    loss = self.compute_loss(inp_train_sb, u_train_sb, inp_train_tb, u_train_tb, inp_train_int, verbose=verbose)
                    loss.backward()

                    history.append(loss.item())
                    return loss
                

                optimizer.step(closure=closure)

        print('Final Loss: ', history[-1])

        return history
    
    def save_checkpoint(self):
        '''save model and optimizer'''
        torch.save({
            'model_state_dict': self.approximate_solution.state_dict()
        }, self.save_dir)
        
    def load_checkpoint(self):
        '''load model and optimizer'''
        checkpoint = torch.load(self.pre_model_save_path)
        self.approximate_solution.load_state_dict(checkpoint['model_state_dict'])
        print('Pretrained model loaded!')

    ################################################################################################


# In[3]:


n_int = 1000
n_sb = 100
n_tb = 100


pre_model_save_path = './model/Adam1000_sqloss.pt'
save_path = './model/LBFGS2000_sqloss.pt'
pinn = Pinns(n_int, n_sb, n_tb, save_path, pre_model_save_path)


# In[ ]:


# Plot the input training points
input_sb_, output_sb_ = pinn.add_spatial_boundary_points()
input_tb_, output_tb_ = pinn.add_temporal_boundary_points()
input_int_, output_int_ = pinn.add_interior_points()

plt.figure(figsize=(5, 5), dpi=150)
plt.scatter(input_sb_[:, 1].detach().numpy(), input_sb_[:, 2].detach().numpy(), c='blue', label="Boundary Points")
plt.scatter(input_int_[:, 1].detach().numpy(), input_int_[:, 2].detach().numpy(), c='green', label="Interior Points")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Rest of the code remains the same...

# Create a 3D figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for spatial boundary points
ax.scatter(input_sb_[:, 1].detach().numpy(), input_sb_[:, 2].detach().numpy(), input_sb_[:, 0].detach().numpy(), c='blue', label="Boundary Points")

# Scatter plot for interior points
ax.scatter(input_int_[:, 1].detach().numpy(), input_int_[:, 2].detach().numpy(), input_int_[:, 0].detach().numpy(), c='orange', label="Interior Points")

# Scatter plot for initial points
ax.scatter(input_tb_[:, 1].detach().numpy(), input_tb_[:, 2].detach().numpy(), input_tb_[:, 0].detach().numpy(), c='green', label="Initial Points")

# Set labels for each axis
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('t')

# Add a legend
ax.legend()

# Show the 3D plot
plt.show()


# In[ ]:


n_epochs = 1
optimizer_LBFGS = optim.LBFGS(pinn.approximate_solution.parameters(),
                              lr=float(0.5),
                              max_iter=50000,
                              max_eval=50000,
                              history_size=150,
                              line_search_fn="strong_wolfe",
                              tolerance_change=1.0 * np.finfo(float).eps)
optimizer_ADAM = optim.Adam(pinn.approximate_solution.parameters(),
                            lr=float(0.001))

hist = pinn.fit(num_epochs=n_epochs,
                optimizer=optimizer_LBFGS,
                verbose=True)

pinn.save_checkpoint()


# In[ ]:




