
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


# In[2]:


# Generate data
num_data = 5
input_size = 20
hidden_sizes = [15, 10]
input_X = torch.randn(num_data, input_size)


# In[3]:


# Hyperparameters
num_epochs = 10000
batch_size = 128
learning_rate = 1e-3
weight_decay = 1e-5
BETA = 3
RHO = 0.01


# In[4]:


class AutoEncoder(nn.Module):
    """
    1-layer autoencoder
    """
    
    def __init__(self, input_size, hidden_size, lr=1e-1, bn=False):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        if bn:
            self.encode = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU()
            )
            self.decode = nn.Sequential(
                nn.Linear(hidden_size, input_size),
                nn.BatchNorm1d(input_size),
            )
        else:
            self.encode = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU()
            )
            self.decode = nn.Sequential(
                nn.Linear(hidden_size, input_size),
            )

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=1e-5)
        
    def forward(self, x):
        # Train each autoencoder individually
        x = x.detach()
        z = self.encode(x)
        
        if self.training:
            x_rec = self.decode(z)
            loss = self.criterion(x_rec, Variable(x.data, requires_grad=False))
            
            # Sparsity regularization
            rho = torch.FloatTensor([RHO]*self.hidden_size).unsqueeze(0)
            rho_hat = torch.sum(z, dim=0, keepdim=True)
            sparsity_penalty = BETA*F.kl_div(rho, rho_hat)
            
            loss += sparsity_penalty
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return z.detach()
    
    def reconstruct(self, z):
        return self.decode(z)


# In[5]:


class StackedAutoEncoder(nn.Module):
    """
    Multi-layer(?) autoencoder
    Layer-wise training then fine-tuning
    """
    
    def __init__(self, input_size, hidden_sizes, lr=1e-1, bn=False):
        super().__init__()
        
        self.aes = nn.ModuleList()
        hidden_sizes = [input_size]+hidden_sizes
        for ae_in, ae_out in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            ae = AutoEncoder(ae_in, ae_out, lr=lr, bn=bn)
            self.aes.append(ae)
             
    def forward(self, x):
        for ae in self.aes:
            x = ae(x)
        
        if self.training:
            return x
        
        else:
            return x, self.reconstruct(x)
        
    def reconstruct(self, x):
        for ae in self.aes[::-1]:
            x = ae.reconstruct(x)
            
        return x


# In[6]:


# Construct model
sae = StackedAutoEncoder(input_size, hidden_sizes, lr=learning_rate, bn=False)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    sae.parameters(), lr=learning_rate, weight_decay=1e-5)


# In[7]:


# Training
for epoch in range(num_epochs):
    sae.train()
    
    # Forward
    encoded = sae(input_X)
    output = sae.reconstruct(encoded)   
    loss = criterion(output, input_X)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print('[epoch {}] loss: {:.4f}'.format(epoch+1, loss.data[0]))


# In[8]:


print(input_X[0])


# In[9]:


print(output[0])

