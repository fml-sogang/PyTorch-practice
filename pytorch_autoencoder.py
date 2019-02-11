
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


# In[2]:


# Hyperparameters
num_epochs = 1000
batch_size = 128
learning_rate = 1e-3


# In[3]:


# Generate data
num_data = 5
input_size = 20
hidden_size = 10
img_1 = torch.randn(num_data, input_size)


# In[4]:


class AutoEncoder(nn.Module):
    """
    1-layer autoencoder
    """
    
    def __init__(self, input_size, hidden_size, bn=False):
        super().__init__()
        self.bn = bn
        
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, input_size)
        
        if self.bn:
            self.bn1 = nn.BatchNorm1d(hidden_size)        
            self.bn2 = nn.BatchNorm1d(input_size)
        
    def encode(self, x):
        x = self.linear1(x)
        if self.bn:
            x = self.bn1(x)
        z = F.relu(x)
        
        return z
    
    def decode(self, z):
        z = self.linear2(z)
        if self.bn:
            z = self.bn2(z)
        x_rec = z
        
        return x_rec
        
    def forward(self, x):
        z = self.encode(x)
        x_rec = self.decode(z)
        
        return x_rec


# In[5]:


# Construct model
ae = AutoEncoder(input_size, hidden_size, bn=False)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    ae.parameters(), lr=learning_rate, weight_decay=1e-5)


# In[6]:


# Training
for epoch in range(num_epochs):
    # Forward
    output = ae(img_1)
    loss = criterion(output, img_1)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print('[epoch {}] loss: {:.4f}'.format(epoch+1, loss.data[0]))


# In[7]:


print(img_1[0])


# In[8]:


print(output[0])

