
# coding: utf-8

# In[1]:


from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('run', 'generate_sine_wave.ipynb')


# In[2]:


class Sequence(nn.Module):
    def __init__(self, dims):
        """
        hidden_dims: list variable containing layer dimension information
                      [input_dim, hidden_dims, output_dim]  
        """
        
        super().__init__()
        self.dims = dims
        self.lstms = nn.ModuleList()
        for lstm_in, lstm_out in zip(dims[0:-2], dims[1:-1]):
            self.lstms.append(nn.LSTMCell(lstm_in, lstm_out))
        self.linear = nn.Linear(dims[-2], dims[-1])    
        
        return
    
    def forward(self, input_X, future=0):
        outputs = []
        hidden_states = []
        cell_states = []
        
        # Initialize hidden and cell states
        batch_size = input_X.size(0)
        for dim in self.dims[1: -1]:
            hidden_states.append(torch.zeros(batch_size, dim, dtype=torch.double))
            cell_states.append(torch.zeros(batch_size, dim, dtype=torch.double))
            
        timesteps = input_X.size(1)
        for i, input_t in enumerate(input_X.chunk(timesteps, dim=1)):
            hidden_states[0], cell_states[0] = self.lstms[0](
                input_t, (hidden_states[0], cell_states[0]))
            for j in range(1, len(self.lstms)):
                hidden_states[j], cell_states[j] = self.lstms[j](
                    hidden_states[j-1], (hidden_states[j], cell_states[j]))
            output = self.linear(hidden_states[-1])
            outputs += [output]
            
        # if we should predict the future
        for i in range(future):
            hidden_states[0], cell_states[0] = self.lstms[0](
                output, (hidden_states[0], cell_states[0]))
            for j in range(1, len(self.lstms)):
                hidden_states[j], cell_states[j] = self.lstms[j](
                    hidden_states[j-1], (hidden_states[j], cell_states[j]))
            output = self.linear(hidden_states[-1])
            outputs += [output]
            
        outputs = torch.stack(outputs, 1).squeeze(2)
        
        return outputs


# In[ ]:


if __name__ == '__main__':
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    data = torch.load('traindata.pt')
    input = torch.from_numpy(data[3:, :-1])
    target = torch.from_numpy(data[3:, 1:])
    test_input = torch.from_numpy(data[:3, :-1])
    test_target = torch.from_numpy(data[:3, 1:])
    # build the model
    
    # Pass dimensions explicitly as parameter to the model(seq)
    dims = [1, 51, 51, 1]
    seq = Sequence(dims)
    
    seq.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    #begin to train
    for i in range(15):
        print('STEP: ', i)
        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = 1000
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            y = pred.detach().numpy()
        # draw the result
        plt.figure(figsize=(30,10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.savefig('predict%d.pdf'%i)
        plt.close()

