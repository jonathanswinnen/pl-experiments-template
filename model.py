"""
Example dummy model. This is just a linear model with a single linear layer.
"""

import torch

class DummyModel(torch.nn.Module):
    def __init__(self, in_sz, out_sz):
        super(DummyModel, self).__init__()
        self.linear = torch.nn.Linear(in_sz, out_sz)
    
    def forward(self, x):
        y = self.linear(x)
        return y
    
    