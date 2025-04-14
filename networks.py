import torch
import torch.nn as nn

class Net_FFN(nn.Module):
    def __init__(self, dim, nOut, n_layers, vNetWidth, activation = "relu", activation_output="id", batchnorm=False):
        super(Net_FFN, self).__init__()
        self.dim = dim
        self.nOut = nOut
        self.batchnorm=batchnorm
        
        if activation=="relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        elif activation=="tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError("unknown activation function {}".format(activation))

        if activation_output == "id":
            self.activation_output = nn.Identity()
        elif activation_output == "softplus":
            self.activation_output = nn.Softplus()
        else:
            raise ValueError("unknown output activation function {}".format(activation_output))
        
        self.i_h = self.hiddenLayerT1(dim, vNetWidth)
        self.h_h = nn.ModuleList([self.hiddenLayerT1(vNetWidth, vNetWidth) for l in range(n_layers-1)])
        self.h_o = self.outputLayer(vNetWidth, nOut)
        
    
    def hiddenLayerT1(self, nIn, nOut):
        if self.batchnorm:
            layer = nn.Sequential(nn.Linear(nIn,nOut,bias=True),
                                  self.activation)   
        else:
            layer = nn.Sequential(nn.Linear(nIn,nOut,bias=True),
                                  self.activation)   
        return layer
    
    
    def outputLayer(self, nIn, nOut):
        layer = nn.Sequential(nn.Linear(nIn, nOut,bias=True), self.activation_output)
        return layer
    
    def forward(self, S):
        h = self.i_h(S)
        for l in range(len(self.h_h)):
            h = self.h_h[l](h)
        output = self.h_o(h)
        return output
    

class Net_timegrid(nn.Module):
    """One feedforward network per timestep!

    """
    def __init__(self, dim, nOut, n_layers, vNetWidth, n_maturities, activation="relu", activation_output="id"):
        super().__init__()
        self.dim = dim
        self.nOut = nOut

        self.net_t = nn.ModuleList([Net_FFN(dim, nOut, n_layers, vNetWidth, activation=activation, activation_output=activation_output) for idx in range(n_maturities)])
        
    def forward_idx(self, idnet, x):
        y = self.net_t[idnet](x)
        return y

    def freeze(self, *args):
        if not args:
            for p in self.net_t.parameters():
                p.requires_grad=False
        else:
            self.unfreeze()
            for idx in args:
                for p in self.net_t[idx].parameters():
                    p.requires_grad_(False)

    def unfreeze(self, *args):
        if not args:
            for p in self.net_t.parameters():
                p.requires_grad=True
        else:
            # we just unfreeze the parameters between [last_T,T]
            self.freeze()
            for idx in args:
                for p in self.net_t[idx].parameters():
                    p.requires_grad=True

