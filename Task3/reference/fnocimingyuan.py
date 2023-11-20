import torch 
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class SpectralConv1d(nn.Module):
    """
        Parameters:
        -----------
            in_channels:    int
            out_channels:   int
            num_spectral:   int
        
        Usage:
        ------
        >>> x = torch.rand(10, 100, 3)       # [batch_size, in_channels, window_size]
        >>> model = SpectralConv1d(3, 2, 16) # in-channels = 3, out_channels = 2, num_spectral = 16
        >>> print(model(x).shape)            # [10, 2, 100] [batch_size, out_channels, window_size]
    
    """
    def __init__(self, 
                 in_channels,
                 out_channels,
                 num_spectral,
                 ):
        super().__init__()
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.num_spectral   = num_spectral
        self.weight         = nn.Parameter(torch.zeros(in_channels, out_channels, num_spectral))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, a=0.0, b=1/(self.in_channels * self.out_channels))
    
    def __str__(self):
        return f"SpectralConv1d({self.in_channels}, {self.out_channels}, num_spectral={self.num_spectral})"

    def __repr__(self):
        return str(self)

    def forward(self, x):
        """
            Parameters:
            -----------
                x:  torch.FloatTensor [batch_size, in_channels, window_size]
            Returns:
            --------
                torch.FloatTensor [batch_size, out_channels, window_size]
        """
        B, W, Co, S = x.shape[0], x.shape[-1], self.out_channels, self.num_spectral
        assert S <= W // 2 + 1
        spectral                  = torch.fft.rfft(x)  
        # spectral [batch_size, in_channels, total_specral]
        spectral_buffer           = torch.zeros([B, Co, W // 2 + 1], device=spectral.device, dtype=spectral.dtype)
        spectral_buffer[:, :, :S] = (spectral[:,:,None, :S] * self.weight[None,:,:,:]).sum(1)
        # spectral [batch_size, out_channels, num_spectral]
        x                         = torch.fft.irfft(spectral_buffer)
        return x
        
class NarrowConv1d(nn.Conv1d):
    def __init__(self,in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, 1, *args, **kwargs)


class Squeeze(nn.Module):
    def forward(self, x):
        return torch.squeeze(x)

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args
    def forward(self, x):
        return torch.permute(x, self.args)
    
class Activation(nn.Module):
    def __init__(self, activation):
        super().__init__()
        if activation.lower() in ['tanh', 'sigmoid']:
            self.activation_fn = getattr(torch, activation.lower())
        else:
            self.activation_fn = getattr(F, activation.lower())
    def forward(self, x):
        return self.activation_fn(x)

class FNO(nn.Module):
    """
        Parameters:
        -----------
            input_size:     int
            output_size:    int
            num_layers:     int
            hidden_size:    int
            activation:     str
            output_reduction:   bool, default False
               if output_reduction is True, the output shape well be [batch_size, output_size]
               otherwise , the output shape well be [batch_size, window_size, output_size]

        Usage:
        ------
        >>> model = FNO(3, 2, 3, 32, 'tanh', output_reduction=True) # input_size = 3, output_size = 2, num_layers = 3, hidden_size = 32, activation = 'tanh'
        >>> x = torch.rand(10, 100, 3)       # [batch_size, window_size, input_size]
        >>> print(model(x).shape)            # [10, 3] [batch_size, output_size]

        >>> model = FNO(3, 2, 3, 32, 'tanh', output_reduction=False) # input_size = 3, output_size = 2, num_layers = 3, hidden_size = 32, activation = 'tanh'
        >>> x = torch.rand(10, 100, 3)       # [batch_size, window_size, input_size]
        >>> print(model(x).shape)            # [10, 100, 3] [batch_size, window_size, output_size]
    """
    def __init__(self,
                input_size:int, 
                output_size:int, 
                num_layers:int, 
                hidden_size:int=32,
                activation:str='tanh',
                num_spectral:int=None,
                window_size:int=None,
                output_reduction:bool=True,
                ):
        super(FNO, self).__init__()
        assert num_layers >= 1
        self.activation = activation
        self.input_size = input_size 
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layeres = num_layers

        if num_spectral is None:
            num_spectral = window_size // 2 + 1

        self.num_spectral = num_spectral
        self.window_size  = window_size
        self.output_reduction = output_reduction

        self.input_transform  = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            Permute(0, 2, 1), # [batch_size, window_size, input_size] -> [batch_size, input_size, window_size]
        )

        if output_reduction:
            self.output_transform = nn.Sequential(
                nn.Linear(window_size, 1),
                Activation(activation),
                Squeeze(),        # [batch_size, hidden_size, 1] -> [batch_size, hidden_size]
                nn.Linear(hidden_size, output_size)
            )   
        else:
            self.output_transform = nn.Sequential(
                Permute(0, 2, 1), # [batch_size, hidden_size, window_size] -> [batch_size, window_size, hidden_size]
                nn.Linear(hidden_size, output_size),
            )   
    
        self.spectral_convs = nn.ModuleList([])
        self.spatial_convs  = nn.ModuleList([])
        for _ in range(num_layers):
            self.spectral_convs.append(SpectralConv1d(hidden_size, hidden_size, num_spectral))
            self.spatial_convs.append(NarrowConv1d(hidden_size, hidden_size))
            
        self.activation_fn = Activation(activation)

    def __len__(self):
        return len(self.spectral_convs)

    def __str__(self):
        return f"FNO_{self.input_size}_{self.output_size}_{self.num_layeres}_{self.hidden_size}_{self.activation}_{self.num_spectral}_{self.window_size}{'_reduce' if self.output_reduction else ''}"

    def forward(self, x):
        """
            Parameters:
            -----------
                x:      torch.FloatTensor[batch_size, window_size, input_size]
            Returns:
            --------
            if output_reduction is True:
                torch.FloatTensor[batch_size, output_size]
            else:
                torch.FloatTensor[batch_size, window_size, output_size]
        """
        x = self.input_transform(x)
        # x [batch_size, input_size, window_size]
        for i in range(len(self)):
            if i > 0:
                x = self.activation_fn(x)
            x = self.spatial_convs[i](x) + self.spectral_convs[i](x)
            
        # x [batch_size, hidden_size, window_size]
        x = self.output_transform(x)
        return x