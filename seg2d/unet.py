from torch import nn
import torch



class Unet(nn.Module):
    """
    My UNet implementation
    """
    
    def __init__(self, depth=5, start_channels=64):
        """
        Creates the network.
        
        Args:
            depth: number of dual convolutional layers
            start_channels: number of output channels in first layer
        """
        
        super(Unet, self).__init__()
        self.depth = depth
        self.start_channels = start_channels
        
        # create downsample path
        self.down_path = nn.ModuleList()
        for n in range(self.depth):
            if n == 0:
                ch_in = 1
            else:
                ch_in  = self.start_channels * (2**(n-1))
            ch_out = self.start_channels * (2**n)
            
            self.down_path.append(self.dual_conv(ch_in, ch_out))
        
        # create maxpool operation
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            
        # create upsample path
        self.up_path_trans = nn.ModuleList()
        self.up_path_conv = nn.ModuleList()
        for n in range(self.depth)[::-1]:
            if n == 0:
                ch_out = 1
            else:
                ch_out  = self.start_channels * (2**(n-1))
            ch_in = self.start_channels * (2**n)
            
            trans = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)
            conv = self.dual_conv(ch_in, ch_out)                 
            self.up_path_trans.append(trans)
            self.up_path_conv.append(conv)   
            
        # create output layer
        self.out = nn.Conv2d(64, 1, kernel_size=1)
            
       
    @staticmethod
    def dual_conv(in_channel, out_channel):
        """
        Returns a dual convolutional layer with ReLU activations in between. 
        """
        conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return conv
    
    
    def forward(self, x):
        "Forward pass through the network"
        
        # pass through downsample path
        self.feature_maps = []
        for n in range(self.depth):
            down_conv = self.down_path[n]
            x = down_conv(x)
            if n < self.depth-1:
                self.feature_maps.append(x)
                x = self.maxpool(x)
            
        print(f'Generated feature maps: {len(self.feature_maps)}')
              
        # pass through upsample path
        for n in range(self.depth-1):
            trans = self.up_path_trans[n]
            conv = self.up_path_conv[n]
            
            x = trans(x)
            y = self.feature_maps[-(n+1)]
            x = conv(torch.cat([x,y], 1))
            
        # pass through output layer
        x = self.out(x)
        
        return x
