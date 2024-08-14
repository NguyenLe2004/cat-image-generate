import torch
import torch.nn as nn

def model_preprocessing(generator, discriminator, device) :
    """
    Prepare the generator and discriminator models for training.
    """
    ngpu = torch.cuda.device_count()
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        generator = nn.DataParallel(generator, list(range(ngpu)))
        discriminator = nn.DataParallel(discriminator, list(range(ngpu)))
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    return generator, discriminator

def weights_init(m):
    """
    Apply custom weight initialization to the layers of the model.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    """
    The generator network for the GAN model.
    """
    def __init__(self, laten_dim=100, size_feature_map=32):
        super(Generator, self).__init__()
        self.main = nn.ModuleList([
            self.StartBlock(laten_dim, size_feature_map * 32),
            self.UpsampleBlock(size_feature_map * 32, size_feature_map * 16),
            self.UpsampleBlock(size_feature_map * 16, size_feature_map * 8),
            self.PixelShuffle(size_feature_map * 8),
        ])
        self.output = nn.ModuleList([
            self.OutputBlock(size_feature_map * 16),
            self.OutputBlock(size_feature_map * 8),
            self.OutputBlock(size_feature_map * 2),            
        ])
    def StartBlock(self,in_channel, out_channel):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, 4, 1, 0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.SELU(inplace=True),
        )
    
    def PixelShuffle(self,in_channel,factor=2):
        return nn.Sequential(
            nn.PixelShuffle(factor),
            nn.BatchNorm2d(in_channel// (factor**2)),
            nn.SELU()
        )
        
    def OutputBlock(self, in_channel, out_channel=3):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Tanh()
        )
    def UpsampleBlock(self, in_channel, out_channel):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.SELU(inplace=True),
        )
    def forward(self, x, idx=5):
        idx = min(idx,len(self.main)-2)
        for layer in self.main[:idx+2]:
            x = layer(x)
        output = self.output[idx](x)
        return output
    
class Discriminator(nn.Module):
    """
    The discriminator network for the GAN model.
    """
    def __init__(self,size_feature_map = 64):
        super(Discriminator, self).__init__()
        self.main = nn.ModuleList([
            self.StartBlock(3, size_feature_map),
            self.DownSampleBlock(size_feature_map,size_feature_map*2),
            self.DownSampleBlock(size_feature_map*2,size_feature_map*4),
            self.DownSampleBlock(size_feature_map*4,size_feature_map*8),
        ])
        self.output = nn.ModuleList([
            self.OutputBlock(size_feature_map * 2),
            self.OutputBlock(size_feature_map * 4),
            self.OutputBlock(size_feature_map * 8),
        ])
    def StartBlock(self,in_channel,out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel,out_channel,4,2,0,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.SELU(inplace=True),
        )
    def DownSampleBlock(self,in_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel,out_channel,4,2,1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.SELU(inplace=True),
        )
    def OutputBlock(self, in_channel, out_channel=1):
        return nn.Sequential(
            nn.Conv2d(in_channel,out_channel,4,2,1,bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, idx):
        idx = min(idx,len(self.main)-2)
        for layer in self.main[:idx+2]:
            x = layer(x)
        output = self.output[idx](x)
        return output