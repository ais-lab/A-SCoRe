import torch 
import torch.nn as nn
from .sg_attn import AttentionalGNN, MLP


class SuperPoint(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1) # useless.
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0) # useless.

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, 256, kernel_size=1, stride=1, padding=0)

        self.convDc = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.convDd = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)

        #self.load_state_dict(torch.load('/media/slam/Data1/Hoang_workspace/erinyes_clone/erinyes/src/common/weights/superpoint_v1.pth'))
        #print('loaded pre-trained SuperPoint.')

    def forward(self, data):
        # encoder
        x = self.relu(self.conv1a(data))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        cDa = self.relu(self.convDa(x))
        cDb = self.relu(self.convDb(cDa))
        cDc = self.relu(self.convDc(cDb))
        descriptors = self.convDd(cDc)
        #descriptors = self.convDb(cDa)

        return descriptors


class SPSG(nn.Module):
    default_conf = {
            "d_model": 512,
            "nhead": 4,
            "nlayer": 5,
            }
    def __init__(self, config):
        super().__init__()
        self.config     = {**self.default_conf, **config}
        self.encoder    = SuperPoint()

        self.feat_transformer = AttentionalGNN(feature_dim=self.config['d_model'],
                                               no_layers=self.config['nlayer'])

        self.mapping = MLP([self.config['d_model'], 512, 1024, 1024, 3])

    def forward(self, data):
        x = self.encoder(data)
        B, N, H, W = x.shape
        x = x.reshape(B, N, H*W)
        x = self.feat_transformer(x)
        x = self.mapping(x)
        x = x.reshape(B, 3, H, W)
        return x


if __name__ == "__main__":
    config = {
            "d_model": 256,
            "nhead": 8,
            "nlayer": 4,
            "layer_names": ['self']*4
            }
    model = SPSG(config)
    input_ = torch.randn(1, 1, 480, 640)
    out = model(input_)

    total_param = sum(p.numel() for p in model.parameters())
    print("Number of param", total_param)




