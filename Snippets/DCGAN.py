from torch.nn import (
    BatchNorm2d,
    BCELoss,
    Conv2d,
    ConvTranspose2d,
    LeakyReLU,
    Module,
    ReLU,
    Sequential,
    Sigmoid,
    Tanh,
)

class Gen(Module):
    def __init__(self):
        super(Gen, self).__init__()
        self.main = Sequential(
            ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            BatchNorm2d(512),
            ReLU(True),
            ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            BatchNorm2d(256),
            ReLU(True),
            ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            BatchNorm2d(128),
            ReLU(True),
            ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            BatchNorm2d(64),
            ReLU(True),
            ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            Tanh(),
        )

    def forward(self, data):
        output = self.main(data)
        return output


class Dis(Module):
    def __init__(self):
        super(Dis, self).__init__()
        self.main = Sequential(
            Conv2d(3, 64, 4, 2, 1, bias=False),
            LeakyReLU(0.2, inplace=True),
            Conv2d(64, 128, 4, 2, 1, bias=False),
            BatchNorm2d(128),
            LeakyReLU(0.2, inplace=True),
            Conv2d(128, 256, 4, 2, 1, bias=False),
            BatchNorm2d(256),
            LeakyReLU(0.2, inplace=True),
            Conv2d(256, 512, 4, 2, 1, bias=False),
            BatchNorm2d(512),
            LeakyReLU(0.2, inplace=True),
            Conv2d(512, 1, 4, 1, 0, bias=False),
            Sigmoid(),
        )

    def forward(self, data):
        output = self.main(data)
        return output.view(-1)

