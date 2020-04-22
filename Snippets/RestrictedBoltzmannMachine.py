# Restricted Boltzmann Machine
"""import torch

class RBM():
    def __init__(self, nv, nh):
        self.w = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)

    def h(self, x):
        wx = torch.mm(x, self.w.t())
        acti = wx + self.a.expand_as(wx)
        hv = torch.sigmoid(acti)
        return hv, torch.bernoulli(hv)

    def v(self, y):
        wy = torch.mm(y, self.w)
        acti = wy + self.b.expand_as(wy)
        vh = torch.sigmoid(acti)
        return vh, torch.bernoulli(vh)

    def train(self, v0, vk, ph0, phk):
        self.w += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)


rbm = RBM(len(trainingSet[0]), 100)

epoch = 5
batchSize = 100

for nbEpoch in range(1, epoch + 1):
    loss = 0
    s = 0.0

    for i in range(0, batchSize, batchSize):
        vk = trainingSet[i:i + batchSize]
        v0 = trainingSet[i:i + batchSize]
        ph0, _ = rbm.h(v0)

        for k in range(10):
            _, hk = rbm.h(vk)
            _, vk = rbm.v(hk)
            vk[v0 < 0] = v0[v0 < 0]

        phk, _ = rbm.h(vk)
        rbm.train(v0, vk, ph0, phk)
        loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
        s += 1.0

    print("Epoch: " + str(nbEpoch) + " Loss: " + str(loss / s))

loss = 0
s = 0.0

for i in range(epoch):
    v = trainingSet[i:i + 1]
    vt = testingSet[i:i + 1]
    if len(vt[vt >= 0]) > 0:
        _, h = rbm.h(v)
        _, v = rbm.v(h)
        loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
        s += 1.0
print("Test Loss: " + str(loss / s))"""