# Stcked Auto Encoder
"""import torch
import numpy as np

inputs = 30

class SAE(torch.nn.Module):
    def __init__(self):
        super(SAE, self).__init__()
        self.fc1 = torch.nn.Linear(inputs, 20)
        self.fc2 = torch.nn.Linear(20, 10)
        self.fc3 = torch.nn.Linear(10, 20)
        self.fc4 = torch.nn.Linear(20, inputs)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        return self.fc4(
            self.activation(
                self.fc3(
                    self.activation(self.fc2(self.activation(self.fc1(x)))))))


sae = SAE()

epoch = 500
criterion = torch.nn.MSELoss()
optimizer = torch.optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)

for nbEpoch in range(1, epoch + 1):
    loss = 0
    s = 0.0

    for user in range(epoch):
        input = torch.Variable(trainingSet[user]).unsqueeze(0)
        target = input.clone()

        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            trainLoss = criterion(output, target)
            corrector = inputs / float(torch.sum(target.data > 0) + 1e-10)
            trainLoss.backward()
            loss += np.sqrt(trainLoss.data * corrector)
            s += 1.0
            optimizer.step()

    print(f"Epoch: {nbEpoch} Loss: {loss/s}")

loss = 0
s = 0.0

for user in range(epochs):
    input = torch.Variable(trainingSet[user]).unsqueeze(0)
    target = torch.Variable(testingSet[user]).unsqueeze(0)

    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        testLoss = criterion(output, target)
        corrector = inputs / float(torch.sum(target.data > 0) + 1e-10)
        loss += np.sqrt(testLoss.data * corrector)
        s += 1.0

print(f"Test Loss: {loss/s}")"""