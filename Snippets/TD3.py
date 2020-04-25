# Twin-Delayed DDPG
"""import numpy as np
import torch

device = torch.device("cuda")


class Replay(object):
    def __init__(self, size=1e6):
        self.memory = []
        self.size = size
        self.pointer = 0

    def add(self, trans):
        if len(self.memory) == self.size:
            self.memory[int(self.pointer)] = trans
            self.pointer = (self.pointer + 1) * self.size

        else:
            self.memory.append(trans)

    def sample(self, batchSize):
        index = np.random.randint(0, len(self.memory), batchSize)
        batchState, batchNextState, batchActions, batchRewards, batchComplets = (
            [],
            [],
            [],
            [],
            [],
        )

        for i in index:
            state, nextState, actions, rewards, complets = self.memory[i]

            batchState.append(np.array(state, copy=False))
            batchNextState.append(np.array(nextState, copy=False))
            batchActions.append(np.array(actions, copy=False))
            batchRewards.append(np.array(rewards, copy=False))
            batchComplets.append(np.array(complets, copy=False))

        return (
            np.array(batchState),
            np.array(batchNextState),
            np.array(batchActions),
            np.array(batchRewards).reshape(-1, 1),
            np.array(batchComplets).reshape(-1, 1),
        )


class Actor(torch.nn.Module):
    def __init__(self, state, action, maxAction):
        super(Actor, self).__init__()
        self.layer1 = torch.nn.Linear(state, 400)
        self.layer2 = torch.nn.Linear(400, 300)
        self.layer3 = torch.nn.Linear(300, action)
        self.maxAction = maxAction

    def forward(self, x):
        return self.maxAction * torch.tanh(
            self.layer3(
                torch.nn.functional.relu(
                    self.layer2(torch.nn.functional.relu(self.layer1(x)))
                )
            )
        )


class Critic(torch.nn.Module):
    def __init__(self, state, action):
        super(Critic, self).__init__()
        self.layer1 = torch.nn.Linear(state + action, 400)
        self.layer2 = torch.nn.Linear(400, 300)
        self.layer3 = torch.nn.Linear(300, 1)

        self.layer4 = torch.nn.Linear(state + action, 400)
        self.layer5 = torch.nn.Linear(400, 300)
        self.layer6 = torch.nn.Linear(300, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        return (
            self.layer3(
                torch.nn.functional.relu(
                    self.layer2(torch.nn.functional.relu(self.layer1(xu)))
                )
            ),
            self.layer6(
                torch.nn.functional.relu(
                    self.layer5(torch.nn.functional.relu(self.layer4(xu)))
                )
            ),
        )

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        return self.layer3(
            torch.nn.functional.relu(
                self.layer2(torch.nn.functional.relu(self.layer1(xu)))
            )
        )


class TD3(object):
    def __init__(self, state, action, maxAction):
        self.actor = Actor(state, action, maxAction).to(device)
        self.actorTarget = Actor(state, action, maxAction).to(device)
        self.actorTarget.load_state_dict(self.actor.state_dict())
        self.actorOptimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state, action).to(device)
        self.criticTarget = Critic(state, action).to(device)
        self.criticTarget.load_state_dict(self.critic.state_dict())
        self.criticOptimizer = torch.optim.Adam(self.critic.parameters())

        self.maxAction = maxAction

    def selectAction(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(
        self,
        replay,
        iterations,
        batchSize=100,
        discount=0.99,
        tau=0.005,
        policyNoise=0.2,
        noiseClip=0.5,
        policyFreq=2,
    ):
        for i in range(iterations):
            batchState, batchNextState, batchActions, batchRewards, batchComplete = replay.sample(
                batchSize
            )

            state = torch.Tensor(batchState).to(device)
            nextState = torch.Tensor(batchNextState).to(device)
            action = torch.Tensor(batchActions).to(device)
            reward = torch.Tensor(batchRewards).to(device)
            complete = torch.Tensor(batchComplete).to(device)

            nextAction = self.actorTarget.forward(nextState)

            noise = (
                torch.Tensor(batchActions)
                .data.normal_(0, policyNoise)
                .to(device)
                .clamp(-noiseClip, noiseClip)
            )
            nextAction = (nextAction + noise).clamp(-self.maxAction, self.maxAction)

            targetQ1, targetQ2 = self.criticTarget.forward(nextState, nextAction)

            targetQ = (
                reward
                + ((1 - complete) * discount * torch.min(targetQ1, targetQ2)).detach()
            )

            currentQ1, currentQ2 = self.critic.forward(state, action)

            criticLoss = torch.nn.functional.mse_loss(
                currentQ1, targetQ
            ) + torch.nn.functional.mse_loss(currentQ2, targetQ)

            self.criticOptimizer.zero_grad()
            criticLoss.backward()
            self.criticOptimizer.step()

            if i % policyFreq == 0:
                actorLoss = -self.critic.Q1(state, self.actor(state)).mean()
                self.actorOptimizer.zero_grad()
                actorLoss.backward()
                self.actorOptimizer.step()

                for param, targetParam in zip(
                    self.actor.parameters(), self.actorTarget.parameters()
                ):
                    targetParam.data.copy_(
                        tau * param.data + (1 - tau) * targetParam.data
                    )

                for param, targetParam in zip(
                    self.critic.parameters(), self.criticTarget.parameters()
                ):
                    targetParam.data.copy_(
                        tau * param.data + (1 - tau) * targetParam.data
                    )

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%sActor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%sCritic.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load("%s/%sActor.pth" % (directory, filename)))
        self.critic.load_state_dict(
            torch.load("%s/%sCritic.pth" % (directory, filename))
        )"""