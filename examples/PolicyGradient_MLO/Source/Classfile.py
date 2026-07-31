import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




class Policy(nn.Module):
        def __init__(self, s_size, a_size, h_size):
            super(Policy, self).__init__()
            self.fc1 = nn.Linear(s_size, h_size)
            self.fc2 = nn.Linear(h_size, a_size)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.softmax(x, dim=1)

        def act(self, state, deterministic=False):

            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            probs = self.forward(state).cpu()
            if deterministic:
                action = torch.argmax(probs, dim=1)
                log_prob = torch.log(probs.squeeze(0)[action])
            else:
                m = Categorical(probs)
                action = m.sample()
                log_prob = m.log_prob(action)

            return action.item(), log_prob
        