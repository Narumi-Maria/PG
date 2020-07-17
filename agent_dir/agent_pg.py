from agent_dir.agent import Agent
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.misc import imsave, imresize

class actor(nn.module):
    def __init__self(self):
        super().__init__()
        self.nt = nn.Sequential(
            nn.Linear(6400, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self,s):
        p=self.nt(s)
        return p.view(-1)

class Agent_PG(Agent):
    def __init__(self, env, args):
        super(Agent_PG, self).__init__(env)
        self.env=env
        self.reward_history, self.reward_list, self.prob_list, self.label_list=[],[],[],[]
        self.net=actor()
        self.opt=optim.Adam(self.net.parameters(),lr=0.0001)
        self.opt.zero_grad()
        self.prev_state=None
        self.curr_state=env.reset()
        self.reward_sum=0
        self.episode_number=0
        self.batchsize=10
        self.gamma=0.99

    def init_game_setting(self):
        self.reward_sum=0
        self.reward_list, self.prob_list, self.label_list = [], [], []
        self.prev_state = None
        self.curr_state = self.env.reset()

    def get_action_label(self,p):
        p = p.detach().cpu().numpy()
        action = 2 if p[0] > np.random.uniform() else 3  # 2 is up / 3 is down
        label = torch.ones(1) if action == 2 else torch.zeros(1)  # Label is 1 if moving up
        return action, label

    def get_prepro_state(self,s):
        s = 0.2126 * s[:, :, 0] + 0.7152 * s[:, :, 1] + 0.0722 * s[:, :, 2]
        s = s.astype(np.uint8)
        s = imresize(s, (80, 80)).ravel()
        return torch.tensor(s, dtype=torch.float).unsqueeze(0)  # Return a tensor

    def get_po_loss(self,p, l, r):
        # 转list为tensor
        p = torch.cat(p, 0)
        l = torch.cat(l, 0)
        r = self.get_discounted_reward(np.array(r), self.gamma)
        eps = 1e-8
        return -torch.sum((l * torch.log(p + eps) + (1 - l) * torch.log(1 - p + eps)) * r)

    def get_discounted_reward(self,r, gamma):
        dsct_r = np.zeros_like(r)
        run_r = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0: running_add = 0
            run_r = run_r * gamma + r[t]
            dsct_r[t] = run_r
        dsct_r = (dsct_r - np.mean(dsct_r)) / np.std(dsct_r)
        return torch.tensor(dsct_r, dtype=torch.float)

    def train(self):
        while True:
            self.curr_state=self.get_prepro_state(self.curr_state)
            diff_state=self.curr_state-(self.prev_state if self.prev_state is not None else 0)
            self.prev_state=self.curr_state

            prob=self.net(diff_state)
            action,label=self.get_action_label(prob)
            self.label_list.append(label)
            self.prob_list.append(prob)

            self.curr_state,reward,done,info=self.env.step(action)

            self.reward_sum+=reward
            self.reward_list.append(reward)

            if done:
                self.episode_number+=1
                self.reward_history.append(self.reward_sum)
                np.save('reward,npy',np.array(self.reward_history))
                loss=self.get_po_loss(self.prob_list,self.label_list,self.reward_list)
                loss.backward()

                if self.episode_number % self.batchsize == 0:
                    print("Episode: {}, Reward: {}, Loss: {}".format(self.episode_number, self.reward_sum, loss.item()))
                    self.opt.step()
                    self.opt.zero_grad()

                if self.episode_number % 100 == 0:
                    torch.save(self.net.state_dict,'actor.ckpt')

                self.init_game_setting()