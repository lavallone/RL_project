import numpy as np
import random
from copy import copy
from copy import deepcopy
import time
import torch
import torch.nn as nn
from collections import namedtuple, deque
import wandb

from src.utils.networks import Q_network

def from_tuple_to_tensor(tuple_of_tensors):
    ris = torch.zeros((len(tuple_of_tensors), tuple_of_tensors[0].shape[1]))
    for i, x in enumerate(tuple_of_tensors):
        ris[i] = x
    return ris

# Experience replay buffer
class ExperienceReplayBuffer:
    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.Buffer = namedtuple('Buffer', field_names=['state', 'action', 'reward', 'done', 'next_state'])
        self.replay_memory = deque(maxlen=memory_size)
    
    def sample_batch(self, batch_size=32):
        samples = np.random.choice(len(self.replay_memory), batch_size, replace=False)
        # use asterisk operator to unpack deque
        batch = zip(*[self.replay_memory[i] for i in samples])
        return batch
    
    def append(self, s_0, a, r, d, s_1):
        self.replay_memory.append(self.Buffer(s_0, a, r, d, s_1))
    
    def burn_in_capacity(self):
        return len(self.replay_memory) / self.burn_in
    
    def capacity(self):
        return len(self.replay_memory) / self.memory_size

# evaluation loop called during training
def evaluate_during_training(agent, eval_episodes):
    with torch.no_grad():
        agent.network.eval()
        total_reward_list, steps_list = [], []
        for _ in range(eval_episodes):    
            state = agent.env.reset()
            total_reward, steps = 0, 0
            done = False
            while not done:
                action = agent.epsilon_greedy_action(agent.preproc_state(state), epsilon=0.0) # greedy
                state, reward, done, _ = agent.env.step(action)
                total_reward += reward
                steps += 1.0
            if steps > 2000: done = True
            total_reward_list.append(total_reward)
            steps_list.append(steps)
    return torch.tensor(total_reward_list).mean(), int(torch.tensor(steps_list).mean())

class DDQNAgent():
    def __init__(self, env, output_dir, type_ = "ddqn", train_steps = 100000, log_every = 5, eval_episodes = 5, gamma = 0.99, min_eps = 0.01, network_update_frequency = 10, network_sync_frequency = 200, \
                 batch_size = 256, lr = 1e-4, adam_eps = 1e-8, n_concat_states = 4):
        super(DDQNAgent, self).__init__()
        
        self.env = env
        self.output_dir = output_dir
        self.type = type_
        self.network_update_frequency = network_update_frequency,
        self.network_sync_frequency = network_sync_frequency
        self.train_steps = train_steps
        self.log_every = log_every
        self.eval_episodes = eval_episodes
        self.gamma = gamma
        self.min_eps = min_eps
        
        self.network_update_frequency = network_update_frequency
        self.network_sync_frequency = network_sync_frequency
        self.batch_size = batch_size
        self.lr = lr
        self.adam_eps = adam_eps
        
        self.n_concat_states = n_concat_states
        self.states = deque(maxlen = self.n_concat_states)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        input_dim = self.n_concat_states * self.env.observation_space.nvec.shape[0]
        output_dim = self.env.action_space.n
        self.network = Q_network(input_dim, output_dim).to(self.device)
        self.target_network = deepcopy(self.network)
        self.buffer = ExperienceReplayBuffer()
        
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr, eps=self.adam_eps)
        self.loss_function = nn.MSELoss(self.device)
        
        
    # policy strategy
    def epsilon_greedy_action(self, state, epsilon):
        action = torch.max(self.network(state), -1)[1].item() if random.random() >= epsilon else self.env.action_space.sample()
        return action
    
    # epsilon decay
    def decay_epsilon(self, current_tot_steps, n_steps):
        a = -(1.0 - self.min_eps) / float(n_steps)
        b = 1.0
        epsilon = max(self.min_eps, a * float(current_tot_steps) + b)
        return epsilon
    
    def preproc_state(self, state):
        scaled_state = torch.tensor(state)/torch.tensor(self.env.observation_space.nvec)
        self.states.append(scaled_state)
        # if the queue is not filled yet
        if len(self.states) < self.n_concat_states:
            while len(self.states) < self.n_concat_states:
                self.states.append(scaled_state)
        # we concatenate the frames and we generate a new representation of the state to be injected to the networks
        states = torch.cat([s for s in self.states], dim=-1).unsqueeze(0).to(self.device).to(torch.float32)
        return states
        
    def compute_loss(self, batch):
        # extract info from batch (reason on batch!)
        states, actions, rewards, dones, next_states = list(batch)
        # transform in torch tensors
        rewards = torch.FloatTensor(rewards).reshape(-1, 1).to(self.device)
        actions = torch.LongTensor(np.array(actions)).reshape(-1, 1).to(self.device)
        dones = torch.IntTensor(dones).reshape(-1, 1).to(self.device)
        states = from_tuple_to_tensor(states).to(self.device).to(torch.float32)
        next_states = from_tuple_to_tensor(next_states).to(self.device).to(torch.float32)

        qvals = torch.gather(self.network.get_qvals(states), 1, actions) # I didn't know this function, but it's properly suitable in this case!
        # DQN update --> r + gamma * (max_a {Q_target(s',a)})
        if self.type == "dqn":
            actions_evaluation = torch.max(self.target_network.get_qvals(next_states), -1)[0].reshape(-1, 1) # this reshape is needed because the above tensors have this kind os size!
        # DDQN update --> r + gamma * (Q_target(s', argmax_a' {Q(s',a')})
        elif self.type == "ddqn":
            actions_selection = torch.max(self.network.get_qvals(next_states), -1)[1].reshape(-1, 1) # argmax
            actions_evaluation = torch.gather(self.target_network.get_qvals(next_states), 1, actions_selection)
        
        y_qvals = rewards + self.gamma * (1-dones) * actions_evaluation
        loss = self.loss_function(y_qvals, qvals)
        return loss

    def train(self, initial_epsilon=1.0, is_wandb = "no_wandb"):

        print("TRAINING STARTED...")
        
        ## init epsilon --> we're going to decay it through the episodes
        epsilon = initial_epsilon
        
        state = self.env.reset()
        # populate replay buffer
        while self.buffer.burn_in_capacity() < 1:
            action = self.epsilon_greedy_action(self.preproc_state(state), epsilon=1.0)
            # simulate action
            next_state, reward, done, _= self.env.step(action)
            # put experience in the buffer
            self.buffer.append(self.preproc_state(state), action, reward, done, self.preproc_state(next_state))
            state = copy(next_state)
            if done: state = self.env.reset()

        try:
            tot_steps = 0
            episode = 1
            losses = []
            while tot_steps <= self.train_steps:
                self.network.train()
                total_reward, steps = 0, 0
                episode_losses = []
                self.states = deque(maxlen = self.n_concat_states)
                
                state = self.env.reset()
                action = self.epsilon_greedy_action(self.preproc_state(state), epsilon)
                done = False
                while not done:
                    # simulate the action
                    next_state, reward, done, _ = self.env.step(action)
                    self.buffer.append(self.preproc_state(state), action, reward, done, self.preproc_state(next_state))
                    total_reward += reward
                    tot_steps += 1
                    steps += 1
                    next_action = self.epsilon_greedy_action(self.preproc_state(next_state), epsilon)
                    
                    # update network
                    if tot_steps % self.network_update_frequency == 0:
                        self.optimizer.zero_grad()
                        batch = self.buffer.sample_batch(batch_size=self.batch_size)
                        loss = self.compute_loss(batch)
                        losses.append(loss.item())
                        episode_losses.append(loss.item())
                        loss.backward()
                        self.optimizer.step()
                    # sync networks
                    if tot_steps % self.network_sync_frequency == 0:
                        self.target_network.load_state_dict(self.network.state_dict())
                    
                    if steps > 2000: break
                    # update current state
                    state = next_state
                    action = next_action
                    
                episode += 1
                # we test the model each 'log_every' episodes for 'eval_episodes' times
                if episode % self.log_every == 0:
                    self.save_model()
                    total_reward, steps = evaluate_during_training(self, self.eval_episodes)
                    template = '\n({0}/{1}) Episode {2}: total reward: {3:.3f}, steps: {4}, epsilon: {5:.6f}, mean_episode_loss: {6:.3f}\n'
                    variables = [tot_steps, self.train_steps, episode + 1, total_reward, steps, 0.0, (torch.tensor(episode_losses).mean())]
                    print(template.format(*variables))
                else:
                    template = '({0}/{1}) Episode {2}: total reward: {3:.3f}, steps: {4}, epsilon: {5:.6f}, mean_episode_loss: {6:.3f}'
                    variables = [tot_steps, self.train_steps, episode + 1, total_reward, steps, epsilon, (torch.tensor(episode_losses).mean())]
                    print(template.format(*variables))
                if is_wandb == "wandb": 
                    wandb.log({"total_reward" : total_reward})
                    wandb.log({"total_steps" : steps})
                # at the end of each training episodes we decay epsilon
                #epsilon = 0.99 * epsilon
                epsilon = self.decay_epsilon(tot_steps, self.train_steps/2)
            print("TRAINING FINISHED!")
        except KeyboardInterrupt:
            pass
    
    def test(self, env, n_episodes=5, visualize=True):
        print('Testing for {} episodes ...'.format(n_episodes))
        with torch.no_grad():
            self.load_model()
            self.network.eval()
            for episode in range(n_episodes):
                total_reward, steps = 0, 0
                state = env.reset()
                if visualize:
                    time.sleep(0.001)
                    env.render(mode='human')
                done = False

                while not done:
                    action = self.epsilon_greedy_action(self.preproc_state(state), epsilon=0.0) # greedy
                    next_state, reward, done, _ = env.step(action)
                    total_reward += reward
                    steps += 1
                    if visualize:
                        time.sleep(0.001)
                        env.render(mode='human')
                    state = next_state

                template = 'Episode {0}: reward: {1:.3f}, steps: {2}'
                variables = [episode + 1, total_reward, steps,]
                print(template.format(*variables))
            
    def save_model(self):
        torch.save(self.network, str(self.output_dir)+"/Q_net.pt")

    def load_model(self):
        self.network = torch.load(str(self.output_dir)+"/Q_net.pt")