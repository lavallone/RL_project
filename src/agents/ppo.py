from torch.utils.data import Dataset, DataLoader
import time
import torch
import torch.nn as nn
from collections import deque
from torch.distributions import Categorical

from src.utils.networks import ActorCritic_network

# Generalized Advantage Estimation
def GAE(rewards, values, dones, gamma, lambda_, last_value, next_done):
    advantages = torch.zeros(len(rewards))
    prev_gae_advantage = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - next_done
            next_value = last_value
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        advantages[t] = prev_gae_advantage = delta + gamma * lambda_ * next_non_terminal * prev_gae_advantage
    returns = advantages + torch.tensor(values)
    return advantages, returns

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
                actions = agent.network.forward(agent.preproc_state(state))
                action = actions.argmax(-1).item()
                state, reward, done, _ = agent.env.step(action)
                total_reward += reward
                steps += 1.0
            total_reward_list.append(total_reward)
            steps_list.append(steps)
    return torch.tensor(total_reward_list).mean(), int(torch.tensor(steps_list).mean())

# rollout data structure
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.logprobs = []
        self.values = []
        self.dones = []

# dataset used during the learning process
class RolloutDataset(Dataset):
    def __init__(self, rollout_buffer, advantages, returns):
        self.data = self.make_data(rollout_buffer, advantages, returns)
    
    def make_data(self, rollout_buffer, advantages, returns):
        ris = []
        for idx in range(len(rollout_buffer.rewards)):
            d = {}
            d["state"] = rollout_buffer.states[idx]
            d["action"] = rollout_buffer.actions[idx]
            d["reward"] = rollout_buffer.rewards[idx]
            d["logprob"] = rollout_buffer.logprobs[idx]
            d["value"] = rollout_buffer.values[idx]
            d["done"] = rollout_buffer.dones[idx]
            d["advantage"] = advantages[idx].item()
            d["return"] = returns[idx].item()
            ris.append(d)
        return ris
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PPOAgent():
    def __init__(self, env, output_dir, train_episodes = 200, n_epochs = 50, n_rollout_steps = 5000, log_every=5, eval_episodes=10, gamma = 0.99, lambda_ = 0.95,  
                 clip_epsilon = 0.2, c_1 = 0.5, c_2 = 0.01, batch_size = 256, lr = 1e-4, adam_eps = 1e-8, n_concat_states = 4):
        super(PPOAgent, self).__init__()
        
        self.env = env
        self.output_dir = output_dir
        self.train_episodes = train_episodes
        self.n_epochs = n_epochs
        self.n_rollout_steps = n_rollout_steps
        self.log_every = log_every
        self.eval_episodes = eval_episodes
        self.gamma = gamma
        self.lambda_ = lambda_
        self.clip_epsilon = clip_epsilon
        self.c_1 = c_1
        self.c_2 = c_2
        
        self.batch_size = batch_size
        self.lr = lr
        self.adam_eps = adam_eps
        
        self.n_concat_states = n_concat_states
        self.states = deque(maxlen = self.n_concat_states)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        input_dim = self.n_concat_states * self.env.observation_space.nvec.shape[0]
        output_dim = self.env.action_space.n
        self.network = ActorCritic_network(input_dim, output_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr, eps=self.adam_eps)
        self.mse_loss = nn.MSELoss(self.device)
    
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
    
    # single rollout step to produce data needed for the subsequent learning step
    def rollout_step(self, state):
        state = self.preproc_state(state)
        actions = self.network.forward(state) # action probabilities
        value = self.network.forward(state, "critic")

        distribution = Categorical(actions)
        action_sampled = distribution.sample()
        action_logprob = distribution.log_prob(action_sampled)

        return state.detach().cpu().squeeze(0), action_sampled.detach().cpu().item(), action_logprob.detach().cpu().item(), value.detach().cpu().item()

    def train(self):
        print("TRAINING STARTED...")
        try:
            for episode in range(1, self.train_episodes+1):
                # lr linear annealing
                frac = 1.0 - (episode - 1.0) / self.train_episodes
                self.optimizer.param_groups[0]["lr"] = frac * self.lr
                self.states = deque(maxlen = self.n_concat_states)
                rollout_buffer = RolloutBuffer()
                
                # ROLLOUT phase
                state = self.env.reset()
                done = False
                for _ in range(self.n_rollout_steps):
                    state, action_sampled, action_logprob, value = self.rollout_step(state)
                    rollout_buffer.states.append(state)
                    rollout_buffer.actions.append(action_sampled)
                    rollout_buffer.logprobs.append(action_logprob)
                    rollout_buffer.values.append(value)
                    rollout_buffer.dones.append(done)
                    state, reward, done, _ = self.env.step(action_sampled)
                    rollout_buffer.rewards.append(reward)
                rollout_buffer_size = torch.count_nonzero(torch.tensor(rollout_buffer.dones) == False).item()
                # 'advantages' estimation and 'returns' computation ('returns' are the the targets for the value function)
                last_value = self.network.forward(self.preproc_state(state), "critic")
                next_done = done
                advantages, returns = GAE(rollout_buffer.rewards, rollout_buffer.values, rollout_buffer.dones, self.gamma, self.lambda_, last_value, next_done)
                
                # LEARNING PHASE
                self.network.train()
                dataset = RolloutDataset(rollout_buffer, advantages, returns)
                dataloader = DataLoader(dataset, self.batch_size, shuffle=True)
                epochs_losses = []
                for _ in range(self.n_epochs):
                    losses = []
                    for batch in dataloader:
                        # NEW values, NEW logprobs and ENTROPY of the updated 'policy'
                        new_values = self.network.forward(batch["state"].to(self.device), "critic")
                        action_probs = self.network.forward(batch["state"].to(self.device), "actor")
                        distribution = Categorical(action_probs)
                        new_log_probs = distribution.log_prob(batch["action"].to(self.device))
                        entropy = distribution.entropy()
                        # RATIO between old and new 'policy'
                        log_ratio = new_log_probs - batch["logprob"].to(self.device)
                        ratio = log_ratio.exp()
                        # NORMALIZE advantages
                        batch["advantage"] = batch["advantage"].to(self.device)
                        advantages = (batch["advantage"] - batch["advantage"].mean()) / (batch["advantage"].std() + 1e-8)
                        
                        # POLICY GRADIENT LOSS
                        pg_loss_1 = advantages * ratio
                        pg_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                        pg_loss = torch.min(pg_loss_1, pg_loss_2).mean()
                        # VALUE FUNCTION LOSS
                        v_loss = self.mse_loss(new_values.to(torch.float64).view(-1), batch["return"].to(self.device)) # MSE loss
                        # ENTROPY LOSS
                        entropy_loss = entropy.mean()
                        # PPO LOSS
                        loss = -pg_loss +(self.c_1 * v_loss) -(self.c_2 * entropy_loss)
                        losses.append(loss)
                        
                        self.optimizer.zero_grad()
                        loss.backward()
                        #nn.utils.clip_grad_norm_(self.parameters(), 0.5) # gradient clipping
                        self.optimizer.step()
                    epochs_losses.append(torch.tensor(losses).mean().item())
                        
                # we test the model each 'log_every' episodes for 'eval_episodes' times
                if episode % self.log_every == 0:
                    self.save_model()
                    total_reward, steps = evaluate_during_training(self, self.eval_episodes)
                    template = '\nEpisode {0}: rollout_buffer_size: {1}, total reward: {2:.3f}, steps: {3}, mean_episode_PPO_loss: {4:.3f}\n'
                    variables = [episode, rollout_buffer_size, total_reward, steps, (torch.tensor(epochs_losses).mean())]
                    print(template.format(*variables))
                else:
                    template = 'Episode {0}: rollout_buffer_size: {1}, mean_episode_PPO_loss: {2:.3f}'
                    variables = [episode, rollout_buffer_size, (torch.tensor(epochs_losses).mean())]
                    print(template.format(*variables))
            print("TRAINING FINISHED!")
        except KeyboardInterrupt:
            pass
    
    def test(self, env, n_episodes=5, visualize=True):
        print('Testing for {} episodes ...'.format(n_episodes))
        with torch.no_grad():
            self.network.eval()
            for episode in range(n_episodes):
                total_reward, steps = 0, 0
                state = env.reset()
                if visualize:
                    time.sleep(0.001)
                    env.render(mode='human')
                done = False
                while not done:
                    actions = self.network.forward(self.preproc_state(state))
                    action = actions.argmax(-1).item() # greedy
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
        torch.save(self.network, str(self.output_dir)+"/ActorCritic.pt")

    def load_model(self):
        self.network = torch.load(str(self.output_dir)+"/ActorCritic.pt")