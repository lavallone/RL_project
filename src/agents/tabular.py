import numpy as np
import random
from copy import copy
import time

# efficient implementation of Q-table
class mydefaultdict(dict):
    def __init__(self, x):
        super().__init__()
        self._default = x
    def __getitem__(self, key):
        if key in self:
            return dict.__getitem__(self, key)
        else:
            self[key] = copy(self._default)
            return self[key]

# Eligibility traces
class Eligibility(object):
    def __init__(self, lambda_, gamma):
        super().__init__()
        self.lambda_ = lambda_
        self.gamma = gamma
        self.traces = mydefaultdict(0.0)
    def get(self, state, action):
        return self.traces[(state, action)]
    def to_zero(self, state, action):
        self.traces.pop((state, action))
    def to_one(self, state, action):
        self.traces[(state, action)] = 1
    def update(self, state, action, *args, **kwargs):
        self.traces[(state, action)] = self.gamma * self.lambda_ * self.traces[(state, action)]
        if self.traces[(state, action)] < 1e-4:
            self.traces.pop(state, action)
    def reset(self):
        self.traces = {}


class TabularAgent():
    def __init__(self, env, td_type = "sarsa", alpha=0.2, gamma=0.99, lambda_= 0.99, min_eps = 0.01, train_steps = 10000):
        super(TabularAgent, self).__init__()
        
        self.env = env
        self.td_type = td_type # sarsa or q-learning
        self.train_steps = train_steps
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.min_eps = min_eps
        
        self.Q = mydefaultdict(np.zeros((env.action_space.n,)))
        self.E = Eligibility(self.lambda_, self.gamma)
        
    # policy strategy
    def epsilon_greedy_action(self, state, epsilon):
        action = (np.argmax(self.Q[state])) if random.random() >= epsilon else self.env.action_space.sample()
        return action
    
    # epsilon decay
    def decay_epsilon(self, current_tot_steps, n_steps):
        a = -(1.0 - self.min_eps) / float(n_steps)
        b = 1.0
        epsilon = max(self.min_eps, a * float(current_tot_steps) + b)
        return epsilon

    def train(self, initial_epsilon=1.0):

        print("TRAINING STARTED...")
        
        ## init epsilon --> we're going to decay it through the episodes
        epsilon = initial_epsilon

        try:
            tot_steps = 0
            episode = 0
            while tot_steps <= self.train_steps:
                total_reward, steps = 0, 0
                state = self.env.reset()
                action = self.epsilon_greedy_action(state, epsilon)
                # we need to initialize Eligibility traces each episode!
                self.E.reset()
                done = False
                while not done:
                    # simulate the action
                    next_state, reward, done, _ = self.env.step(action)
                    total_reward += reward
                    steps += 1
                    next_action = self.epsilon_greedy_action(next_state, epsilon)
                    
                    if self.td_type == "sarsa":
                        td_error = reward + self.gamma * (1-done) * self.Q[next_state][next_action] - self.Q[state][action]
                        self.E.to_one(state, action)
                        for (s, a) in set(self.E.traces.keys()):
                            self.Q[s][a] += self.alpha * td_error * self.E.get(s, a)
                            self.E.update(s, a)
                        
                    elif self.td_type == "q":
                        Q_a = np.max(self.Q[next_state])
                        actions_star = np.argwhere(self.Q[next_state] == Q_a).flatten().tolist()
                        td_error = reward + self.gamma * Q_a - self.Q[state][action]
                        self.E.to_one(state, action)
                        for (s, a) in set(self.E.traces.keys()):
                            self.Q[s][a] += self.alpha * td_error * self.E.get(s, a)
                            if next_action in actions_star:
                                self.E.update(s, a)
                            else:
                                self.E.to_zero(s, a)
                    
                    # update current state
                    state = next_state
                    action = next_action
                tot_steps += steps
                episode += 1
                template = '({0}/{1}) Episode {2}: total reward: {3:.3f}, steps: {4}, epsilon: {5:.6f}'
                variables = [tot_steps, self.train_steps, episode + 1, total_reward, steps, epsilon,]
                print(template.format(*variables))
                # at the end of each training episodes we decay epsilon
                #epsilon = 0.99 * epsilon
                epsilon = self.decay_epsilon(tot_steps, self.train_steps/10)
            print("TRAINING FINISHED!")
        except KeyboardInterrupt:
            pass
    
    def test(self, env, n_episodes=5, visualize=True):
        print('Testing for {} episodes ...'.format(n_episodes))
        
        for episode in range(n_episodes):
            total_reward, steps = 0, 0
            state = env.reset()
            if visualize:
                time.sleep(0.001)
                env.render(mode='human')
            done = False

            while not done:
                action = self.epsilon_greedy_action(state, epsilon=0.0) # greedy
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
    
    