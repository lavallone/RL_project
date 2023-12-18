import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# orthogonal initialization
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
    
class Q_network(nn.Module):

    def __init__(self, n_inputs, n_outputs, bias=True):
        super(Q_network, self).__init__()

        # network receives as input a state and outputs a value for each action
        self.activation_function= nn.Tanh()
        self.layer1 = layer_init(nn.Linear(n_inputs, 128,  bias=bias))
        self.layer2 = layer_init(nn.Linear(128, 64, bias=bias))
        self.layer3 = layer_init(nn.Linear(64, 32, bias=bias))
        self.layer4 = layer_init(nn.Linear(32, n_outputs, bias=bias))

    def forward(self, x):
        x = self.activation_function( self.layer1(x) )
        x = self.activation_function( self.layer2(x) )
        x = self.activation_function( self.layer3(x) )
        y = self.layer4(x)
        return y
    
    def get_qvals(self, state):
        return self(state)
    
class ActorCritic_network(nn.Module):

    def __init__(self, n_inputs, n_outputs, bias=True):
        super(ActorCritic_network, self).__init__()

        self.backbone = nn.Sequential(layer_init(nn.Linear(n_inputs, 32,  bias=bias)),
                                      nn.Tanh(),
                                      layer_init(nn.Linear(32, 64, bias=bias)),
                                      nn.Tanh(),
                                      layer_init(nn.Linear(64, 128, bias=bias)),
                                      nn.Tanh(),
                                     )
        # ACTOR head
        self.actor_head = layer_init(nn.Linear(128, n_outputs), std=0.01)
        # CRITIC head
        self.critic_head = layer_init(nn.Linear(128, 1), std=1)

    def forward(self, x, actor_or_critic = "actor"):
        if actor_or_critic == "critic":
            return self.critic_head(self.backbone(x))
        else: # actor
            logits = self.actor_head(self.backbone(x))
            return F.softmax(logits, dim = -1)