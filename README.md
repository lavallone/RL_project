# RL_project - Restraining Bolts 🔩 applied to Breakout game 🕹️
Project for Reinforcement Learning course at Sapienza University held by professor Roberto Capobianco.

<p align="center">
  <img src="https://imgur.com/E8XjGbu.png" width="500" height="250">
</p>

The idea of the project came from my interest in merging *Knowledge Representation* techniques with *learning* algorithms (supervised and unsupervised learning methods, but in particular to *Reinforcement Learning* ones). I firmly believe that only thanks to a wise integration of the two AI approaches we can create truly intelligent systems in the future. This thesis has been argued in detail in my essay [**Learning and Reasoning as *yin and yang* of future AI**](https://lavallone.github.io/reasoning_project/essay.pdf), where, alongside the formal (and less formal) explanation of some methods in the literature, I tried to verify their correctnesss by running some *experiments* 🔬🧪 (see the [repository](https://github.com/lavallone/reasoning_project) for more details). Among them, *Restraining Bolts* [1] were the approach that excited me the most. In the first place, because they were devised by a Sapienza research group, and secondly for their practical implications that they can have in real world. <br>
Basically, a *Restraining Bolt* (RB) is a set of logical specifications that can be applied to an RL agent. The agent learns to act so as to conform as much as possible to the RB specifications. But the nice thing about RBs, and this is the reason why they are so powerful, is 
that their representation is completely decoupled from the RL agent's. We have two distinct sets of features: one for the MDP *states* and one for logical formulas' *fluents*.

<p align="center">
  <img src="https://imgur.com/zJ5CVSh.png" width="400" height="180">
</p>

This allows, in practice, to *design* the learning agent feature space separately from the features needed to express the logical goal. This encourages *separation of concerns*, facilitating *modularity* and the *reuse* of representations. For instance, the same agent can learn from different RBs and the same bolt can be applied to different agents. That's exactly the aim of this project:

> ⚡ The primary objective is to individually design distinct Restraining Bolts and Reinforcement Learning algorithms, such as TD methods and DQN, and subsequently explore their integration in various combinations to demonstrate the effectiveness of the paradigm. The chosen test field is *Breakout* environment (Atari game). It's been adopted for its simplicity, its adaptability in setting various levels of difficulty, and its capacity to allow external observers to intuitively grasp the inherent potential of RBs.

## References
<a id="1">[1]</a> 
De Giacomo, G., Iocchi, L., Favorito, M., & Patrizi, F. (2019). Foundations for Restraining Bolts: Reinforcement Learning with LTLf/LDLf Restraining Specifications. Proceedings of the International Conference on Automated Planning and Scheduling, 29(1), 128-136.
