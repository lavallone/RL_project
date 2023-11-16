# RL_project - Restraining Bolts 🔩 applied to Breakout game 🕹️
Project for Reinforcement Learning course at Sapienza University held by professor Roberto Capobianco.

<p align="center">
  <img src="https://imgur.com/E8XjGbu.png" width="500" height="250">
</p>

The idea of the project came from my interest in merging *Knowledge Representation* techniques with *learning* algorithms. In particular with *Reinforcement Learning* ones. I firmly believe that only thanks to a wise integration of the two AI approaches we can create truly intelligent systems in the future. This thesis has been precisely argued in my essay [**Learning and Reasoning as *yin and yang* of future AI**](https://lavallone.github.io/reasoning_project/essay.pdf), where, alongside the formal (and less formal) explanation of some methods in the literature, I tried to verify their correctnesss by running some *experiments* 🔬🧪 (see the [repository](https://github.com/lavallone/reasoning_project) for more details). Among them, *Restraining Bolts* [1] were the approach that excited me the most. In the first place, because they were devised by a Sapienza research group, and secondly for their practical implications that they can have in real world. <br>
Basically, a *restraining bolt* (RB) is a set of logical specifications that can be applied to an RL agent. The nice thing about RBs, and this is the reason why they are so powerful, is the decoupled nature 

Apart from restraining bolts, the interest in having sepa-rate representations is manifold. The learning agent featurespace can be designed separately from the features neededto express the goal, thus promotingseparation of concernswhich, in turn, facilitates the design, providing formodular-ityandreuseof representations (the same agent can learnfrom different bolts and the same bolt can be applied to dif-ferent agents). Also, a reduced agent’s feature space allowsfor realizingsimpler agents(think, e.g., of a mobile robotplatform, where one can avoid specific sensors and percep-tion routines), while preserving the possibility of acting ac-cording to complex declarative specifications which cannotbe represented in the agent’s feature space

The main result of this paper is that, in spite of the looseconnection between the two models, under general circum-stances, theagent can learn to act so as to conform as muchas possible to theLTLf/LDLfspecifications. Observe that wedeal with two separate representations (i.e., two distinct setsof features), one for the agent and one for the bolt, which areapparently unrelated, but in reality, correlated by the worlditself, cf., (Brooks 1991). The crucial point is that, in orderto perform RL effectively in presence of a restraining boltsuch a correlation does not need to be formalized.

the aim of  the project is indeed  show how we can design RL algorithms and applying the same RB or viceversa... 

poi qua parlo del game che è stato scelto...

## References
<a id="1">[1]</a> 
De Giacomo, G., Iocchi, L., Favorito, M., & Patrizi, F. (2019). Foundations for Restraining Bolts: Reinforcement Learning with LTLf/LDLf Restraining Specifications. Proceedings of the International Conference on Automated Planning and Scheduling, 29(1), 128-136.
