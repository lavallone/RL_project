import os
import gym
import numpy as np
from flloat.parser.ldlf import LDLfParser
from gym.spaces import MultiDiscrete
from gym_breakout_pygame.wrappers.dict_space import BreakoutDictSpace
from gym_breakout_pygame.wrappers.normal_space import  BreakoutNMultiDiscrete
from gym_breakout_pygame.breakout_env import BreakoutConfiguration
from temprl.wrapper import TemporalGoalWrapper, TemporalGoal
from flloat.semantics import PLInterpretation

"""  ENVIRONMENT related classes """
# we have this wrapper to modify the way we access observation informations
# this to extract in a more suitable way fluents from the environment
class UnwrappedBreakoutEnv(gym.ObservationWrapper):
    def __init__(self, config: BreakoutConfiguration, action_type: str):
        super().__init__(BreakoutDictSpace(config))
        # we need to understand if a column c_i or a row r_i have been destroyed in an environment step 
        self._previous_brick_matrix = None
        self._next_brick_matrix = None
        self.action_type = action_type
        
        # according to the type of action we change the observation space accordingly
        if self.action_type == "fire":
            self.observation_space = MultiDiscrete((
                self.env.observation_space.spaces["paddle_x"].n,
            ))
        else:
            self.observation_space = MultiDiscrete((
                self.env.observation_space.spaces["paddle_x"].n,
                self.env.observation_space.spaces["ball_x"].n,
                self.env.observation_space.spaces["ball_y"].n,
                self.env.observation_space.spaces["ball_x_speed"].n,
                self.env.observation_space.spaces["ball_y_speed"].n
            ))
    
    # here is implemented the logic to check how to bricks matrix is changed or not
    def observation(self, observation):
        new_observation = observation
        new_observation["previous_bricks_matrix"] = self._previous_brick_matrix
        self._previous_brick_matrix = np.copy(self._next_brick_matrix)
        self._next_brick_matrix = new_observation["bricks_matrix"]
        return new_observation
    
    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self._previous_brick_matrix = np.copy(obs["bricks_matrix"])
        self._next_brick_matrix = self._previous_brick_matrix
        return obs
    
class NoGoalEnvWrapper(gym.Wrapper):

    def __init__(self, env, feature_extractor = None):
        super().__init__(env)
        self.feature_extractor = feature_extractor\
            if feature_extractor is not None else (lambda obs, action: obs)
        self.observation_space = self._get_observation_space()

    def _get_observation_space(self) -> gym.spaces.Space:
        return MultiDiscrete(tuple(self.env.observation_space.nvec))

    def step(self, action):
        obs, reward, done, info = super().step(action)
        features = self.feature_extractor(obs=obs, action=action)
        return features, reward, done, info

    def reset(self, **kwargs):
        obs = super().reset()
        features = self.feature_extractor(obs, None)
        return features

"""  ENVIRONMENT utility functions """
# here we should be able to create different type of goals
def make_goal(nb_columns: int = 3, nb_rows: int = 3, rb_type: str = "sx2dx") -> str:
    """
    Define the goal expressed in LDLf logic.
    """
    if rb_type == "sx2dx":
        labels = ["c" + str(column_id) for column_id in range(nb_columns)]
    elif rb_type == "dx2sx":
        labels = ["c" + str(column_id) for column_id in range(nb_columns)]
        labels.reverse()
    elif rb_type == "down2up":
        labels = ["r" + str(row_id) for row_id in range(nb_rows)]
        labels.reverse() # è interessante anche il contrario
    elif rb_type == "up2down":
        labels = ["r" + str(row_id) for row_id in range(nb_rows)]
    elif rb_type == "2targets":
        labels = ["t0", "t1"]
    elif rb_type == "3targets":
        labels = ["t0", "t1", "t2"]
    empty = "(!" + " & !".join(labels) + ")"
    f = "<" + empty + "*;{}>tt"
    regexp = (";" + empty + "*;").join(labels)
    f = f.format(regexp)
    return f

def extract_breakout_columns_fluents(obs, action, targets) -> PLInterpretation:
    brick_matrix = obs["bricks_matrix"]  # type: np.ndarray
    previous_brick_matrix = obs["previous_bricks_matrix"]  # type: np.ndarray
    # here we analyze which columns are broken in the current and in the previous bricks matrix
    previous_broken_columns = np.all(previous_brick_matrix == 0.0, axis=1)
    current_broken_columns = np.all(brick_matrix == 0.0, axis=1)
    compare = (previous_broken_columns == current_broken_columns)  # type: np.ndarray
    # if nothing changed, we return {}
    if compare.all():
        result = PLInterpretation(set())
        return result
    # if a column 'i' has been broken, we return the fluent 'c_i'
    else:
        index = np.argmin(compare)
        fluent = "c" + str(index)
        result = PLInterpretation({fluent})
        return result

def extract_breakout_rows_fluents(obs, action, targets) -> PLInterpretation:
    brick_matrix = obs["bricks_matrix"]  # type: np.ndarray
    previous_brick_matrix = obs["previous_bricks_matrix"]  # type: np.ndarray
    # here we analyze which columns are broken in the current and in the previous bricks matrix
    previous_broken_rows = np.all(previous_brick_matrix == 0.0, axis=0)
    current_broken_rows = np.all(brick_matrix == 0.0, axis=0)
    compare = (previous_broken_rows == current_broken_rows)  # type: np.ndarray
    # if nothing changed, we return {}
    if compare.all():
        result = PLInterpretation(set())
        return result
    # if a row 'i' has been broken, we return the fluent 'r_i'
    else:
        index = np.argmin(compare)
        fluent = "r" + str(index)
        result = PLInterpretation({fluent})
        return result
    
def extract_breakout_targets_fluents(obs, action, targets) -> PLInterpretation:
    brick_matrix = obs["bricks_matrix"]  # type: np.ndarray
    previous_brick_matrix = obs["previous_bricks_matrix"]  # type: np.ndarray
    compare = (previous_brick_matrix == brick_matrix)  # type: np.ndarray
    # if nothing changed, we return {}
    if compare.all():
        result = PLInterpretation(set())
        return result
    # if a target brick 'i' has been broken, we return the fluent 't_i'
    else:
        fluents = set()
        for (t_i,t_j) in targets:
            if compare[t_i, t_j] == False:
                idx = targets.index((t_i,t_j))
                fluent = "t" + str(idx)
                fluents.add(fluent)
        result = PLInterpretation(fluents)
        return result

"""  make ENVIRONMENT function """
def make_env(config: BreakoutConfiguration, output_dir, goal_reward: float = 1000.0,
             action_type: str = "fire_ball", rb_type: str = "sx2dx",
             restraining_bolt: str = "rb", reward_shaping: bool = True, targets: list = []) -> gym.Env:
    """
    Make the Breakout environment.

    :param config: the Breakout configuration.
    :param output_dir: the path to the output directory.
    :param reward_shaping: apply automata-based reward shaping.
    :return: the Gym environment.
    """
    
    if action_type == "fire": 
        combine_function=lambda obs, qs: tuple((obs, *qs))
        feature_extractor_function=(lambda obs, action: (obs["paddle_x"]))
    else:
        combine_function=lambda obs, qs: tuple((*obs, *qs))
        feature_extractor_function=(lambda obs, action: (obs["paddle_x"], obs["ball_x"], obs["ball_y"], obs["ball_x_speed"], obs["ball_y_speed"],))
    
    ####### RESTRAINING BOLTS #######
    if restraining_bolt == "rb":
        if rb_type == "sx2dx" or rb_type == "dx2sx":
            extract_fluents = extract_breakout_columns_fluents
            labels = {"c{}".format(i) for i in range(config.brick_cols)}
        elif rb_type == "down2up" or rb_type == "up2down":
            extract_fluents = extract_breakout_rows_fluents
            labels = {"r{}".format(i) for i in range(config.brick_cols)}
        elif rb_type == "2targets" or rb_type == "3targets":
            extract_fluents = extract_breakout_targets_fluents
            num_labels = 2 if rb_type == "2targets" else 3
            labels = {"t{}".format(i) for i in range(num_labels)}
        
        unwrapped_env = UnwrappedBreakoutEnv(config, action_type)
        
        print("Instantiating the DFA...")
        formula_string = make_goal(config.brick_cols, config.brick_rows, rb_type) # is a simple string representing LDL_f formula
        print(formula_string)
        
        formula = LDLfParser()(formula_string) # flloat --> takes a lot of time! --> non indagare il perchè...
        # abstract class to represent a temporal goal
        tg = TemporalGoal(formula=formula,
                        reward=goal_reward, # reward when the formula is satisfied
                        labels=labels,
                        reward_shaping=reward_shaping,
                        zero_terminal_state=False,
                        extract_fluents=extract_fluents,
                        targets = targets)

        print("Formula: {}".format(formula_string))
        print("Done!")
        # we save the corrispondent automaton
        tg._automaton.to_dot(os.path.join(output_dir, "true_automaton"))
        print("Original automaton at {}".format(os.path.join(output_dir, "true_automaton.svg")))
        
        # gym wrapper to include the temporal goal in the environment
        env = TemporalGoalWrapper(
            unwrapped_env,
            [tg],
            combine=combine_function,
            feature_extractor=feature_extractor_function
        )
    
    ####### NO RESTRAINING BOLTS #######
    elif restraining_bolt == "no_rb":
        unwrapped_env = UnwrappedBreakoutEnv(config, action_type)
        env = NoGoalEnvWrapper(
            unwrapped_env,
            feature_extractor=feature_extractor_function
        )

    return env