import gym
from train import train_dqn
from dqn_agent import Agent
from simulate import simulate_env


'''
##################################################
############ LUNAR LANDER ENVIRONMENT ############
##################################################

(from: https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py)

STATE = [position_x, position_y,             vel_x,               vel_y,
              angle,  angular_v, left_leg_on_groud, right_leg_on_ground]
                   
ACTION: Discrete(4)- 
        [Do Nothing, fire left engine, main engine, right engine]
    
REWARD: 
    - moving from the top of the screen to landing pad & zero speed : +100..140 
    - If lander moves away from landing pad, it loses reward back
    - Episode finish w. lander crashing       : -100 
    - Episode finish w. lander coming to rest : +100
    - Each leg ground contact                 : +10
    - Firing main engine                      : -0.3/frame
    - Solved                                  : +200 
'''

# 1. Initialize_environment
env = gym.make('LunarLander-v2')
env.seed(0)
state = env.reset()

print('State shape: ', env.observation_space.shape, type(state))
print(state)

# 2. Initialize Agent
agent = Agent(state_size=8, action_size=4, seed=0)

# 3. Train Agent
scores = train_dqn(env, agent, n_episodes=4000)

# 4. Simulate Agent in the Environment
simulate_env(env, agent, model_path='models/checkpoint_1900.pth')