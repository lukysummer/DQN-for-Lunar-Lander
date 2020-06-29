from dqn_agent import Agent
import time
import torch
import matplotlib.pyplot as plt

agent = Agent(state_size=8, action_size=4, seed=0)

def simulate_env(env, agent, model_path, n_simul=5):
    agent.qnetwork_local.load_state_dict(torch.load(model_path, map_location='cpu'))
    agent.qnetwork_target.load_state_dict(torch.load(model_path, map_location='cpu'))

    for _ in range(n_simul):
        state = env.reset()
        plt.imshow(env.render(mode='rgb_array'))
        time.sleep(0.01)
        score = 0
        for _ in range(1000):
            action = agent.act(state)
            plt.imshow(env.render(mode='rgb_array'))
            time.sleep(0.01)
            state, reward, done, _ = env.step(action)
            if done:
                break       
            score += reward
            
        print(score)
            
    env.close()