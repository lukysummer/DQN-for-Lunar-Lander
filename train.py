import time
import torch
import numpy as np
from collections import deque

def train_dqn(env, agent, n_episodes, 
              max_step_per_episode=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, print_every=100):
    """PARAMS:
        -env        : (OpenAI environment) OpenAI environment to train the agent in
        -agent      : (Agent class) agent to train 
        -n_episodes : (int)   max # of training episodes
        -max_t      : (int)   max # of timesteps / episode
        -eps_start  : (float) start value of epsilon, for epsilon-greedy action selection
        -eps_end    : (float) minimum value of epsilon
        -eps_decay  : (float) multiplicative factor (per episode) for decreasing epsilon 
        -print_every: (int) print progress & save model every these # of steps 
    """
    
    start = time.time()  
    eps = eps_start
    prev_mean_score = -np.inf
    scores = []
    most_recent_100_scores = deque(maxlen=100)
    for e in range(1, n_episodes, 1):
        state = env.reset()
        score = 0
        
        for _ in range(max_step_per_episode):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            score += reward
            agent.step(state, action, reward, next_state, done)
            if done:
                break
            state = next_state
        
        eps = max(eps*eps_decay, eps_end)            
        scores.append(score)  
        most_recent_100_scores.append(score)
        mean_score = np.mean(most_recent_100_scores)

        if e % print_every == 0:
            print("Episode: {}\t Total Reward: {:.2f}".format(e, mean_score))
            if mean_score > prev_mean_score:
                torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_' + str(e) + '.pth')
                prev_mean_score = mean_score
        
    print("Took {:.2f} minutes for {} episodes.".format((time.time() - start)/60, n_episodes))
    
    return scores