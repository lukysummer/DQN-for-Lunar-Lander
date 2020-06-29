# Deep Q-Network for OpenAI's [Lunar Lander](https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py) (PyTorch)


## Result

<p align="center"><img src="final_simul.mp4" width = "600" height = "400"></p>


## DQN was implemented with following tricks:

* **Fixed Q-target** : separate local & target networks
* **Experience Replay** : Having a buffer of (state, action, reward, next_state, done) tuples to sample from 
* **Double DQN** : using target network to evaluate the model- when choosing action maximizing action-value function 
* **ε-greedy Policy** : choosing non-greedy action with probability = ε (starts at 1 and decays to 0 each episode) 


## Weight Update formula for DQN: 

<p align="center"><img src="formula.png" width = "550" height = "100"></p>


## Hyperparameters

* **n_episodes** : 4000
* **model architecture** : 2 fully connected layers (h=32)
* **Reply Buffer Capacity** : 100,000 tuples
* **batch size** : 64       
* **discount rate, γ** : 0.99    
* **soft update factor, τ ** (for target network params) : 0.001 1e-3     
* **learning rate** : 0.0005               
* **Update weights every** 4 episode steps 

Final model checkpoint producing above simulation is in `models/` folder. 
