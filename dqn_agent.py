import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.optim as optim

BUFFER_SIZE  = int(1e5) 
BATCH_SIZE   = 64       
GAMMA        = 0.99     # discount factor
TAU          = 1e-3     # for soft update of target parameters
LR           = 5e-4              
UPDATE_EVERY = 4        # how often to update the network, C

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """ Interacts with and learns from the environment."""
    def __init__(self, state_size, action_size, seed):
        self.state_size  = state_size
        self.action_size = action_size
        self.seed        = random.seed(seed)

        self.qnetwork_local  = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)   # Replay memory, D
        self.C_step = 1   # Initialize time step (for updating every UPDATE_EVERY steps)
    
    
    def step(self, state, action, reward, next_state, done):
        ''' Update Replay Memory & Learn '''
        
        # 1. Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # 2. Once every [UPDATE_EVERY] model updates, sample a batch of experiences and learn
        if self.C_step % UPDATE_EVERY == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()  
                self.learn(experiences, GAMMA)   
            self.C_step = 1       
        else:
            self.C_step += 1

                
    def act(self, state, eps=0.):    
        ''' Return an action for given state, with epsilon_greedy policy '''
        
        # 1. Predict action values of the state on evaluation mode
        state = torch.from_numpy(np.array(state)).unsqueeze(0).float().to(device) # (1,8)
        #self.qnetwork_local.eval()   
        self.qnetwork_target.eval()   # Double DQN
        with torch.no_grad():
            #action_values = self.qnetwork_local(state)
            action_values = self.qnetwork_target(state)   # Double DQN
        
        # 2. Choose action w. epsilon_greedy policy
        nA = self.action_size
        policy = np.ones(nA)*eps/nA
        policy[torch.argmax(action_values)] = 1 - eps + eps/nA
        action = np.random.choice(range(nA), p=policy)
        
        return action        

        
    def learn(self, experiences, gamma):   
        ''' Update parameters using given batch of experience tuples '''
        
        states, actions, rewards, next_states, dones = experiences  # (batch_size, 1): tensors 
                
        # 1. Predict action values for CURRENT states
        self.qnetwork_local.train()       
        Q_preds_curr_state = self.qnetwork_local(states)            # (batch_size, action_size)
        Q_preds = Q_preds_curr_state.gather(dim=1, index=actions)   # (batch_size, 1)
        
        # 2. Predict action values for NEXT states
        self.qnetwork_target.eval()
        with torch.no_grad():
            Q_preds_next_state = self.qnetwork_target(next_states)  # (batch_size, action_size)
        
        Q_targets = rewards + (gamma * Q_preds_next_state.data.max(1)[0].unsqueeze(1) * (1-dones)) # (batch_size, 1)
             
        # 4. Calculate Loss & Update local network's parameters
        self.optimizer.zero_grad()
        criterion = torch.nn.MSELoss()
        loss = criterion(Q_preds, Q_targets)
        loss.backward()       
        self.optimizer.step()
        
        # 5. Update target network's parameters
        self.soft_targetN_update(self.qnetwork_local, self.qnetwork_target, TAU)  
        
         
    def soft_targetN_update(self,            # w_target = τ*w_local + (1-τ)*w_target
                            local_model,     # weights will be copied from
                            target_model,    # weights will be copied to
                            tau):            # (float): interpolation parameter
        
        for target_w, local_w in zip(target_model.parameters(), local_model.parameters()):
            target_w.data.copy_( tau*local_w.data  + (1.0-tau)*target_w.data )

            
'''namedtuple:
        Student = namedtuple('Student',['name','age','DOB']) 
        S = Student('Nara','19','2541997') 
        S[1]             -> 19
        S.name           -> Nara
        getattr(S,'DOB') -> 2541997
'''
class ReplayBuffer:
    def __init__(self, 
                 action_size,  # dimension of each action
                 buffer_size,  # maximum size of buffer, N
                 batch_size,   # size of each training batch
                 seed          # random seed
        ):        
        self.action_size = action_size
        self.memory      = deque(maxlen = buffer_size)  # if appended (maxlen + 1)th element, 1st element is removed
        self.batch_size  = batch_size
        self.experience  = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])
        self.seed        = random.seed(seed)
        
    
    def add(self, state, action, reward, next_state, done):   
        ''' Add a new experience to memory '''
        
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    
    def sample(self):    
        ''' Randomly sample a batch of experiences from memory '''
        
        tuples = random.sample(self.memory, self.batch_size)       
        # shapes: (batch_size, 1)
        states      = torch.from_numpy(np.vstack([t.state      for t in tuples])).float().to(device) 
        actions     = torch.from_numpy(np.vstack([t.action     for t in tuples])).long().to(device) 
        rewards     = torch.from_numpy(np.vstack([t.reward     for t in tuples])).float().to(device) 
        next_states = torch.from_numpy(np.vstack([t.next_state for t in tuples])).float().to(device) 
        dones       = torch.from_numpy(np.vstack([int(t.done)  for t in tuples])).float().to(device) 

        return (states, actions, rewards, next_states, dones)
    
    
    def __len__(self):
        return len(self.memory)  # size of current internal memory