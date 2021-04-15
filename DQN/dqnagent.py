import numpy as np
import random 
from collections import namedtuple, deque 

##Importing the model (function approximator for Q-table)
from DQN.model_architecture import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim


BUFFER_SIZE = int(1e5)  #replay buffer size
BATCH_SIZE = 512         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 32        # how often to update the network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns form environment."""
    
    def __init__(self, state_size, action_size, seed, dfas_size, rmax=100):
        """Initialize an Agent object.
        
        Params
        =======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        
        #Q- Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed, dfas_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, dfas_size).to(device)
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr=LR)
        
        # Replay memory 
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE,BATCH_SIZE,seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.rmax = rmax 
        
    def step(self, state, action, reward, next_step, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_step, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step+1)% UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get radom subset and learn

            if len(self.memory)>BATCH_SIZE:
                experience = self.memory.sample()
                self.learn(experience, GAMMA)


    def read_model(self, path) : 
        self.qnetwork_local.load_state_dict(torch.load(path))
        self.qnetwork_local.eval()

    def act(self, state, eps = 0):
        """Returns action for given state as per current policy
        Params
        =======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        img_state = state['image']
        q_state = state['q']

        state = torch.from_numpy(img_state).float().unsqueeze(0).to(device)
        q_state = q_state.float().unsqueeze(0).to(device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state, q_state)
        self.qnetwork_local.train()

        #Epsilon -greedy action selction
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
            
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        =======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_state, dones, q_states, q_next_states = experiences
        ## TODO: compute and minimize the loss
        criterion = torch.nn.MSELoss()
        self.qnetwork_local.train()
        self.qnetwork_target.eval()
        #shape of output from the model (batch_size,action_dim) = (64,4)
        predicted_targets = self.qnetwork_local(states, q_states).gather(1,actions)
    
        with torch.no_grad():
            labels_next = self.qnetwork_target(next_state, q_next_states).detach().max(1)[0].unsqueeze(1)

        # .detach() ->  Returns a new Tensor, detached from the current graph.
        labels = rewards + (gamma* labels_next*(1-dones))
        
        loss = criterion(predicted_targets,labels).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local,self.qnetwork_target,TAU)
            
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),
                                           local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)
            
class ReplayBuffer:
    """Fixed -size buffe to store experience tuples."""
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.pos_memory = deque(maxlen=buffer_size)

        self.eps = 0.01

        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["state",
                                                               "action",
                                                               "reward",
                                                               "next_state",
                                                               "done",
                                                               "q",
                                                               "next_q"])
        self.seed = random.seed(seed)
        
    def add(self,state, action, reward, next_state,done):
        """Add a new experience to memory."""
        e = self.experiences(state['image'], action, reward, next_state['image'], done, state['q'], next_state['q'])
        self.memory.append(e)

        if(reward > 0):
            self.pos_memory.append(e)

        
    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        # print ("MEMORY SIZE", len(self.memory), len(self.pos_memory), self.batch_size)

        pos_exp_count = int(min(len(self.pos_memory), self.batch_size/4))
        other_exp_count = self.batch_size - pos_exp_count

        experiences = random.sample(self.memory,k= other_exp_count)
        pos_experiences = random.sample(self.pos_memory, k= pos_exp_count)
        experiences = experiences + pos_experiences

        
        # for e in experiences : 
        #     print ("EXPERIENCE : ")
        #     print ("STATE, ", e.state.shape)
        #     print ("Q STATE, ", e.q.shape)

        states = torch.from_numpy(np.vstack([np.expand_dims(e.state, axis=0) for e in experiences if e is not None])).float().to(device)
        q_states = torch.from_numpy(np.vstack([e.q.unsqueeze(0) for e in experiences if e is not None])).float().to(device)

        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        
        next_states = torch.from_numpy(np.vstack([np.expand_dims(e.next_state, axis=0) for e in experiences if e is not None])).float().to(device)
        next_q_states = torch.from_numpy(np.vstack([e.next_q.unsqueeze(0) for e in experiences if e is not None])).float().to(device)

        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states,actions,rewards,next_states,dones, q_states, next_q_states)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)