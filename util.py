from ltl_2_dfa.DFA_Graph import DFA
import numpy as np
import os

def update_obs(dfa_list, image):
    image = np.asarray(image)
    q_states = []
    for dfa in dfa_list:
        state = dfa.get_dfa_state(image)  # yet to be implemented
        one_hot = np.zeros(shape=(dfa.num_states,))
        one_hot[int(state)] = 1  # assuming states are starting from 1 to n
        q_states.append(one_hot)

    for state in q_states:
        t = np.zeros(shape=(image.shape[0],image.shape[1],state.shape[0]))
        for i in range(t.shape[0]):
            t[i,i,:] = state
        image = np.concatenate((image,t),axis=-1)

    return image

def update_reward(dfa_list, reward, image):
    r = 0
    i = 3  # initializing to 3 as first three channels are RGB for the actual image.
    for dfa in dfa_list:
        n_states = dfa.num_states
        one_hot = image[0,0,i:i+n_states]
        i = i + n_states  # to point to the next dfa one hot index
        index = one_hot.argmax()
        r += dfa.get_reward()  # yet to be implemented

    return reward + r


def get_dfa_list():
    dfa_names = os.listdir("./dfas/")
    dfa_list = []
    base_path = os.curdir + "./dfas/"
    for dfa_name in dfa_names:
        dfa_list.append(DFA(base_path + dfa_name))

    return dfa_list
