from ltl_2_dfa.DFA_Graph import DFA
import numpy as np
import os

def update_obs(dfa_list, image):
    l = False
    if type(image) == type([]):
        image = image[0]
        l = True
    # image = np.asarray(image)
    q_states = []
    final_vector = None
    for dfa in dfa_list:
        state = dfa.get_dfa_state(image["image"])  # yet to be implemented
        one_hot = np.zeros(shape=(dfa.num_states,))
        one_hot[int(state)] = 1  # assuming states are starting from 1 to n
        if final_vector is None:
            final_vector = one_hot
        else:
            final_vector = np.concatenate((final_vector,one_hot),axis=0)

    image["dfa_states"] = final_vector
    if l:
        image = [image]
    return image

def update_reward(dfa_list, reward):
    r = 0
    i = 3  # initializing to 3 as first three channels are RGB for the actual image.
    for dfa in dfa_list:
        r += dfa.get_reward()  # yet to be implemented

    return reward + r


def get_dfa_list():
    dfa_names = os.listdir("./dfas/")
    dfa_list = []
    base_path = os.curdir + "./dfas/"
    for dfa_name in dfa_names:
        dfa_list.append(DFA(base_path + dfa_name))

    return dfa_list
