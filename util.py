from human_observation.human_obs import DFAWrapper
import numpy as np
import os

def update_obs(dfa_list, images):
    l = False

    # image = np.asarray(image)
    q_states = []
    for i in range(len(images)):
        dfas = dfa_list[i]
        image = images[i]
        final_vector = None
        for dfa in dfas:
            state = dfa.get_dfa_state(image["image"])  # yet to be implemented
            one_hot = np.zeros(shape=(dfa.get_states_count(),))
            one_hot[int(state)] = 1  # assuming states are starting from 1 to n
            if final_vector is None:
                final_vector = one_hot
            else:
                final_vector = np.concatenate((final_vector,one_hot),axis=0)

        image["dfa_states"] = final_vector

    return images

def update_reward(dfa_list, reward):
    reward = list(reward)
    i = 3  # initializing to 3 as first three channels are RGB for the actual image.
    for i in range(len(reward)):
        r = 0
        dfas = dfa_list[i]
        for dfa in dfas:
            r += dfa.get_reward()  # yet to be implemented
        reward[i] = reward[i] + r

    return tuple(reward)


def get_dfa_list(nprocs):
    # dfa_names = os.listdir("./dfas/")
    # dfa_list = []
    # base_path = os.curdir + "./dfas/"
    # for dfa_name in dfa_names:
    #     dfa_list.append(DFA(base_path + dfa_name))
    final_list = []
    for i in range(nprocs):
        dfa_list =[]
        dfa_paths = ["./ltl_2_dfa/neverClaimFiles/never_claim_4.txt"]
        for dfa_name in dfa_paths:
            dfa_list.append(DFAWrapper(dfa_name))
        final_list.append(dfa_list)
    return final_list
