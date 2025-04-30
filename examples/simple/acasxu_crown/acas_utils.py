import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stanleybak_closed_loop.acasxu_dubins import State, state7_to_state5
import numpy as np 
import torch

### return the advisory and normalized acas states
def check_sb(own_state: np.ndarray, int_state: np.ndarray, tau_idx: int = 0, last_cmd: int = 1): # default net is COC and tau=0
    sb_state = [own_state[i] for i in [0,1,3]]+[int_state[i] for i in [0,1,3]]+[0] # x, y, theta, x2, y2, theta2, time (unused)
    s = State(sb_state, tau_idx, -1, own_state[-1], int_state[-1], last_cmd-1)
    sb_ads, sb_norm_acas = s.update_command()
    sb_acas = state7_to_state5(s.vec, s.v_own, s.v_int)
    return sb_ads, sb_acas, sb_norm_acas 