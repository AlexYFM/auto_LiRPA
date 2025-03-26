from typing import List, Tuple
from numba import njit
from aircraft_agent import AircraftAgent
from aircraft_agent_intruder import AircraftAgent_Int
import warnings
from verse.plotter.plotter3D import *
from verse.plotter.plotter2D import *
import matplotlib.pyplot as plt
import os
import functools as ft
import ipdb
import jax
import jax.random as jr
import jax.tree_util as jtu
from jax_guam.functional.guam_new import FuncGUAM, GuamState
from jax_guam.utils.jax_utils import jax2np, jax_use_cpu, jax_use_double
from jax_guam.utils.logging import set_logger_format
from loguru import logger

from GUAM_sensor import GUAMSensor

from verse.map.example_map.map_tacas import M1
from verse.scenario.scenario import Benchmark
from enum import Enum, auto
from verse.plotter.plotter2D import *
from verse import Scenario, ScenarioConfig
from verse.analysis.verifier import ReachabilityMethod
import sys
import plotly.graph_objects as go
import torch
from auto_LiRPA import BoundedTensor
from verse.utils.utils import wrap_to_pi
import numpy as np 
import torch
from collections import deque
from torch import nn
import time

class AgentMode(Enum):
    COC = auto()
    WL = auto()
    WR = auto()
    SL = auto()
    SR = auto()

class TrackMode(Enum):
    T0 = auto()
    T1 = auto()
    T2 = auto()
    M01 = auto()
    M12 = auto()
    M21 = auto()
    M10 = auto()

means_for_scaling = torch.FloatTensor([19791.091, 0.0, 0.0, 650.0, 600.0])
range_for_scaling = torch.FloatTensor([60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0])

def get_acas_state(own_state: np.ndarray, int_state: np.ndarray) -> torch.Tensor:
    dist = np.sqrt((own_state[0]-int_state[0])**2+(own_state[1]-int_state[1])**2)
    theta = wrap_to_pi((2*np.pi-own_state[2])+np.arctan2(int_state[1]-own_state[1], int_state[0]-own_state[0]))
    psi = wrap_to_pi(int_state[2]-own_state[2])
    return torch.tensor([dist, theta, psi, own_state[3], int_state[3]])

def get_acas_state_torch(own_state: torch.Tensor, int_state: torch.Tensor) -> torch.Tensor:
    def wtp(x: float): 
        return torch.remainder((x + torch.pi), (2 * torch.pi)) - torch.pi
    dist = torch.sqrt((own_state[0]-int_state[0])**2+(own_state[1]-int_state[1])**2)
    theta = wtp((2*torch.pi-own_state[2])+torch.arctan2(int_state[1]-own_state[1], int_state[0]-own_state[0]))
    psi = wtp(int_state[2]-own_state[2])
    return torch.tensor([dist, theta, psi, own_state[3], int_state[3]])

def dubins_to_guam_2d(state: List) -> List:
    v = state[-1]
    theta = state[-2]
    vy = v*np.sin(theta)
    vx = v*np.cos(theta)
    vz = 0 # set to a constant for now
    x,y,z = state[0], state[1], 0
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, vx, vy, vz, 0.0, 0.0, 0.0, x, y, z, 1.0, 0.0, -4.3136e-05, 0., 0.0, 0.0, -0.000780906088785921, -0.000780906088785921, 0.0, 0.000, -1.0]

# assuming time is not a part of the state
def guam_to_dubins_2d(state: np.ndarray) -> List: 
    vx, vy, vz = state[6:9]
    x, y = state[12:14]
    theta = wrap_to_pi(np.arctan2(vy, vx))
    v = np.sqrt(vx**2+vy**2+vz**2)
    return [x,y,theta,v]

def get_final_states_sim(n) -> Tuple[List]: 
    own_state = n.trace['car1'][-1]
    int_state = n.trace['car2'][-1]
    return own_state, int_state

if __name__ == "__main__":
    import os
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "dl_acas.py")
    # car = CarAgent('car1', file_name=input_code_name)
    # car2 = NPCAgent('car2')
    car = AircraftAgent("car1", file_name=input_code_name)
    input_code_name = os.path.join(script_dir, "dl_acas_intruder.py")
    car2 = AircraftAgent_Int("car2", file_name=input_code_name)
    scenario = Scenario(ScenarioConfig(parallel=False))
    scenario.set_sensor(GUAMSensor())
    
    car.set_initial(
        # initial_state=[[0, -0.5, 0, 1.0], [0.01, 0.5, 0, 1.0]],
        # initial_state=[[0, -1000, np.pi/3, 100], [0, -1000, np.pi/3, 100]],
        initial_state=[dubins_to_guam_2d([0, -1100, np.pi/3, 100]), dubins_to_guam_2d([0, -900, np.pi/3, 100])],
        initial_mode=([AgentMode.COC])
    )
    car2.set_initial(
        # initial_state=[[15, 15, 0, 0.5], [15, 15, 0, 0.5]],
        initial_state=[dubins_to_guam_2d([-2000, 0, 0, 100]), dubins_to_guam_2d([-2000, 0, 0, 100])],
        initial_mode=([AgentMode.COC])
    )
    T = 100
    Tv = 1
    ts = 0.1
    N = 1
    models = [torch.load(f"./examples/simple/acasxu_crown/ACASXU_run2a_{net + 1}_1_batch_2000.pth") for net in range(5)]
    scenario.config.print_level = 0
    scenario.add_agent(car)
    scenario.add_agent(car2)
    start = time.perf_counter()
    traces = []

    for i in range(N):
        # scenario.set_init(
        #     [[dubins_to_guam_2d([0, -1000, np.pi/3, 100]), dubins_to_guam_2d([0, -1000, np.pi/3, 100])], 
        #      [dubins_to_guam_2d([-2000, 0, 0, 100]), dubins_to_guam_2d([-2000, 0, 0, 100])]],
        #     [([AgentMode.COC]), ([AgentMode.COC])]
        # )
        trace = scenario.simulate(Tv, ts) # this is the root
        id = 1+trace.root.id
        # net = 0 # eventually this could be modified in the loop by some cmd_list var
        # model = torch.load(f"./examples/simple/acasxu_crown/ACASXU_run2a_{net + 1}_1_batch_2000.pth")
        queue = deque()
        queue.append(trace.root) # queue should only contain ATNs  
        ### begin looping
        while len(queue):
            cur_node = queue.popleft() # equivalent to trace.nodes[0] in this case
            own_state, int_state = get_final_states_sim(cur_node)
            dub_own_state, dub_int_state = guam_to_dubins_2d(own_state[1:]), guam_to_dubins_2d(int_state[1:])
            print(dub_own_state, dub_int_state)
            # acas_state = get_acas_state(own_state[1:], int_state[1:]).float()
            acas_state = get_acas_state(dub_own_state, dub_int_state).float()
            acas_state = (acas_state-means_for_scaling)/range_for_scaling # normalization
            # ads = model(acas_state.view(1,5)).detach().numpy()
            last_cmd = getattr(AgentMode, cur_node.mode['car1'][0]).value  # cur_mode.mode[.] is some string 
            ads = models[last_cmd-1](acas_state.view(1,5)).detach().numpy()
            # print(ads)
            # new_mode = np.argmax(ads[0])+1 # will eventually be a list
            new_mode = np.argmin(ads[0])+1 # will eventually be a list
            # print(AgentMode(new_mode))
            # if AgentMode(new_mode)==AgentMode.WL:
            #     print(cur_node.start_time)
            #     exit()
            # this will eventually be a loop
            # scenario.set_init(
            #     [[own_state[1:], own_state[1:]], [int_state[1:], int_state[1:]]], # this should eventually be a range 
            #     [([AgentMode(new_mode)]),([AgentMode.COC])]
            # )
            car.set_initial(
                initial_state=[own_state[1:], own_state[1:]],
                initial_mode=([AgentMode(new_mode)])
            )
            car2.set_initial(
                initial_state=[int_state[1:], int_state[1:]],
                initial_mode=([AgentMode.COC])
            )
            scenario.add_agent(car)
            scenario.add_agent(car2)
            id += 1
            new_trace = scenario.simulate(Tv, ts)
            temp_root = new_trace.root
            new_node = cur_node.new_child(temp_root.init, temp_root.mode, temp_root.trace, cur_node.start_time + Tv, id)
            cur_node.child.append(new_node)
            if new_node.start_time + Tv>=T: # if the time of the current simulation + start_time is at or above total time, don't add
                continue
            queue.append(new_node)

        trace.nodes = trace._get_all_nodes(trace.root)
        traces.append(trace)
    # for node in trace.nodes:
    #     print(f'Start time: {node.start_time}, Mode: ', node.mode['car1'][0])
    print(f'Total runtime: {time.perf_counter()-start} for {N} simulation(s)')
    fig = go.Figure()
    for trace in traces:
        # fig = simulation_tree(trace, None, fig, 1, 2, [1, 2], "fill", "trace")
        fig = simulation_tree(trace, None, fig, 13, 14, [13, 14], "fill", "trace")
    fig.show()
    # trace = scenario.verify(0.2,0.1) # increasing ts to 0.1 to increase learning speed, do the same for dryvr2
    # fig = reachtube_tree(trace) 
    # fig.show() 