from verse.map.example_map.map_tacas import M1
from verse.scenario.scenario import Benchmark
from enum import Enum, auto
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
from jax_guam.subsystems.genctrl_inputs.genctrl_circle_inputs import QrotZ, quaternion_to_euler, euler_to_quaternion

from typing import List, Tuple
from numba import njit
from aircraft_agent import AircraftAgent
from aircraft_agent_intruder import AircraftAgent_Int
import warnings
from verse.plotter.plotter3D_new import *
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

class AgentMode(Enum):
    COC = auto()
    WL = auto()
    WR = auto()
    SL = auto()
    SR = auto()

means_for_scaling = torch.FloatTensor([19791.091, 0.0, 0.0, 650.0, 600.0])
range_for_scaling = torch.FloatTensor([60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0])
tau_list = [0, 1, 5, 10, 20, 50, 60, 80, 100] 

def get_acas_state(own_state: np.ndarray, int_state: np.ndarray) -> torch.Tensor:
    dist = np.sqrt((own_state[0]-int_state[0])**2+(own_state[1]-int_state[1])**2)
    theta = wrap_to_pi((2*np.pi-own_state[3])+np.arctan2(int_state[1]-own_state[1], int_state[0]-own_state[0]))
    psi = wrap_to_pi(int_state[3]-own_state[3])
    return torch.tensor([dist, theta, psi, own_state[-1], int_state[-1]])

def dubins_to_guam_3d(state: List) -> List:
    v = state[-1]
    theta = np.pi/2-state[3]
    psi = state[4]
    # quat = QrotZ(theta)
    quat = euler_to_quaternion(0,0,theta)
    x,y,z = state[0], state[1], state[2]
    climb_rate = -v*np.sin(psi)
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, v, 0, 0, 0.0, 0.0, 0.0, y, x, -z, float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]), 0.0, 0.0, -0.000780906088785921, -0.000780906088785921, 0.0, 0.000, climb_rate]

# assuming time is not a part of the state
def guam_to_dubins_3d(state: np.ndarray) -> List: 
    vx, vy, vz = state[6:9]
    y, x, z = state[12:15]
    _, psi, theta = quaternion_to_euler(state[15:19])
    v = np.sqrt(vx**2+vy**2+vz**2)
    return [x,y,-z, np.pi/2-float(theta),float(psi), v]

def get_final_states_sim(n) -> Tuple[List]: 
    own_state = n.trace['car1'][-1]
    int_state = n.trace['car2'][-1]
    return own_state, int_state

def get_point_tau(own_state: np.ndarray, int_state: np.ndarray, vz_own, vz_int) -> float:
    z_own, z_int = own_state[2], int_state[2]
    return -(z_int-z_own)/(vz_int-vz_own) # will be negative when z and vz are not aligned, which is fine

def get_tau_idx(own_state: np.ndarray, int_state: np.ndarray, vz_own: float, vz_int: float) -> int:
    tau = get_point_tau(own_state, int_state, vz_own, vz_int)
    # print(tau)
    if tau<0:
        return 0 # following Stanley Bak, if tau<0, return 0 -- note that Stanley Bak also ends simulation if tau<0
    if tau>tau_list[-1]:
        return len(tau_list)-1 
    for i in range(len(tau_list)-1):
        tau_low, tau_up = tau_list[i], tau_list[i+1]
        if tau_low <= tau <= tau_up:
            if np.abs(tau-tau_low)<=np.abs(tau-tau_up):
                return i
            else:
                return i+1
            
    return len(tau_list)-1 # this should be unreachable

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
    # scenario.set_sensor(GUAMSensor())
    
    T = 20
    Tv = 1
    ts = 0.1
    N = 1
    models = [[torch.load(f"./examples/simple/acasxu_crown/nets/ACASXU_run2a_{net + 1}_{tau + 1}_batch_2000.pth") for tau in range(9)] for net in range(5)]
    scenario.config.print_level = 0
    scenario.add_agent(car)
    scenario.add_agent(car2)
    start = time.perf_counter()
    traces = []

    for i in range(N):
        scenario.set_init(
            [[dubins_to_guam_3d([0, -1000, -1, np.pi/3, np.pi/12, 100]), dubins_to_guam_3d([0, -1000, 1, np.pi/3, np.pi/12, 100])],
            # [[dubins_to_guam_3d([-2, -1000, 0, np.pi/3, 0, 100]), dubins_to_guam_3d([-1,-999, 0, np.pi/3, 0, 100])],
            # [[dubins_to_guam_3d([-100, -1000, -1, np.pi/3, 0, 100]), dubins_to_guam_3d([100, -900, 1, np.pi/3, 0, 100])],
              [dubins_to_guam_3d([-2001, -1, 499, 0,0, 100]), dubins_to_guam_3d([-1999, 1, 501, 0,0, 100])]],
            # [dubins_to_guam_3d([-1001, -1, 0, 0,0, 100]), dubins_to_guam_3d([-999, 1, 0, 0,0, 100])]],
            # [[dubins_to_guam_3d([-2, -1, 0, np.pi, 0,100]), dubins_to_guam_3d([-1,1,  0, np.pi,  0,100])],
            # [dubins_to_guam_3d([-1001, -1, 0, 0, 0,100]), dubins_to_guam_3d([-999, 1,  0, 0, 0,100])]],
            [(AgentMode.COC,), (AgentMode.COC,)]
        )
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
            vz_own, vz_int = own_state[-1], int_state[-1]
            dub_own_state, dub_int_state = guam_to_dubins_3d(own_state[1:]), guam_to_dubins_3d(int_state[1:])
            tau_idx = get_tau_idx(dub_own_state, dub_int_state, vz_own, vz_int)
            acas_state = get_acas_state(dub_own_state, dub_int_state).float()
            print(f'ACAS state: {acas_state}')
            acas_state = (acas_state-means_for_scaling)/range_for_scaling # normalization
            last_cmd = getattr(AgentMode, cur_node.mode['car1'][0]).value  # cur_mode.mode[.] is some string 
            ads = models[last_cmd-1][tau_idx](acas_state.view(1,5)).detach().numpy()
            new_mode = np.argmin(ads[0])+1 # will eventually be a list
            car.set_initial(
                initial_state=[own_state[1:], own_state[1:]],
                initial_mode=([AgentMode(new_mode)])
            )
            car2.set_initial(
                initial_state=[int_state[1:], int_state[1:]],
                initial_mode=([AgentMode.COC])
            )
            print(f'New mode: {AgentMode(new_mode)}')
            print(f'Advisory scores: {ads}')
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
        print(f'Simulation {i} complete')
        traces.append(trace)
    # for node in trace.nodes:
    #     print(f'Start time: {node.start_time}, Mode: ', node.mode['car1'][0])
    print(f'Total runtime: {time.perf_counter()-start} for {N} simulation(s) given time steps = {ts}')
    fig = go.Figure()
    for trace in traces:
        # fig = simulation_tree(trace, None, fig, 14, 13, [14, 13], "fill", "trace")
        fig = simulation_tree_3d(trace, fig,14,'x', 13,'y',15,'z')
    fig.show()