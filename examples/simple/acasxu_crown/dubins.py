from dubins_agent import CarAgent, NPCAgent
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
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

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

class Model(nn.Module):
    def __init__(self, net: int = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = torch.load(f"./examples/simple/acasxu_crown/ACASXU_run2a_{net + 1}_1_batch_2000.pth")
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        res = get_acas_state_torch(x,y)
        res = self.model(res)
        return res

def get_acas_state(own_state: np.ndarray, int_state: np.ndarray) -> torch.Tensor:
    dist = np.sqrt((own_state[0]-int_state[0])**2+(own_state[1]-int_state[1])**2)
    theta = wrap_to_pi((2*np.pi-own_state[2])+np.arctan2(int_state[1], int_state[0]))
    psi = wrap_to_pi(int_state[2]-own_state[2])
    return torch.tensor([dist, theta, psi, own_state[3], int_state[3]])

def wtp(x: float): 
    return torch.remainder((x + torch.pi), (2 * torch.pi)) - torch.pi

def get_acas_state_torch(own_state: torch.Tensor, int_state: torch.Tensor) -> torch.Tensor:
    dist = torch.sqrt((own_state[:,0:1]-int_state[:,0:1])**2+(own_state[:,1:2]-int_state[:,1:2])**2)
    theta = wtp((2*torch.pi-own_state[:,2:3])+torch.arctan2(int_state[:,1:2], int_state[:,0:1]))
    # theta = wtp((2*torch.pi-own_state[:,2:3])+torch.arctan(int_state[:,1:2]/int_state[:,0:1]))
    psi = wtp(int_state[:,2:3]-own_state[:,2:3])
    # return torch.cat([dist, own_state[:,3:4], psi, own_state[:,3:4], int_state[:,3:4]], dim=1)
    return torch.cat([dist, theta, psi, own_state[:,3:4], int_state[:,3:4]], dim=1)

def get_final_states_sim(n) -> Tuple[List]: 
    own_state = n.trace['car1'][-1]
    int_state = n.trace['car2'][-1]
    return own_state, int_state

def get_final_states_verify(n) -> Tuple[List]: 
    own_state = n.trace['car1'][-2:]
    int_state = n.trace['car2'][-2:]
    return own_state, int_state

if __name__ == "__main__":
    import os
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "controller.py")
    car = CarAgent('car1', file_name=input_code_name)
    car2 = NPCAgent('car2')
    scenario = Scenario(ScenarioConfig(parallel=False))
    car.set_initial(
        # initial_state=[[0, -0.5, 0, 1.0], [0.01, 0.5, 0, 1.0]],
        initial_state=[[0, -1000, np.pi/3, 100], [0, -1000, np.pi/3, 100]],
        initial_mode=(AgentMode.COC, TrackMode.T1)
    )
    car2.set_initial(
        # initial_state=[[15, 15, 0, 0.5], [15, 15, 0, 0.5]],
        initial_state=[[-2000, 0, 0, 100], [-2000, 0, 0, 100]],
        initial_mode=(AgentMode.COC, TrackMode.T1)
    )
    T = 20
    Tv = 0.1
    ts = 0.05

    scenario.config.print_level = 0
    scenario.add_agent(car)
    scenario.add_agent(car2)
    # trace = scenario.simulate(Tv, ts) # this is the root
    trace = scenario.verify(Tv, ts) # this is the root
    id = 1+trace.root.id
    net = 0 # eventually this could be modified in the loop by some cmd_list var
    # model = torch.load(f"./examples/simple/acasxu_crown/ACASXU_run2a_{net + 1}_1_batch_2000.pth")
    model = Model()
    norm = float("inf")


    queue = deque()
    queue.append(trace.root) # queue should only contain ATNs  
    ### begin looping
    while len(queue):
        cur_node = queue.popleft() # equivalent to trace.nodes[0] in this case
        # own_state, int_state = get_final_states_sim(cur_node)
        own_state, int_state = get_final_states_verify(cur_node)
        # acas_state = get_acas_state(own_state[1:], int_state[1:]).float()
        # ads = model(acas_state.view(1,5)).detach().numpy()
        x_l, x_u = torch.tensor(own_state[0]).float().view(1,5), torch.tensor(own_state[1]).float().view(1,5)
        x = (x_l+x_u)/2
        y_l, y_u = torch.tensor(int_state[0]).float().view(1,5), torch.tensor(int_state[1]).float().view(1,5)
        y = (y_l+y_u)/2
        lirpa_model = BoundedModule(model, (torch.empty_like(x), torch.empty_like(y)))
        ptb_x = PerturbationLpNorm(norm = norm, x_L=x_l, x_U=x_u)
        ptb_y = PerturbationLpNorm(norm = norm, x_L=y_l, x_U=y_u)
        bounded_x = BoundedTensor(x, ptb=ptb_x)
        bounded_y = BoundedTensor(y, ptb=ptb_y)
        lb, ub = lirpa_model.compute_bounds((bounded_x,bounded_y), method='alpha-CROWN')

        print(lb, ub)
        exit()
        ads = model(torch.tensor(own_state[1:]).float(), torch.tensor(int_state[1:]).float()).detach().numpy()

        print(ads)
        new_mode = np.argmax(ads[0])+1 # will eventually be a list
        print(AgentMode(new_mode))
        # this will eventually be a loop
        scenario.set_init(
            [[own_state[1:], own_state[1:]], [int_state[1:], int_state[1:]]], # this should eventually be a range 
            [(AgentMode(new_mode), TrackMode.T1),(AgentMode.COC, TrackMode.T1)]
        )
        id += 1
        new_trace = scenario.simulate(Tv, ts)
        temp_root = new_trace.root
        new_node = cur_node.new_child(temp_root.init, temp_root.mode, temp_root.trace, cur_node.start_time + Tv, id)
        cur_node.child.append(new_node)
        if new_node.start_time + Tv>=T: # if the time of the current simulation + start_time is at or above total time, don't add
            continue
        queue.append(new_node)

    trace.nodes = trace._get_all_nodes(trace.root)
    fig = go.Figure()
    fig = simulation_tree(trace, None, fig, 1, 2, [1, 2], "fill", "trace")
    fig.show()
    # trace = scenario.verify(0.2,0.1) # increasing ts to 0.1 to increase learning speed, do the same for dryvr2
    # fig = reachtube_tree(trace) 
    # fig.show() 