from dubins_3d_agent import CarAgent, NPCAgent
from verse.map.example_map.map_tacas import M1
from verse.scenario.scenario import Benchmark
from enum import Enum, auto
from verse.plotter.plotter2D import *
from verse.plotter.plotter3D_new import *
from verse import Scenario, ScenarioConfig
from verse.analysis.verifier import ReachabilityMethod
import sys
import plotly.graph_objects as go
import torch
from auto_LiRPA import BoundedTensor
from verse.utils.utils import wrap_to_pi
import numpy as np 
from collections import deque
from torch import nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
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
tau_list = [0, 1, 5, 10, 20, 50, 60, 80, 100] 
# tau = -(z_int-z_own)/(vz_int-vz_own)
# corresponds to last index (1,2,...,9), if >100 return index of 100, if <0 can return index of 0 but also means no chance of collision -- for ref, SB just stops simulating past tau<0
# if between two taus (e.g., 60 and 80), then choose the closer one, rounding down following SB's examples (e.g. if at tau=70, choose index of 60 instead of 80)

# recall new_state is [x,y,z,th,psi,v]
def get_acas_state(own_state: np.ndarray, int_state: np.ndarray) -> torch.Tensor:
    dist = np.sqrt((own_state[0]-int_state[0])**2+(own_state[1]-int_state[1])**2)
    theta = wrap_to_pi((2*np.pi-own_state[3])+np.arctan2(int_state[1]-own_state[1], int_state[0]-own_state[0]))
    psi = wrap_to_pi(int_state[3]-own_state[3])
    return torch.tensor([dist, theta, psi, own_state[-1], int_state[-1]])

# recall new_state is [x,y,z,th,psi,v]
### expects some 2x5 np arrays for both sets
def get_acas_reach(own_set: np.ndarray, int_set: np.ndarray) -> list[tuple[torch.Tensor]]: 
    def dist(pnt1, pnt2):
        return np.linalg.norm(
            np.array(pnt1) - np.array(pnt2)
        )

    def get_extreme(rect1, rect2):
        lb11 = rect1[0]
        lb12 = rect1[1]
        ub11 = rect1[2]
        ub12 = rect1[3]

        lb21 = rect2[0]
        lb22 = rect2[1]
        ub21 = rect2[2]
        ub22 = rect2[3]

        # Using rect 2 as reference
        left = lb21 > ub11 
        right = ub21 < lb11 
        bottom = lb22 > ub12
        top = ub22 < lb12

        if top and left: 
            dist_min = dist((ub11, lb12),(lb21, ub22))
            dist_max = dist((lb11, ub12),(ub21, lb22))
        elif bottom and left:
            dist_min = dist((ub11, ub12),(lb21, lb22))
            dist_max = dist((lb11, lb12),(ub21, ub22))
        elif top and right:
            dist_min = dist((lb11, lb12), (ub21, ub22))
            dist_max = dist((ub11, ub12), (lb21, lb22))
        elif bottom and right:
            dist_min = dist((lb11, ub12),(ub21, lb22))
            dist_max = dist((ub11, lb12),(lb21, ub22))
        elif left:
            dist_min = lb21 - ub11 
            dist_max = np.sqrt((lb21 - ub11)**2 + max((ub22-lb12)**2, (ub12-lb22)**2))
        elif right: 
            dist_min = lb11 - ub21 
            dist_max = np.sqrt((lb21 - ub11)**2 + max((ub22-lb12)**2, (ub12-lb22)**2))
        elif top: 
            dist_min = lb12 - ub22
            dist_max = np.sqrt((ub12 - lb22)**2 + max((ub21-lb11)**2, (ub11-lb21)**2))
        elif bottom: 
            dist_min = lb22 - ub12 
            dist_max = np.sqrt((ub22 - lb12)**2 + max((ub21-lb11)**2, (ub11-lb21)**2)) 
        else: 
            dist_min = 0 
            dist_max = max(
                dist((lb11, lb12), (ub21, ub22)),
                dist((lb11, ub12), (ub21, lb22)),
                dist((ub11, lb12), (lb21, ub12)),
                dist((ub11, ub12), (lb21, lb22))
            )
        return dist_min, dist_max

    own_rect = [own_set[i//2][i%2] for i in range(4)]
    int_rect = [int_set[i//2][i%2] for i in range(4)]
    d_min, d_max = get_extreme(own_rect, int_rect)

    own_ext = [(own_set[i%2][0], own_set[i//2][1]) for i in range(4)] # will get ll, lr, ul, ur in order
    int_ext = [(int_set[i%2][0], int_set[i//2][1]) for i in range(4)] 

    arho_min = np.inf # does this make sense
    arho_max = -np.inf
    arho_pi_wrap = []
    arho_origin_wrap = []
    for own_vert in own_ext:
        for int_vert in int_ext:
            # arho = np.arctan2(int_vert[1]-own_vert[1],int_vert[0]-own_vert[0]) % (2*np.pi) 
            # arho_max = max(arho_max, arho)
            # arho_min = min(arho_min, arho)
            arho_pi_wrap.append(wrap_to_pi(np.arctan2(int_vert[1]-own_vert[1],int_vert[0]-own_vert[0])))
            arho_origin_wrap.append(np.arctan2(int_vert[1]-own_vert[1],int_vert[0]-own_vert[0]) % (2*np.pi))
    if max(arho_origin_wrap)-min(arho_origin_wrap)<max(arho_pi_wrap)-min(arho_pi_wrap):
        arho_min, arho_max = min(arho_origin_wrap), max(arho_origin_wrap)
    else:
        arho_min, arho_max = min(arho_pi_wrap), max(arho_pi_wrap)

    theta_min = wrap_to_pi((2*np.pi-own_set[1][3])+arho_min)
    theta_max = wrap_to_pi((2*np.pi-own_set[0][3])+arho_max) 
    theta_maxs = []
    theta_mins = []
    if theta_max<theta_min: # bound issue due to wrapping
        theta_mins = [-np.pi, theta_min]
        theta_maxs = [theta_max, np.pi]
    else:
        theta_mins = [theta_min]
        theta_maxs = [theta_max]

    psi_min = wrap_to_pi(int_set[0][3]-own_set[1][3])
    psi_max = wrap_to_pi(int_set[1][3]-own_set[0][3])
    psi_maxs = []
    psi_mins = []
    if psi_max<psi_min: # bound issue due to wrapping
        psi_mins = [-np.pi, psi_min]
        psi_maxs = [psi_max, np.pi]
    else:
        psi_mins = [psi_min]
        psi_maxs = [psi_max]

    sets = [(torch.tensor([d_min, theta_mins[i], psi_mins[j], own_set[0][-1], int_set[0][-1]]), 
             torch.tensor([d_max, theta_maxs[i], psi_maxs[j], own_set[1][-1], int_set[1][-1]])) for i in range(len(theta_mins)) for j in range(len(psi_mins))]
    
    return sets

def wtp(x: float): 
    return torch.remainder((x + torch.pi), (2 * torch.pi)) - torch.pi

def get_acas_state_torch(own_state: torch.Tensor, int_state: torch.Tensor) -> torch.Tensor:
    dist = torch.sqrt((own_state[:,0:1]-int_state[:,0:1])**2+(own_state[:,1:2]-int_state[:,1:2])**2)
    theta = wtp((2*torch.pi-own_state[:,3:4])+torch.arctan2(int_state[:,1:2], int_state[:,0:1]))
    # theta = wtp((2*torch.pi-own_state[:,2:3])+torch.arctan(int_state[:,1:2]/int_state[:,0:1]))
    psi = wtp(int_state[:,3:4]-own_state[:,3:4])
    # return torch.cat([dist, own_state[:,3:4], psi, own_state[:,3:4], int_state[:,3:4]], dim=1)
    return torch.cat([dist, theta, psi, own_state[:,5:6], int_state[:,5:6]], dim=1)

def get_final_states_sim(n) -> Tuple[List]: 
    own_state = n.trace['car1'][-1]
    int_state = n.trace['car2'][-1]
    return own_state, int_state

def get_final_states_verify(n) -> Tuple[List]: 
    own_state = n.trace['car1'][-2:]
    int_state = n.trace['car2'][-2:]
    return own_state, int_state

def get_point_tau(own_state: np.ndarray, int_state: np.ndarray) -> float:
    z_own, z_int = own_state[2], int_state[2]
    vz_own, vz_int = own_state[-1]*np.sin(own_state[-2]), int_state[-1]*np.sin(int_state[-2])
    return -(z_int-z_own)/(vz_int-vz_own) # will be negative when z and vz are not aligned, which is fine

def get_tau_idx(own_state: np.ndarray, int_state: np.ndarray) -> int:
    tau = get_point_tau(own_state, int_state)
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
'''
NOTE: In general, if vz is constant, then tau_dot = -1
'''

if __name__ == "__main__":
    import os
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "controller_3d.py")
    car = CarAgent('car1', file_name=input_code_name)
    car2 = CarAgent('car2', file_name=input_code_name)
    scenario = Scenario(ScenarioConfig(parallel=False))
    # [[[-100, -100, np.pi, 100], [100, 100, np.pi, 100]], [[-4000, 0, 0, 100], [-4000, 0, 0, 100]]]
    car.set_initial(
        # initial_state=[[-1, -1010, -1, np.pi/3, np.pi/6, 100], [1, -990, 1, np.pi/3, np.pi/6, 100]],
        # initial_state=[[0, -1000, 0, np.pi/3, np.pi/6, 100], [0, -1000, 0, np.pi/3, np.pi/6, 100]],
        # initial_state=[[-1, -1, -1, np.pi, np.pi/6, 100], [1, 1, 1, np.pi, np.pi/6, 100]],
        initial_state = [[-2, -1, -2, np.pi, np.pi/6, 100], [-1,1, -1, np.pi, np.pi/6, 100]],
        initial_mode=(AgentMode.COC, )
    )
    car2.set_initial(
        # initial_state=[[-2000, 0, 1000, 0,0, 100], [-2000, 0, 1000, 0,0, 100]],
        # initial_state=[[-2001, -10, 999, 0,0, 100], [-1999, 10, 1001, 0,0, 100]],
        # initial_state=[[-4001, -1, 999, 0,0, 100], [-3999, 1, 1000, 0,0, 100]],
        initial_state= [[-1001, -1, 499, 0,0, 100], [-999, 1, 500, 0,0, 100]],
        initial_mode=(AgentMode.COC, )
    )
    T = 10
    Tv = 1
    ts = 0.01
    # observation: for Tv = 0.1 and a larger initial set of radius 10 in y dim, the number of 

    scenario.config.print_level = 0
    scenario.config.reachability_method = ReachabilityMethod.DRYVR_DISC
    scenario.add_agent(car)
    scenario.add_agent(car2)
    start = time.perf_counter()
    trace = scenario.verify(Tv, ts) # this is the root
    id = 1+trace.root.id
    models = [[torch.load(f"./examples/simple/acasxu_crown/nets/ACASXU_run2a_{net + 1}_{tau + 1}_batch_2000.pth") for tau in range(9)] for net in range(5)]
    norm = float("inf")

    queue = deque()
    queue.append(trace.root) # queue should only contain ATNs  
    ### begin looping
    while len(queue):
        cur_node = queue.popleft() # equivalent to trace.nodes[0] in this case
        own_state, int_state = get_final_states_verify(cur_node)
        tau_idx_min, tau_idx_max = get_tau_idx(own_state[1][1:], int_state[0][1:]), get_tau_idx(own_state[0][1:], int_state[1][1:]) 
        
        ### make below a function when I have time, will eventually need to do pairwise comparisons between all agents 
        modes = set()
        reachsets = get_acas_reach(np.array(own_state)[:,1:], np.array(int_state)[:,1:])
        # print(reachsets)
        for reachset in reachsets:
            if len(modes)==5: # if all modes are possible, stop iterating
                break 
            acas_min, acas_max = reachset
            acas_min, acas_max = (acas_min-means_for_scaling)/range_for_scaling, (acas_max-means_for_scaling)/range_for_scaling
            x_l, x_u = torch.tensor(acas_min).float().view(1,5), torch.tensor(acas_max).float().view(1,5)
            x = (x_l+x_u)/2
            print(reachset)
            last_cmd = getattr(AgentMode, cur_node.mode['car1'][0]).value  # cur_mode.mode[.] is some string 
            for tau_idx in range(tau_idx_min, tau_idx_max+1):
                lirpa_model = BoundedModule(models[last_cmd-1][tau_idx], (torch.empty_like(x))) 
                # lirpa_model = BoundedModule(model, (torch.empty_like(x))) 
                ptb_x = PerturbationLpNorm(norm = norm, x_L=x_l, x_U=x_u)
                bounded_x = BoundedTensor(x, ptb=ptb_x)
                lb, ub = lirpa_model.compute_bounds(bounded_x, method='alpha-CROWN')
                print(f'\n Own Advisory ranges:', lb, ub,'\n')

                # new_mode = np.argmax(ub.numpy())+1 # will eventually be a list/need to check upper and lower bounds
                new_mode = np.argmin(lb.numpy())+1 # will eventually be a list/need to check upper and lower bounds
                
                new_modes = []
                for i in range(len(ub.numpy()[0])):
                    # upper = ub.numpy()[0][i]
                    # if upper>=lb.numpy()[0][new_mode-1]:
                    #     new_modes.append(i+1)
                    lower = lb.numpy()[0][i]
                    if lower<=ub.numpy()[0][new_mode-1]:
                        new_modes.append(i+1)
                modes.update(new_modes)
        
        int_modes = set()
        int_reachsets = get_acas_reach(np.array(int_state)[:,1:], np.array(own_state)[:,1:])
        tau_idx_min_int, tau_idx_max_int = get_tau_idx(int_state[1][1:], own_state[0][1:]), get_tau_idx(int_state[0][1:], own_state[1][1:]) 
        # print('Tau own:',tau_idx_min, tau_idx_max)
        # print('Tau int:',tau_idx_min_int, tau_idx_max_int)
        # print(reachsets)
        for reachset in int_reachsets:
            if len(int_modes)==5: # if all modes are possible, stop iterating
                break 
            acas_min, acas_max = reachset
            acas_min, acas_max = (acas_min-means_for_scaling)/range_for_scaling, (acas_max-means_for_scaling)/range_for_scaling
            x_l, x_u = torch.tensor(acas_min).float().view(1,5), torch.tensor(acas_max).float().view(1,5)
            x = (x_l+x_u)/2

            last_cmd = getattr(AgentMode, cur_node.mode['car2'][0]).value  # cur_mode.mode[.] is some string 
            for tau_idx in range(tau_idx_min_int, tau_idx_max_int+1):
                lirpa_model = BoundedModule(models[last_cmd-1][tau_idx], (torch.empty_like(x))) 
                # lirpa_model = BoundedModule(model, (torch.empty_like(x))) 
                ptb_x = PerturbationLpNorm(norm = norm, x_L=x_l, x_U=x_u)
                bounded_x = BoundedTensor(x, ptb=ptb_x)
                lb, ub = lirpa_model.compute_bounds(bounded_x, method='alpha-CROWN')

                # new_mode = np.argmax(ub.numpy())+1 # will eventually be a list/need to check upper and lower bounds
                new_mode = np.argmin(lb.numpy())+1 # will eventually be a list/need to check upper and lower bounds
                
                new_modes = []
                for i in range(len(ub.numpy()[0])):
                    # upper = ub.numpy()[0][i]
                    # if upper>=lb.numpy()[0][new_mode-1]:
                    #     new_modes.append(i+1)
                    lower = lb.numpy()[0][i]
                    if lower<=ub.numpy()[0][new_mode-1]:
                        new_modes.append(i+1)
                int_modes.update(new_modes)
        # print(modes, cur_node.start_time) # at 15 s, all modes possible -- investigate why
        
        # iterate over all elements in the cross product of modes and int_modes instead in the future instead of this double loop
        for new_int_m in int_modes: 
            for new_m in modes:
                scenario.set_init(
                    [[own_state[0][1:], own_state[1][1:]], [int_state[0][1:], int_state[1][1:]]], # this should eventually be a range 
                    [(AgentMode(new_m), ),(AgentMode(new_int_m), )]
                )
                id += 1
                # new_trace = scenario.simulate(Tv, ts)
                new_trace = scenario.verify(Tv, ts)
                temp_root = new_trace.root
                new_node = cur_node.new_child(temp_root.init, temp_root.mode, temp_root.trace, cur_node.start_time + Tv, id)
                cur_node.child.append(new_node)
                print(f'Start time: {new_node.start_time}\nNode ID: {id}\nNew modes: {(AgentMode(new_m), AgentMode(new_int_m))}')
                    
                if new_node.start_time + Tv>=T: # if the time of the current simulation + start_time is at or above total time, don't add
                    continue
                queue.append(new_node)

    trace.nodes = trace._get_all_nodes(trace.root)
    print(f'Verification time: {time.perf_counter()-start}')

    fig = go.Figure()
    # fig = simulation_tree(trace, None, fig, 1, 2, [1, 2], "fill", "trace")
    # fig = reachtube_tree(trace, None, fig, 1, 2, [1, 2], "fill", "trace")
    fig = reachtube_tree_3d(trace, fig,1,'x', 2,'y',3,'z')
    fig.show()