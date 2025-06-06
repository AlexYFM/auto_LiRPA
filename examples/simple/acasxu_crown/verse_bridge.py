
import time
import os
import numpy as np
from enum import Enum,auto
from typing import Tuple, List
# Define whatever here

#=====================================================================================

from dubins_3d_agent import CarAgent, NPCAgent



from verse.map.example_map.map_tacas import M1
from verse import Scenario, ScenarioConfig

from verse.analysis.verifier import ReachabilityMethod
import torch
from auto_LiRPA import BoundedTensor
from verse.utils.utils import wrap_to_pi
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
    for own_vert in own_ext:
        for int_vert in int_ext:
            arho = np.arctan2(int_vert[1]-own_vert[1],int_vert[0]-own_vert[0]) % (2*np.pi)
            arho_max = max(arho_max, arho)
            arho_min = min(arho_min, arho)

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

def get_final_states_sim(n) -> Tuple[List]: 
    own_state = n.trace['car1'][-1]
    int_state = n.trace['car2'][-1]
    return own_state, int_state

def get_final_states_verify(n) -> Tuple[List]: 
    own_state = n.trace['car1'][-2:]
    int_states = [n.trace['car2'][-2:], n.trace['car3'][-2:]]
    return own_state, int_states

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

#=====================================================================================





class VerseBridge():

    def __init__(self,ax):
        self.agents = {}
        self.agent_dict = {"Car": CarAgent, "NPC": NPCAgent}

        self.mode_dict = {"Car": AgentMode.COC, "NPC": AgentMode.COC}
        self.plotter = ax

        #Can hardcode like this:

        self.updatePlane(id="car1", agent_type="Car", dl ="controller_3d.py")
        self.addInitialSet("car1", [[-1, -1001, -1, np.pi/3, np.pi/6, 100], [1, -999, 1, np.pi/3, np.pi/6, 100]])

        self.updatePlane(id="car2", agent_type="NPC" )
        self.addInitialSet("car2",[[-2001, 99, 999, 0,0, 100], [-1999, 101, 1001, 0,0, 100]])

        self.updatePlane(id="car3", agent_type="NPC" )
        self.addInitialSet("car3",[[1999, -1, 999, np.pi,0, 100], [2001, 1, 1001, np.pi,0, 100]])

    #uses input from from GUI
    def updatePlane(self, id="", verify=True, agent_type=None,  dl=None, x=0, y=0, z=0, radius=0, yaw=0, pitch=0, v=0   ):

        init_set = [[x - radius ,y -radius,z-radius,pitch, yaw, v  ],[x + radius ,y + radius,z+radius, pitch, yaw, v]]
        self.agents[id] = {"init_set":init_set, 
                           "init_mode": self.mode_dict[agent_type], 
                           "agent_type": self.agent_dict[agent_type],
                            "dl": dl  }
         

    # To set the initial set directly 
    def addInitialSet(self, id, initialSet= []):
        self.agents[id]["init_set"] = initialSet


    # removes plane
    def removePlane(self, id):
        self.agents.pop(id)



    def run_verse(self, ax= None, time_horizon=80, time_step=50,  x_dim=1, y_dim=2, z_dim=3, num_sims= 0):
        scenario = Scenario(ScenarioConfig( parallel=False))

        script_dir = os.path.realpath(os.path.dirname(__file__))

        for id, dict in self.agents.items():

            dl = dict["dl"]
            agent_type = dict["agent_type"]
            init_mode = dict["init_mode"]
            init_set = dict["init_set"]

            if(dl != None):
                input_code_name = os.path.join(script_dir, dl )
                plane = agent_type(id, file_name=input_code_name)
            else:
                plane = agent_type(id)

            scenario.add_agent(plane) 

            scenario.set_init_single(
                id, init_set, (init_mode,)
            )
        self.plotter.clear()

        self.plotter.show_grid()

        #normally just run verify once

        # if(num_sims ==0 ):
        #     scenario.verify(time_horizon, time_step, self.plotter)
        # else:
        #     for i in range(num_sims):
        #         scenario.simulate(time_horizon, time_step, self.plotter)

        # Define whatever here:

        #=====================================================================================================================

        T = 20

        #should define in plotter config instead
        Tv = 1
        ts = 0.01


        # observation: for Tv = 0.1 and a larger initial set of radius 10 in y dim, the number of 

        scenario.config.print_level = 0
        scenario.config.reachability_method = ReachabilityMethod.DRYVR_DISC
       
        start = time.perf_counter()

        # self.plotter.render_window.GetRenderers().GetFirstRenderer().SetInteractive(False)
        # self.plotter.interactor.SetInteractorStyle(None)
        #self.plotter.render_window.SetAbortRender(True)    

        trace = scenario.verify(Tv,ts, self.plotter) # this is the root
        id = 1+trace.root.id
        models = [[torch.load(f"./examples/simple/acasxu_crown/nets/ACASXU_run2a_{net + 1}_{tau + 1}_batch_2000.pth") for tau in range(9)] for net in range(5)]
        norm = float("inf")

        queue = deque()
        queue.append(trace.root) # queue should only contain ATNs  
        ### begin looping
        while len(queue):
            cur_node = queue.popleft() # equivalent to trace.nodes[0] in this case
            own_state, int_states = get_final_states_verify(cur_node)
            # in general, for i in range(num_intruders)
            tau_idx_min, tau_idx_max = [get_tau_idx(own_state[1], int_states[i][0]) for i in range(2)], [get_tau_idx(own_state[0], int_states[i][1]) for i in range(2)] 
            # print(tau_idx_min, tau_idx_max)
            modes = set()
            reachsets = [get_acas_reach(np.array(own_state)[:,1:], np.array(int_states[0])[:,1:]), get_acas_reach(np.array(own_state)[:,1:], np.array(int_states[1])[:,1:])]
            # print(reachsets)
            closer_idx = []
            if reachsets[0][0][1][0]<reachsets[1][0][0][0]: # if int 1 always closer than int 2
                closer_idx = [0]
            elif reachsets[1][0][1][0]<reachsets[0][0][0][0]: # if int 2 always closer than int 1
                closer_idx = [1]
            else: # in general, find intruder with min dist, use for loop to iterate over all intruders, and add intruders with min dist<max(min dist) in order 
                closer_idx = [0,1]
                print('_______\nChecking reachsets of both agents\n________')
                print([reachsets[0][0][i][0] for i in range(2)], [reachsets[1][0][i][0] for i in range(2)])
            for idx in closer_idx: # iterate over closest intruder(s) 
                for reachset in reachsets[idx]: # iterate over reachsets
                    if len(modes)==5: # if all modes are possible, stop iterating
                        break 
                    acas_min, acas_max = reachset
                    acas_min, acas_max = (acas_min-means_for_scaling)/range_for_scaling, (acas_max-means_for_scaling)/range_for_scaling
                    x_l, x_u = torch.tensor(acas_min).float().view(1,5), torch.tensor(acas_max).float().view(1,5)
                    x = (x_l+x_u)/2

                    last_cmd = getattr(AgentMode, cur_node.mode['car1'][0]).value  # cur_mode.mode[.] is some string 
                    for tau_idx in range(tau_idx_min[idx], tau_idx_max[idx]+1):
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
                        modes.update(new_modes)
            
            # print(modes, cur_node.start_time) # at 15 s, all modes possible -- investigate why
            for new_m in modes:
                scenario.set_init(
                    [[own_state[0][1:], own_state[1][1:]], 
                    [int_states[0][0][1:], int_states[0][1][1:]],
                    [int_states[1][0][1:], int_states[1][1][1:]]
                    ], # this should eventually be a range 
                    [(AgentMode(new_m),   ),(AgentMode.COC,   ),(AgentMode.COC,   )]
                )
                id += 1
                # new_trace = scenario.simulate(Tv, ts)
                
                start_ver = time.perf_counter() 
                new_trace = scenario.verify(Tv, ts, self.plotter)
                
                print(f'Verification time: {time.perf_counter()-start_ver:.2f} s')

                temp_root = new_trace.root
                new_node = cur_node.new_child(temp_root.init, temp_root.mode, temp_root.trace, cur_node.start_time + Tv, id)
                cur_node.child.append(new_node)
                print(f'Start time: {new_node.start_time}\nNode ID: {id}\nNew mode: {AgentMode(new_m)}')
                    
                if new_node.start_time + Tv>=T: # if the time of the current simulation + start_time is at or above total time, don't add
                    continue
                queue.append(new_node)

        trace.nodes = trace._get_all_nodes(trace.root)
        print(f'Verification time: {time.perf_counter()-start}')


       
        

    

