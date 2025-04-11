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
# from auto_LiRPA import BoundedTensor
from verse.utils.utils import wrap_to_pi
import pyvista as pv
import numpy as np
from dubin_sensor import DubinSensor

class AgentMode(Enum):
    COC = auto()
    WL = auto()
    WR = auto()
    SL = auto()
    SR = auto()

def get_acas_state(own_state: List[float], int_state: List[float]) -> torch.Tensor:
    dist = np.sqrt((own_state[0]-int_state[0])**2+(own_state[1]-int_state[1])**2)
    theta = wrap_to_pi((2*np.pi-own_state[2])+np.arctan2(int_state[0], int_state[1]))
    psi = wrap_to_pi(int_state[2]-own_state[2])
    return torch.tensor([dist, theta, own_state[3], int_state[3]])


if __name__ == "__main__":
    import os
    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "controller_v2.py")
    car = CarAgent('car1', file_name=input_code_name)
    car2 = NPCAgent('car2')
    scenario = Scenario(ScenarioConfig(parallel=False))
    scenario.set_sensor(DubinSensor())
    # scenario.config.reachability_method = ReachabilityMethod.DRYVR_DISC
    car.set_initial(
        initial_state=[[-100, -1100, np.pi/3, 100, 0, 0], [100, -900, np.pi/3, 100, 0, 0]],
        # initial_state=[[0, -1010, np.pi/3, 100, 0, 0], [0, -990, np.pi/3, 100, 0, 0]],
        initial_mode=(AgentMode.COC, )
    )
    car2.set_initial(
            initial_state=[[-2000, 0, 0, 100, 0, 0], [-2000, 0, 0, 100, 0, 0]],
        initial_mode=(AgentMode.COC, )
    )
    scenario.add_agent(car)
    scenario.add_agent(car2)
    # trace = scenario.simulate(20, 1)
    fig = go.Figure()
    # traces = []
    trace = scenario.verify(20, 1) # increasing ts to 0.1 to increase learning speed, do the same for dryvr2
    N = 1
    # trace = scenario.simulate(20,0.01)
    # fig = simulation_tree(trace, None, fig, 1, 2, [1, 2], "fill", "trace")
    # for i in range(N):
        # traces.append(scenario.simulate(20,0.01))
        # fig = simulation_tree(traces[-1], None, fig, 1, 2, [1, 2], "fill", "trace")
    fig = reachtube_tree(trace, None, fig, 1, 2, [1, 2], "fill", "trace")
    fig.show() 