"""comparable to the guam controller file"""


from enum import Enum, auto
import copy
import random
from typing import List, NamedTuple
import functools as ft
import logging

import jax
import tqdm
from jax_guam.functional.aero_prop_new import FuncAeroProp
from jax_guam.functional.lc_control import LCControl, LCControlState
from jax_guam.functional.surf_engine import SurfEngine, SurfEngineState
from jax_guam.functional.vehicle_eom_simple import VehicleEOMSimple
from jax_guam.guam_types import AircraftState, AircraftStateVec, EnvData, PwrCmd, RefInputs
from jax_guam.subsystems.environment.environment import Environment
from jax_guam.subsystems.vehicle_model_ref.power_system import power_system
from jax_guam.utils.ode import ode3

# from acasxu_dubin_testing import *
# from acasxu_dubin_testing import load_network, run_network
from GUAM_sensor import GUAMSensor

# from tree_reader import *

# Configure logging at the start of your script.
# Set up logging to file
# logging.basicConfig(filename='myapp.log', level=logging.INFO)

class AgentMode(Enum):
    COC = auto()
    WL = auto()
    WR = auto()
    SL = auto()
    SR = auto()


# class TrackMode(Enum):
#     T0 = auto()
#     T1 = auto()
#     T2 = auto()
#     T3 = auto()
#     T4 = auto()
class State:
    rho: float
    theta: float
    psi: float
    v_own: float
    v_int: float
    
    d05: float
    d06: float
    d07: float
    d08: float
    d09: float
    d10: float  
    d11: float  
    d12: float  
    d13: float  
    d14: float  
    d15: float  
    d16: float  
    d17: float  
    d18: float  
    d19: float   
    d20: float   
    d21: float   
    d22: float   
    d23: float   
    d24: float    
    # want timer as 25 (maybe 24)
    timer: float
    
    agent_mode: AgentMode
    
###################
# OLD STATE CLASS
###################
# class State:
#     # init state: [x, y, theta, random_attr]
#     # x: float
#     # y: float
#     # theta: float
#     # random_attr: float
#     # velo: float
#     # mode: CraftMode
#    
#     x1: float
#     x2: float
#     x3: float
#     x4: float
#     x5: float
#     x6: float
    
#     x7: float
#     x8: float
#     x9: float
#     x10: float
#     x11: float
#     x12: float
#     x13: float
#     x14: float
#     x15: float
#     x16: float
#     x17: float
#     x18: float
#     x19: float
    
#     x20: float
#     x21: float
#     x22: float
#     x23: float
#     x24: float
    

#     random_attr: float
#     mode: CraftMode
#     timer: float
    

    # def __init_(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, random_attr, mode, timer):
    #     pass

    # def __init__(self, x, y, theta, random_attr, velo, mode):
    #     pass
#def test_fcn(ego, other):
#    return ego.x1 - other.x1
def decisionLogic(ego: State, others: List[State]):
    next = copy.deepcopy(ego)
    
    return next