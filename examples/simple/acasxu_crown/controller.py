from enum import Enum, auto
import copy
from typing import List


class LaneObjectMode(Enum):
    Vehicle = auto()
    Ped = auto()  # Pedestrians
    Sign = auto()  # Signs, stop signs, merge, yield etc.
    Signal = auto()  # Traffic lights
    Obstacle = auto()  # Static (to road/lane) obstacles


class AgentMode(Enum):
    COC = auto()
    WL = auto()
    WR = auto()
    SL = auto()
    SR = auto()

class State:
    x: float
    y: float
    theta: float
    v: float
    agent_mode: AgentMode

    def __init__(self, x, y, theta, v, agent_mode: AgentMode):
        pass


def decisionLogic(ego: State, others: List[State], track_map):
    output = copy.deepcopy(ego)
    # assert not vehicle_close(ego, others)
    return output

