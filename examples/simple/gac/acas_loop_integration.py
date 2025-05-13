from typing import List, Tuple

import numpy as np
import torch
import verse
from numba import njit
from scipy.integrate import ode
# from tutorial_utils import drone_params
from verse import BaseAgent, LaneMap
from verse.analysis.analysis_tree import TraceType
from verse.utils.utils import wrap_to_pi
from verse.map.lane_map_3d import LaneMap_3d
# from tutorial_sensor import DefaultSensor

from aircraft_agent import AircraftAgent
from aircraft_agent_intruder import AircraftAgent_Int
# from tutorial_map import M4
from verse.scenario import Scenario, ScenarioConfig

import warnings

import pyvista as pv
from verse.plotter.plotter3D import *
from verse.plotter.plotter3D_new import *

import plotly.graph_objects as go
from verse.plotter.plotter2D import *

from verse import Scenario
from verse.plotter import *
import numpy as np 
import plotly.graph_objects as go
from enum import Enum, auto

import matplotlib.pyplot as plt
from verse.plotter.plotter2D import reachtube_tree, simulation_tree
# from verse.plotter.plotter2D_old import plot_reachtube_tree, plot_simulation_tree, get_trace_data
from verse.plotter.plotter2D_old import plot_reachtube_tree, plot_simulation_tree
# plot_relative_distance
import os

import functools as ft

import ipdb
import jax
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import tqdm # for progress bar
from jax_guam.functional.guam_new import FuncGUAM, GuamState
from jax_guam.utils.jax_utils import jax2np, jax_use_cpu, jax_use_double
from jax_guam.utils.logging import set_logger_format
from loguru import logger

from GUAM_sensor import GUAMSensor
from dl_acas import AgentMode

#from tree_reader import tree_reader

import time

from pathlib import Path
from guam_sim import guam_to_dubins_2d, get_final_states_sim
from jax_guam.subsystems.genctrl_inputs.genctrl_circle_inputs import QrotZ, quaternion_to_euler, euler_to_quaternion

def dubins_to_guam_3d(state: List) -> List:
    v = state[-1]
    theta = np.pi/2-state[3]
    psi = state[4]
    # quat = QrotZ(theta)
    quat = euler_to_quaternion(0,psi,theta)
    x,y,z = state[0], state[1], state[2]
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, v, 0, 0, 0.0, 0.0, 0.0, y, x, -z, float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]), 0.0, 0.0, -0.000780906088785921, -0.000780906088785921, 0.0, 0.000, -1.0]
    #      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, v, vy,vz, 0 , 0.0, 0.0, x, y, z,  float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]), 0.0, 0.0, -0.000780906088785921, -0.000780906088785921, 0.0, 0.000, 1.0]

x_folder = "{}{}{}".format('batch_figures_good', os.sep, "x_pos")
y_folder = "{}{}{}".format('batch_figures_good', os.sep, "y_pos")
xy_folder = "{}{}{}".format('batch_figures_good', os.sep, "xy_pos")
rho_folder = "{}{}{}".format('batch_figures_good', os.sep, "rho")
Path(x_folder).mkdir(parents=True, exist_ok=True)
Path(y_folder).mkdir(parents=True, exist_ok=True)
Path(xy_folder).mkdir(parents=True, exist_ok=True)
Path(rho_folder).mkdir(parents=True, exist_ok=True)

start_time = time.time()
ego_mode = AgentMode.COC
int_mode = AgentMode.COC

# decisions_ego = np.load("test1.npy")
# decisions_ego = decisions_ego.tolist()


# x_init_ego = np.arange(start = -200, stop = 200, step = 200)
# y_init_ego = np.arange(start = -800, stop = 800, step = 800)
# x_vel_ego = np.arange(start = -100, stop = 100.1, step = 100)

# Most recent
x_init_ego = np.arange(start = -1000, stop = 1000.1, step = 500)
y_init_ego = np.arange(start = -2000, stop = 2000.1, step = 500)
# x_vel_ego = np.arange(start = -100, stop = 100.1, step = 50)
x_vel_ego = np.array([100])

# x_init_ego = np.array([-1000,0])
# y_init_ego = np.array([-8000, -4000])
# x_vel_ego = np.array([0, 50])

min_dist_list = []

for x_init_val in x_init_ego:
    for y_init_val in y_init_ego:
        for x_init_vel in x_vel_ego:
            file_string = f'x_{x_init_val:05}_y_{y_init_val:05}_vx_{x_init_vel:04}'
            print(file_string)


            scenario = Scenario(ScenarioConfig(parallel=False))
            scenario.set_sensor(GUAMSensor())
            # scenario.set_map(M4())

            #simulation_step = 0.1, 
            #nn_step = 1.0

            script_dir = os.path.realpath(os.path.dirname(__file__))
            input_code_name = os.path.join(script_dir, "dl_acas.py")
            # ac1 = AircraftAgent("aircraft1", file_name="dl_acas.py", initial_mode=ego_mode)
            ac1 = AircraftAgent("car1", file_name=input_code_name, initial_mode=ego_mode)

            # set of the initial states:
            # 1. make sure x[12] is identical for ac1 and ac2 (i.e., co-altitude);
            # 2. make sure the quaternion (x[15:19]) is set as [1. 0, 0, 0] (if no pitch, roll, yaw)


            # ac1.set_initial(
            #     [
            #         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 0, 0.0, 0.0, 0.0, 10, 0, -10, 1.0, 0.0, -4.3136e-05, 0., 0.0, 0.0, -0.000780906088785921, -0.000780906088785921, 0.0, 0.0],
            #         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 0, 0.0, 0.0, 0.0, 10, 0, -10, 1.0, 0.0, -4.3136e-05, 0., 0.0, 0.0, -0.000780906088785921, -0.000780906088785921, 0.0, 0.0]
            #     ],
            #     ([AgentMode.Coc]),
            # )
            # ac1.set_initial(
            #     [
            #         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0, 0.0, 0.0, 0.0, 0, 0, -10, 1.0, 0.0, -4.3136e-05, 0., 0.0, 0.0, -0.000780906088785921, -0.000780906088785921, 0.0, 1.0],
            #         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0, 0.0, 0.0, 0.0, 0, 0, -10, 1.0, 0.0, -4.3136e-05, 0., 0.0, 0.0, -0.000780906088785921, -0.000780906088785921, 0.0, 1.0]
            #     ],
            #     ([AgentMode.Coc]),
            # ) # scenario in slides -- working example 1 of ACAS Xu advisory

            # Add time here?

            ######
            # Init states (11/29)
            ######
            own_0_vx = x_init_vel
            own_0_vy = np.sqrt(100**2 - x_init_vel**2)
            own_0_vz = 0.0

            own_0_x = x_init_val
            own_0_y = y_init_val
            own_0_z = 0
            quat = euler_to_quaternion(0, np.pi/12, 0)
            # quat = euler_to_quaternion(0, 0, 0)
            ac1.set_initial(
    
                    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, own_0_vx, own_0_vy, own_0_vz, 0.0, 0.0, 0.0, own_0_x-10, own_0_y-10, own_0_z, 1.0, 0.0, -4.3136e-05, 0., 0.0, 0.0, -0.000780906088785921, -0.000780906088785921, 0.0, 0.000, -25.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, own_0_vx, own_0_vy, own_0_vz, 0.0, 0.0, 0.0, own_0_x+10, own_0_y+10, own_0_z+1, 1.0, 0.0, -4.3136e-05, 0., 0.0, 0.0, -0.000780906088785921, -0.000780906088785921, 0.0, 0.000, -25.0]],
                    # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, own_0_vx, own_0_vy, own_0_vz, 0.0, 0.0, 0.0, own_0_x, own_0_y, own_0_z, 0.707, 0.0, 0, 0.707, 0.0, 0.0, -0.000780906088785921, -0.000780906088785921, 0.0, 0.000, 1.0],
                    # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, own_0_vx, own_0_vy, own_0_vz, 0.0, 0.0, 0.0, own_0_x, own_0_y, own_0_z, 0.707, 0.0, 0, 0.707, 0.0, 0.0, -0.000780906088785921, -0.000780906088785921, 0.0, 0.000, 1.0]
                    # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, own_0_vx, own_0_vy, own_0_vz, 0.0, 0.0, 0.0, own_0_x, own_0_y, own_0_z,float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]), 0.0, 0.0, -0.000780906088785921, -0.000780906088785921, 0.0, 0.000, 1.0],
                    # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, own_0_vx, own_0_vy, own_0_vz, 0.0, 0.0, 0.0, own_0_x, own_0_y, own_0_z, float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]), 0.0, 0.0, -0.000780906088785921, -0.000780906088785921, 0.0, 0.000, 1.0]           
                    # [dubins_to_guam_3d([-2, -1, 0, 0, 0, 100]), dubins_to_guam_3d([-1,1, 0, 0, 0, 100])],
                    # [dubins_to_guam_3d([-2, -1000, 0, np.pi/3, np.pi/12, 100]), dubins_to_guam_3d([-1,-999, 0, np.pi/3, np.pi/12, 100])],
                # ([AgentMode.SR]),
                ([AgentMode.COC]),
            ) # scenario in slides -- working example 2 of ACAS Xu advisory


            # decisions_int = [0] * 96
            script_dir = os.path.realpath(os.path.dirname(__file__))
            input_code_name = os.path.join(script_dir, "dl_acas_intruder.py")
            # ac2 = AircraftAgent_Int("car2", file_name=input_code_name, initial_mode=int_mode)
            ac2 = AircraftAgent_Int("car2")

            # ac2.set_initial(
            #     [
            #         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.8, 3.3, 0, 0.0, 0.0, 0.0, 0, 0, -10, 1.0, 0.0, -4.3136e-05, 0., 0.0, 0.0, -0.000780906088785921, -0.000780906088785921, 0.0, 0.0],
            #         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.8, 3.3, 0, 0.0, 0.0, 0.0, 0, 0, -10, 1.0, 0.0, -4.3136e-05, 0., 0.0, 0.0, -0.000780906088785921, -0.000780906088785921, 0.0, 0.0]
            #     ],
            #     ([AgentMode.Coc]),
            # )
            # The `ac2.set_initial()` function is setting the initial state for the second aircraft agent (`ac2`)
            # in a simulation scenario. The function call specifies the initial state for the agent by providing a
            # list of initial state values for two different time steps. Each time step has a corresponding list
            # of values representing the state of the agent at that time.
            # ac2.set_initial(
            #     [
            #         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5423385832990382, 1.27326026187387, 0, 0.0, 0.0, 0.0, 92.46419421, 157.2976814, -10, 1.0, 0.0, -4.3136e-05, 0., 0.0, 0.0, -0.000780906088785921, -0.000780906088785921, 0.0, -1.0],
            #         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5423385832990382, 1.27326026187387, 0, 0.0, 0.0, 0.0, 92.46419421, 157.2976814, -10, 1.0, 0.0, -4.3136e-05, 0., 0.0, 0.0, -0.000780906088785921, -0.000780906088785921, 0.0, -1.0]
            #     ],
            #     ([AgentMode.Coc]),
            # ) # scenario in slides -- working example 1 of ACAS Xu advisory
            '''
            ac2.set_initial(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 200.0, -10, 1.0, 0.0, -4.3136e-05, 0., 0.0, 0.0, -0.000780906088785921, -0.000780906088785921, 0.0,0.000, -1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 200.0, -10, 1.0, 0.0, -4.3136e-05, 0., 0.0, 0.0, -0.000780906088785921, -0.000780906088785921, 0.0,0.000, -1.0]
                ],
                ([AgentMode.Coc]),
            ) # scenario in slides -- working example 2 of ACAS Xu advisory
            '''
            ######
            # Init states (11/29)
            ######
            int_0_vx = 100.0 #1.532
            int_0_vy = 0.0 #1.28
            int_0_vz = 0.0

            int_0_x = -2000
            int_0_y = 0
            int_0_z = -1000
            quat2 = euler_to_quaternion(0, 0, 0)
            ac2.set_initial(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, int_0_vx, int_0_vy, int_0_vz, 0.0, 0.0, 0.0, int_0_x, int_0_y-10, int_0_z, 1.0, 0.0, -4.3136e-05, 0., 0.0, 0.0, -0.000780906088785921, -0.000780906088785921, 0.0, 0.000, 0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, int_0_vx, int_0_vy, int_0_vz, 0.0, 0.0, 0.0, int_0_x+10, int_0_y+10, int_0_z+10, 1.0, 0.0, -4.3136e-05, 0., 0.0, 0.0, -0.000780906088785921, -0.000780906088785921, 0.0, 0.000, 0]
                    # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, int_0_vx, int_0_vy, int_0_vz, 0.0, 0.0, 0.0,int_0_y, int_0_x, int_0_z, float(quat2[0]), float(quat2[1]), float(quat2[2]), float(quat2[3]), 0.0, 0.0, -0.000780906088785921, -0.000780906088785921, 0.0, 0.000, -1.0],
                    # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, int_0_vx, int_0_vy, int_0_vz, 0.0, 0.0, 0.0,int_0_y, int_0_x, int_0_z, float(quat2[0]), float(quat2[1]), float(quat2[2]), float(quat2[3]), 0.0, 0.0, -0.000780906088785921, -0.000780906088785921, 0.0, 0.000, -1.0]
                    # dubins_to_guam_3d([-2001, -1, 999, 0,0, 100]), dubins_to_guam_3d([-1999, 1, 1001, 0,0, 100])
                ],
                ([AgentMode.COC]),
            ) # scenario in slides -- working example 2 of ACAS Xu advisory

            scenario.add_agent(ac1)
            scenario.add_agent(ac2)

            # scenario.set_sensor(DefaultSensor())

            # traces_simu = scenario.simulate(80, 0.2)
            start = time.time()
            # traces_simu = scenario.simulate(30, 0.1) # NOTE:  DO NOT SET FINAL TIME TO ANY MULTIPLE OF YOUR SAMPLE TIME (DELETES AN EGO STATE AT SAMPLE TIME AND YOU GET SHAPE ERRORS)
            traces_simu = scenario.verify(20, 0.1) # NOTE:  DO NOT SET FINAL TIME TO ANY MULTIPLE OF YOUR SAMPLE TIME (DELETES AN EGO STATE AT SAMPLE TIME AND YOU GET SHAPE ERRORS)

            # own_state, int_state = get_final_states_sim(traces_simu.root)
            # dub_own_state, dub_int_state = guam_to_dubins_2d(own_state[1:]), guam_to_dubins_2d(int_state[1:])
            # print(dub_own_state, dub_int_state)
            #traces_veri = scenario.verify(80, 10)#0.2)

            warnings.filterwarnings("ignore")

            fig = go.Figure()
            fig.update_layout(xaxis_title='E [ft]', yaxis_title='N [ft]')

            # fig = simulation_tree(traces_simu, None, fig, 14, 13)  # red
            # fig = simulation_tree_3d(traces_simu, fig,14,'x', 13,'y',15,'z')
            fig = reachtube_tree_3d(traces_simu, fig,14,'x', 13,'y',15,'z')

            # fig = reachtube_tree(traces_veri, None, fig, 13, 14, plot_color= [['#0000CC', '#0000FF', '#3333FF', '#6666FF', '#9999FF', '#CCCCFF'], # blue
            #                                                                   ['#CC0000', '#FF0000', '#FF3333', '#FF6666', '#FF9999', '#FFCCCC']])  # red
            # fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Ego'))  # Add a dummy trace for the legend
            # fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Intruder'))  # Add a dummy trace for the legend
            # fig.update_xaxes(range = [-200, 200])
            fig.show()

            print(start-time.time())
            exit()
            #fig1, ax1 = plt.subplots()
            #fig2, ax2 = plt.subplots()
            #fig3, ax3 = plt.subplots()

            # Calculate rho
            # print(len(traces_simu.nodes))

            
                

            # ego_sim_node = traces_simu.nodes[1]
            # ego_sim_trace = ego_sim_node.trace
            # ego_sim_trace = ego_sim_trace['aircraft1']

            # int_sim_node = traces_simu.nodes[1]
            # int_sim_trace = int_sim_node.trace
            # int_sim_trace = int_sim_trace['aircraft2']

            # #print(ego_sim_trace)
            # print(len(int_sim_trace))
            # print(len(int_sim_trace[0]))
            # time_steps = int_sim_trace[0:4][0]
            # print(time_steps)
            time_steps = []
            relative_distance = []
            ego_x = []
            ego_y = []
            ego_z = []
            int_x = []
            int_y = []
            int_z = []

            # Iterate through each node
            for node_ind in range(0, len(traces_simu.nodes)):
                sim_node = traces_simu.nodes[node_ind]
                sim_trace = sim_node.trace
                
                ego_trace = sim_trace['aircraft1']
                int_trace = sim_trace['aircraft2']
                
                #Iterate through each time step in a single node
                for i in range(0, len(int_trace)):
                    time_steps.append(int_trace[i][0])
                    # time_steps = int_sim_trace[:][0]
                    ego_x.append(ego_trace[i][13])
                    ego_y.append(ego_trace[i][14])
                    ego_z.append(ego_trace[i][15])
                    int_x.append(int_trace[i][13])
                    int_y.append(int_trace[i][14])
                    int_z.append(int_trace[i][15])
                    
                    rel_x = ego_trace[i][13] - int_trace[i][13]
                    rel_y = ego_trace[i][14] - int_trace[i][14]


                    relative_distance.append(np.sqrt(rel_x**2 + rel_y**2))


            print(f"Ego init x:  {ego_trace[0][13]}")
            print(f"INt init x:  {int_trace[0][13]}")
            
            print(f"Ego init y:  {ego_trace[0][14]}")
            print(f"INt init y:  {int_trace[0][14]}")
            
            min_dist_list.append(min(relative_distance))

            plt.figure()
            plt.plot(time_steps, ego_x, label = 'Ego')
            plt.plot(time_steps, int_x, label = 'Intruder')
            plt.xlabel('Time (s)')
            plt.ylabel('x (ft)')
            plt.legend()
            plt.title('X-Position of Ego and Intruder Vehicles')
            plt.savefig(x_folder + os.sep + file_string + '.png', format = 'png')

            plt.figure()
            plt.plot(time_steps, ego_y, label = 'Ego')
            plt.plot(time_steps, int_y, label = 'Intruder')
            plt.xlabel('Time (s)')
            plt.ylabel('y (ft)')
            plt.legend()
            plt.title('Y-Position of Ego and Intruder Vehicles')
            plt.savefig(y_folder + os.sep + file_string + '.png', format = 'png')

            # plt.figure()
            # plt.plot(time_steps, ego_z, label = 'Ego')
            # plt.plot(time_steps, int_z, label = 'Intruder')
            # plt.xlabel('Time (s)')
            # plt.ylabel('z (ft)')
            # plt.legend()
            # plt.title('Z-Position of Ego and Intruder Vehicles')
            # plt.savefig(z_folder + os.sep + file_string + '.png', format = 'png')

            plt.figure()
            plt.plot(ego_x, ego_y, label = 'Ego')
            plt.plot(int_x, int_y, label = 'Intruder')
            plt.xlabel('x (ft)')
            plt.ylabel('y (ft)')
            plt.legend()
            plt.title('XY-Position of Ego and Intruder Vehicles')
            plt.savefig(xy_folder + os.sep + file_string + '.png', format = 'png')

            # plt.figure()
            # plt.plot(ego_x, ego_y, label = 'Ego', alpha = np.array([x / max(time_steps) for x in time_steps]))
            # plt.plot(int_x, int_y, label = 'Intruder', alpha = np.array([x / max(time_steps) for x in time_steps]))
            # plt.xlabel('x (ft)')
            # plt.ylabel('y (ft)')
            # plt.legend()
            # plt.title('XY-Position of Ego and Intruder Vehicles')


            plt.figure()
            plt.plot(time_steps, relative_distance, color='b')
            plt.xlabel('Time (s)')
            plt.ylabel('Relative Distance (ft)')
            plt.title('Relative Distance Between Ego and Intruder Vehicles')
            plt.savefig(rho_folder + os.sep + file_string + '.png', format = 'png')
            #plt.show()



            # `fig4 = go.Figure()` is creating a new figure object using Plotly's graph_objects module. This
            # `go.Figure()` function initializes a new figure that can be used to create interactive plots
            # using Plotly. This figure object can then be used to add traces, annotations, layout settings,
            # and more to create visualizations in Plotly.
            # fig4 = go.Figure()


            ###########
            ## UNCOMMENT THIS WHEN YOU VERIFY
            ###########
            # ax1.set_xlabel('t [sec]')
            # ax1.set_ylabel('x [m]')
            # fig1 = plot_reachtube_tree(traces_veri.root, 'aircraft2', 0, [13], color='b', fig=fig1)
            # ax1.plot([], [], 'r', label='Ego')  # Add an empty plot for the legend entry
            # fig1 = plot_reachtube_tree(traces_veri.root, 'aircraft1', 0, [13], color='r', fig=fig1)
            # ax1.plot([], [], 'b', label='Intruder')  # Add an empty plot for the legend entry
            # ax1.legend()
            ###########
            ## END:  UNCOMMENT THIS WHEN YOU VERIFY
            ###########


            # plt.xlabel('t [sec]')
            # plt.ylabel('y [m]')
            # fig2 = plot_reachtube_tree(traces_veri.root, 'aircraft2', 0, [14], color='r', fig=fig2)
            # fig2 = plot_reachtube_tree(traces_veri.root, 'aircraft1', 0, [14], color='b', fig=fig2)
            # fig2.show()

            # plt.xlabel('t [sec]')
            # plt.ylabel('z [m]')
            # fig3 = plot_reachtube_tree(traces_veri.root, 'aircraft2', 0, [15], color='r', fig=fig3)
            # fig3 = plot_reachtube_tree(traces_veri.root, 'aircraft1', 0, [15], color='b', fig=fig3)
            # fig3.show()

            # # Extract trace data for both agents
            # trace_agent1 = get_trace_data(traces_veri.root, agent_id='aircraft1', x_dim=0, y_dim_list=[13])
            # trace_agent2 = get_trace_data(traces_veri.root, agent_id='aircraft2', x_dim=0, y_dim_list=[13])
            #trace_agent1 = get_trace_data(traces_simu.root, agent_id='aircraft1', x_dim=0, y_dim_list=[13])
            #trace_agent2 = get_trace_data(traces_simu.root, agent_id='aircraft2', x_dim=0, y_dim_list=[13])

            # Compute the relative distance
            # time_steps = trace_agent1[:, 0]
            # relative_distance = np.sqrt((trace_agent2[:, 1] - trace_agent1[:, 1])**2 + 
            #                             (trace_agent2[:, 2] - trace_agent1[:, 2])**2 +
            #                             (trace_agent2[:, 3] - trace_agent1[:, 3])**2)
            # print("---data logging----")
            # print(trace_agent1)
            # print(trace_agent2)
            # Plot the relative distance over time
            # plt.figure()
            # plt.plot(time_steps, relative_distance, color='b')
            # plt.xlabel('Time')
            # plt.ylabel('Relative Distance')
            # plt.title('Relative Distance Between Aircraft 1 and 2')
            # plt.show()
            # plt.xlabel('x [m]')
            # plt.ylabel('y [m]')
            # fig4 = reachtube_tree(traces_veri, None, fig4, 13, 14,
            #                             print_dim_list=[1,2])
            # fig4.show()


            ''' UNCOMMENT ONCE VERIFY IS WORKING
            fig = plot_relative_distance(traces_veri.root, 'aircraft1', 'aircraft2', dims=[13, 14, 15], color='b')
            plt.tight_layout()
            plt.show() 
            '''

            # plt.tight_layout()
            # plt.show()
        
min_dist_list_np = np.array(min_dist_list)
np.save('min_dist.npy', min_dist_list_np) # save
print(min_dist_list_np)

end_time = time.time()
print(f"Process finished in {end_time - start_time} seconds")
