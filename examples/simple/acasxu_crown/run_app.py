from wsgiref.validate import PartialIteratorWrapper
from dubins_3d_agent import CarAgent, NPCAgent
from verse import Scenario, ScenarioConfig

from verse.plotter.plotter2D import *
from verse.plotter.plotter3D import *
from verse.plotter.plotter3D_new import *


from verse.map.example_map.map_tacas import M1

from verse.analysis.verifier import ReachabilityMethod
import torch
from auto_LiRPA import BoundedTensor
from verse.utils.utils import wrap_to_pi
from collections import deque
from torch import nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

import time
import sys
import numpy as np
import os
import threading
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, 
    QSizePolicy, QLabel,QTextEdit, QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox
)

from PyQt6.QtCore import Qt
from PyQt6.QtWebEngineWidgets import QWebEngineView
from ui_components import StyledButton, SvgPlaneSlider, OverlayTab, RightOverlayTab,RightInfoPanel


from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
import pyvistaqt as pvqt
import json

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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plane Visualization")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize variables
        self.plane1_position = 0
        self.plane2_position = 0
        self.thread = None
        self.overlay_visible = True
        self.right_overlay_visible = True  # Add this line
        
        # Setup main UI
        self.setup_main_ui()
        
        # Setup left overlay
        self.setup_overlay()
        
        # Setup right overlay
        self.setup_right_overlay()  # Add this line
        
        # Setup web view
        self.setup_web_view()
        
        # Setup side tabs
        self.setup_side_tab()
        self.setup_right_tab()  # Add this line
        
        # Make sure overlays are visible
        self.overlay_container.raise_()
        self.overlay_container.show()
        
        self.right_overlay_container.raise_()  # Add this line
        self.right_overlay_container.show()  # Add this line
        
        self.side_tab.raise_()
        self.right_side_tab.raise_()  # Add this line


    def setup_main_ui(self):
        """Setup the main UI components"""
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = pvqt.QtInteractor()
        self.plotter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.main_layout.addWidget(self.plotter.interactor)

        # cube = pv.Cube(center=(0, 0, 0), x_length=1, y_length=1, z_length=1)
        # self.plotter.add_mesh(cube, show_edges=True, color='lightgreen', opacity=0.5)
        

    # In setup_overlay method - Add text field for file loading
    def setup_overlay(self):
        """Setup the overlay container and its components"""
        self.overlay_container = QWidget(self.main_widget)
        self.overlay_container.setGeometry(0, 0, 480, 500)
        
        # Create info panel
        # self.info_panel = InfoPanel(self.overlay_container)
        # self.info_panel.setGeometry(10, 10, 350, 350)

        # Add text field for file loading
        self.file_label = QLabel("Load From File:", self.overlay_container)
        self.file_label.setGeometry(10, 450, 100, 25)
        self.file_label.setStyleSheet("color: white; font-weight: bold;")
        
        self.file_input = QLineEdit(self.overlay_container)
        self.file_input.setGeometry(110, 450, 240, 25)
        self.file_input.setStyleSheet("""
            QLineEdit {
                background-color: white;
                border: 1px solid #3498db;
                border-radius: 4px;
                padding: 2px 4px;
            }
            QLineEdit:focus {
                border: 2px solid #2980b9;
            }
        """)
        
        # Setup sliders
        self.setup_sliders()
        
        # Setup buttons
        self.setup_buttons()

    def setup_side_tab(self):
        """Setup the side tab for showing/hiding overlay"""
        self.side_tab = OverlayTab(self.main_widget)
        self.side_tab.overlay_visible = self.overlay_visible
        # Instead of assigning to a 'clicked' attribute, set a callback function
        self.side_tab.on_click_callback = self.toggle_overlay
        self.update_side_tab_position()
    def toggle_right_overlay(self):
        """Toggle visibility of the right overlay"""
        self.right_overlay_visible = not self.right_overlay_visible
        
        # Delete and recreate the tab with the new position
        if hasattr(self, 'right_side_tab'):
            self.right_side_tab.deleteLater()
        
        # Create new tab in the correct position
        self.setup_right_tab()
    
        # Show/hide overlay as needed
        if self.right_overlay_visible:
            # Set position relative to the current window width
            overlay_width = 380
            self.right_overlay_container.setGeometry(self.width() - overlay_width, 0, overlay_width, 300)
            self.right_overlay_container.show()
            self.right_overlay_container.raise_()
        else:
            self.right_overlay_container.hide()
    
        self.right_side_tab.show()

    def update_right_tab_position(self):
        """Update the position of the right side tab based on overlay visibility"""
        tab_width = 15
        if self.right_overlay_visible:
            # Position the tab next to the visible overlay
            overlay_width = 380
            self.right_side_tab.setGeometry(self.width() - overlay_width - tab_width, 150, tab_width, 100)
        else:
            # Position the tab at the edge of the screen when overlay is hidden
            self.right_side_tab.setGeometry(self.width() - tab_width, 150, tab_width, 100)

    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        
        # Keep left side tab in position relative to the overlay
        self.update_side_tab_position()
        
        # Keep right overlay correctly positioned relative to the current window edge
        if hasattr(self, 'right_overlay_container') and self.right_overlay_visible:
            overlay_width = 380
            self.right_overlay_container.setGeometry(self.width() - overlay_width, 0, overlay_width, 300)
        
        # Update right tab position
        if hasattr(self, 'right_side_tab'):
            self.update_right_tab_position()

    # Add to setup_right_overlay - Step size dropdown and time step input
    def setup_right_overlay(self):
        """Setup the right side overlay container and its components"""
        self.right_overlay_container = QWidget(self.main_widget)
        # Position relative to the current window width
        overlay_width = 380
        self.right_overlay_container.setGeometry(self.width() - overlay_width, 0, overlay_width, 400)
        
        # Create info panel
        self.right_info_panel = RightInfoPanel(self.right_overlay_container)
        self.right_info_panel.setGeometry(20, 10, 340, 340)  # Made this smaller to fit new controls
        
        # Add dimension controls
        self.dimensions_label = QLabel("Dimensions:", self.right_overlay_container)
        self.dimensions_label.setGeometry(30, 110, 80, 30)
        self.dimensions_label.setStyleSheet("color: white; font-weight: bold;")
        
        # X dimension
        self.x_dim_label = QLabel("X:", self.right_overlay_container)
        self.x_dim_label.setGeometry(110, 110, 20, 30)
        self.x_dim_label.setStyleSheet("color: white;")
        
        self.x_dim_input = QSpinBox(self.right_overlay_container)
        self.x_dim_input.setGeometry(130, 110, 50, 30)
        self.x_dim_input.setRange(0, 10)
        self.x_dim_input.setValue(0)
        self.x_dim_input.setStyleSheet("""
            QSpinBox {
                background-color: white;
                border: 1px solid #D477B1;
                border-radius: 4px;
                padding: 2px 4px;
            }
            QSpinBox:focus {
                border: 2px solid #b92980;
            }
        """)
        
        # Y dimension
        self.y_dim_label = QLabel("Y:", self.right_overlay_container)
        self.y_dim_label.setGeometry(190, 110, 20, 30)
        self.y_dim_label.setStyleSheet("color: white;")
        
        self.y_dim_input = QSpinBox(self.right_overlay_container)
        self.y_dim_input.setGeometry(210, 110, 50, 30)
        self.y_dim_input.setRange(0, 9999)
        self.y_dim_input.setValue(1)
        self.y_dim_input.setStyleSheet("""
            QSpinBox {
                background-color: white;
                border: 1px solid #D477B1;
                border-radius: 4px;
                padding: 2px 4px;
            }
            QSpinBox:focus {
                border: 2px solid #b92980;
            }
        """)
        
        # Z dimension
        self.z_dim_label = QLabel("Z:", self.right_overlay_container)
        self.z_dim_label.setGeometry(270, 110, 20, 30)
        self.z_dim_label.setStyleSheet("color: white;")
        
        self.z_dim_input = QSpinBox(self.right_overlay_container)
        self.z_dim_input.setGeometry(290, 110, 50, 30)
        self.z_dim_input.setRange(0, 9999)
        self.z_dim_input.setValue(2)
        self.z_dim_input.setStyleSheet("""
            QSpinBox {
                background-color: white;
                border: 1px solid #D477B1;
                border-radius: 4px;
                padding: 2px 4px;
            }
            QSpinBox:focus {
                border: 2px solid #b92980;
            }
        """)
        
        # Speed control
        self.speed_label = QLabel("Plot Speed:", self.right_overlay_container)
        self.speed_label.setGeometry(30, 150, 60, 30)
        self.speed_label.setStyleSheet("color: white; font-weight: bold;")
        
        self.speed_input = QSpinBox(self.right_overlay_container)
        self.speed_input.setGeometry(130, 150, 50, 30)
        self.speed_input.setRange(1, 99999)
        self.speed_input.setValue(99999)
        self.speed_input.setStyleSheet("""
            QSpinBox {
                background-color: white;
                border: 1px solid #D477B1;
                border-radius: 4px;
                padding: 2px 4px;
            }
            QSpinBox:focus {
                border: 2px solid #b92980;
            }
        """)
        
        # Add time step input field
        self.time_step_label = QLabel("Time Step:", self.right_overlay_container)
        self.time_step_label.setGeometry(30, 70, 100, 30)
        self.time_step_label.setStyleSheet("color: white; font-weight: bold;")
        
        self.time_step_input = QDoubleSpinBox(self.right_overlay_container)
        self.time_step_input.setGeometry(130, 70, 100, 30)
        self.time_step_input.setRange(0.01, 10.0)
        self.time_step_input.setValue(0.1)
        self.time_step_input.setSingleStep(0.1)
        self.time_step_input.setDecimals(2)
        self.time_step_input.setStyleSheet("""
            QDoubleSpinBox {
                background-color: white;
                border: 1px solid #D477B1;
                border-radius: 4px;
                padding: 2px 4px;
            }
            QDoubleSpinBox:focus {
                border: 2px solid #b92980;
            }
        """)
        
        # Save to file checkbox
        self.save_to_file_label = QLabel("Save to File:", self.right_overlay_container)
        self.save_to_file_label.setGeometry(30, 180, 100, 30)
        self.save_to_file_label.setStyleSheet("color: white; font-weight: bold;")
        
        self.save_to_file_checkbox = QCheckBox(self.right_overlay_container)
        self.save_to_file_checkbox.setGeometry(130, 185, 20, 20)
        self.save_to_file_checkbox.setStyleSheet("""
            QCheckBox {
                background-color: transparent;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
            }
            QCheckBox::indicator:unchecked {
                background-color: white;
                border: 1px solid #D477B1;
                border-radius: 4px;
            }
            QCheckBox::indicator:checked {
                background-color: #D477B1;
                border: 1px solid #D477B1;
                border-radius: 4px;
            }
        """)
        
        # Log file path
        self.log_file_label = QLabel("Log File:", self.right_overlay_container)
        self.log_file_label.setGeometry(30, 225, 100, 30)
        self.log_file_label.setStyleSheet("color: white; font-weight: bold;")
        
        self.log_file_input = QLineEdit(self.right_overlay_container)
        self.log_file_input.setGeometry(130, 225, 160, 30)
        self.log_file_input.setText("boxes.txt")
        self.log_file_input.setStyleSheet("""
            QLineEdit {
                background-color: white;
                border: 1px solid #D477B1;
                border-radius: 4px;
                padding: 2px 4px;
            }
            QLineEdit:focus {
                border: 2px solid #b92980;
            }
        """)
        
        # Save config button
        self.save_config_button = StyledButton("Save Config", self.right_overlay_container)
        self.save_config_button.setGeometry(120, 263, 150, 30)
        self.save_config_button.clicked.connect(self.save_config)
        
        # Load config on startup
        self.load_config()
        
        # Initialize visibility
        self.right_overlay_visible = True

    def setup_right_tab(self):
        """Setup the side tab for showing/hiding right overlay"""
        self.right_side_tab = RightOverlayTab(self.main_widget)
        self.right_side_tab.on_click_callback = self.toggle_right_overlay
        self.update_right_tab_position()
        self.right_side_tab.raise_()
    def save_config(self):
        """Save the current configuration to a JSON file"""
        config = {
            'x_dim': self.x_dim_input.value(),
            'y_dim': self.y_dim_input.value(),
            'z_dim': self.z_dim_input.value(),
            'speed': self.speed_input.value(),
            'save': self.save_to_file_checkbox.isChecked(),
            'log_file': self.log_file_input.text()
        }
        
        try:
            with open('plotter_config.json', 'w') as f:
                json.dump(config, f, indent=4)
            # Show success message in the right info panel
            if hasattr(self.right_info_panel, 'status_label'):
                self.right_info_panel.status_label.setText("Configuration saved successfully!")
        except Exception as e:
            if hasattr(self.right_info_panel, 'status_label'):
                self.right_info_panel.status_label.setText(f"Error saving config: {str(e)}")

    def load_config(self):
        """Load configuration from a JSON file if it exists"""
        try:
            if os.path.exists('plotter_config.json'):
                with open('plotter_config.json', 'r') as f:
                    config = json.load(f)
                    
                # Update UI elements with loaded values
                self.x_dim_input.setValue(config.get('x_dim', 0))
                self.y_dim_input.setValue(config.get('y_dim', 1))
                self.z_dim_input.setValue(config.get('z_dim', 2))
                self.speed_input.setValue(int(config.get('speed', 100)))
                self.save_to_file_checkbox.setChecked(bool(config.get('save', False)))
                self.log_file_input.setText(config.get('log_file', 'boxes.txt'))
                
                if hasattr(self.right_info_panel, 'status_label'):
                    self.right_info_panel.status_label.setText("Configuration loaded!")
        except Exception as e:
            if hasattr(self.right_info_panel, 'status_label'):
                self.right_info_panel.status_label.setText(f"Error loading config: {str(e)}")

    def update_right_tab_position(self):
        """Update the position of the right side tab based on overlay visibility"""
        tab_width = 15
        if self.right_overlay_visible:
            # Position the tab next to the visible overlay
            overlay_width = 380
            self.right_side_tab.setGeometry(self.width() - overlay_width - tab_width, 150, tab_width, 100)
        else:
            # Position the tab at the edge of the screen when overlay is hidden
            self.right_side_tab.setGeometry(self.width() - tab_width, 150, tab_width, 100)

    # def setup_right_buttons(self):
    #     """Setup the control buttons for right panel"""
    #     self.load_boxes_button = StyledButton("Load Boxes", self.right_overlay_container)
    #     self.load_boxes_button.setGeometry(20, 260, 340, 30)
    #     self.load_boxes_button.clicked.connect(self.load_boxes_clicked)

    def toggle_right_overlay(self):
        """Toggle visibility of the right overlay"""
        self.right_overlay_visible = not self.right_overlay_visible
        
        # Delete and recreate the tab with the new position
        if hasattr(self, 'right_side_tab'):
            self.right_side_tab.deleteLater()
        
        # Create new tab in the correct position
        self.setup_right_tab()
        
        # Show/hide overlay as needed
        if self.right_overlay_visible:
            # Set position relative to the current window width
            overlay_width = 380
            self.right_overlay_container.setGeometry(self.width() - overlay_width, 0, overlay_width, 300)
            self.right_overlay_container.show()
            self.right_overlay_container.raise_()
        else:
            self.right_overlay_container.hide()
        
        self.right_side_tab.show()

    # def load_boxes_clicked(self):
    #     """Callback for the Load Boxes button"""
    #     #self.right_info_panel.status_label.setText("Loading boxes...")

    #     self.plotter.clear()
    #     self.plotter.show_grid(all_edges=True, padding = 1.0)

    #     load_and_plot_boxes(self.plotter, 1, log_file="boxes1.txt")         #Verse-library/demo/traffic_signalB/
    #     # grid_bounds = [-2400, 300, -1100, 600, 0, 1100]
    #     # self.plotter.show_grid(bounds=grid_bounds)

    # Update resizeEvent to handle right panel resizing
    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        
        # Keep left side tab in position relative to the overlay
        self.update_side_tab_position()
        
        # Keep right overlay correctly positioned relative to the current window edge
        if hasattr(self, 'right_overlay_container') and self.right_overlay_visible:
            overlay_width = 380
            self.right_overlay_container.setGeometry(self.width() - overlay_width, 0, overlay_width, 300)
        
        # Update right tab position
        if hasattr(self, 'right_side_tab'):
            self.update_right_tab_position()
       

    def update_side_tab_position(self):
        """Update the position of the side tab based on overlay visibility"""
        if self.overlay_visible:

            self.side_tab.setGeometry(480, 150, 40, 80)

            
        else:
            #self.side_tab.setGeometry(0, 150, 40, 80)

            self.side_tab.setGeometry(0, 150, 40, 80)



    def setup_sliders(self):
        """Setup the altitude sliders with styling that matches the webview"""
        self.slider_container = QWidget(self.overlay_container)
        self.slider_container.setGeometry(370, 10, 100, 350)
        
        # Create a label to show "Altitude" text
        self.altitude_label = QLabel("Altitude", self.slider_container)
        self.altitude_label.setGeometry(20, 3, 60, 20)
        self.altitude_label.setStyleSheet("color: #3498db; font-weight: bold;")
        
        self.blue_slider = SvgPlaneSlider("plane_blue.svg", self.slider_container)
        self.blue_slider.setGeometry(5, 30, 30, 280)
        self.blue_slider.setValue(50)

        self.red_slider = SvgPlaneSlider("plane_red.svg", self.slider_container)
        self.red_slider.setGeometry(55, 30, 30, 280)
        self.red_slider.setValue(50)
            

    def setup_slider_markers(self):
        """Setup markers for the sliders"""
        marker_values = [0, 33, 66, 100]
        
        # Blue slider markers
        self.blue_markers = []
        for value in marker_values:
            marker = QLabel(f"{value}", self.slider_container)
            y_pos = 30 + 280 - (280 * value / 100) - 10
            marker.setGeometry(36, int(y_pos), 30, 20)
            marker.setStyleSheet("color: #3498db; font-size: 10px;")
            self.blue_markers.append(marker)
        
        # Red slider markers
        self.red_markers = []
        for value in marker_values:
            marker = QLabel(f"{value}", self.slider_container)
            y_pos = 30 + 280 - (280 * value / 100) - 10
            marker.setGeometry(86, int(y_pos), 30, 20)
            marker.setStyleSheet("color: #e74c3c; font-size: 10px;")
            self.red_markers.append(marker)


    def setup_buttons(self):
        """Setup the control buttons"""
        button_width = 170  # Half the width of the original button
    
        self.run_button = StyledButton("Verify", self.overlay_container)
        self.run_button.setGeometry(10, 365, button_width, 30)
        self.run_button.clicked.connect(self.run_button_clicked)
        
        self.run_simulate_button = StyledButton("Simulate", self.overlay_container)
        self.run_simulate_button.setGeometry(10 + button_width + 10, 365, button_width, 30)  # Position next to first button with a small gap
        self.run_simulate_button.clicked.connect(self.run_simulate_clicked)


    def setup_web_view(self):
        """Setup the web view for the visualization with console support"""
        self.web_view = QWebEngineView(self.overlay_container)
        self.web_view.setGeometry(10, 10, 350, 350)

        
        self.web_view.setHtml(self.get_web_content())

    def get_web_content(self):
        """Get the HTML content for the web view"""
        return """
        <html>
            <head>
                <title>Traffic Signal Visualization</title>
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    body { 
                        overflow: hidden;
                        background-color: transparent;
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    }
                    #container {
                        position: absolute;
                        top: 0px;
                        left: 0px;
                        width: 350px;
                        height: 350px;
                        background-color: rgba(225, 244, 255, 0.8);
                        border: 2px solid #3498db;
                        border-radius: 8px;
                        z-index: 2;
                        overflow: hidden;
                    }
                    .axis {
                        position: absolute;
                        background-color: #3498db;
                        z-index: 1;
                    }
                    #x-axis {
                        width: 100%;
                        height: 2px;
                        bottom: 0;
                        left: 0;
                    }
                    #y-axis {
                        width: 2px;
                        height: 100%;
                        left: 0;
                        top: 0;
                    }
                    .marker {
                        position: absolute;
                        font-size: 12px;
                        color: #3498db;
                        font-weight: bold;
                    }
                    .x-marker {
                        bottom: 5px;
                        transform: translateX(-50%);
                    }
                    .y-marker {
                        left: 5px;
                        transform: translateY(-50%);
                    }
                    /* Origin marker */
                    .origin-marker {
                        bottom: 5px;
                        left: 5px;
                    }
                    /* Grid lines */
                    .grid-line {
                        position: absolute;
                        background-color: rgba(52, 152, 219, 0.15);
                        z-index: 0;
                    }
                    .grid-line-x {
                        width: 100%;
                        height: 1px;
                    }
                    .grid-line-y {
                        width: 1px;
                        height: 100%;
                    }
                    .draggable {
                        width: 20px;
                        height: 20px;
                        color: white;
                        text-align: center;
                        line-height: 20px;
                        font-weight: bold;
                        border-radius: 10px;
                        position: absolute;
                        cursor: grab;
                        z-index: 2;
                        transition: transform 0.2s;
                        transform-origin: center center;
                    }
                    .draggable:hover {
                        transform: scale(1.1);
                    }
                    #plane1 { left: 10px; top: 10px; }
                    #plane2 { right: 10px; top: 10px; }
                </style>
                <script>
                    let draggedElement = null;
                    let selectedPlane = null;
                    let initialSize = 20;
                    let plane1Size = initialSize;
                    let plane2Size = initialSize;
                    function dragStart(event) {
                        draggedElement = event.target;
                    }

                    function dragOver(event) {
                        event.preventDefault();
                    }

                    function drop(event) {
                        if (draggedElement) {
                            let container = document.getElementById("container");
                            let rect = container.getBoundingClientRect();

                            let newX = event.clientX - rect.left - draggedElement.offsetWidth / 2;
                            let newY = event.clientY - rect.top - draggedElement.offsetHeight / 2;

                            newX = Math.max(0, Math.min(newX, rect.width - draggedElement.offsetWidth));
                            newY = Math.max(0, Math.min(newY, rect.height - draggedElement.offsetHeight));

                            draggedElement.style.left = newX + "px";
                            draggedElement.style.top = newY + "px";

                            event.preventDefault();
                        }
                    }

                    function planeClick(event) {
                        event.stopPropagation();
                        
                        // Find the parent element with class "draggable" (the plane div)
                        let planeElement = event.target;
                        while (planeElement && !planeElement.classList.contains('draggable')) {
                            planeElement = planeElement.parentElement;
                        }
                        
                        // If we couldn't find a draggable parent, exit
                        if (!planeElement) return;
                        
                        // If clicking the same plane, deselect it
                        if (selectedPlane === planeElement) {
                            selectedPlane = null;
                            planeElement.style.border = "none";
                            return;
                        }
                        
                        // Deselect previous plane if any
                        if (selectedPlane) {
                            selectedPlane.style.border = "none";
                        }
                        
                        // Select new plane
                        selectedPlane = planeElement;
                        selectedPlane.style.border = "2px dashed black";
                        
                        if (planeElement === draggedElement) {
                            draggedElement = null;
                        }
                        
                        console.log("Selected plane ID:", selectedPlane.id);
                    }
                   

                    // Modified handleWheel function
                    function handleWheel(event) {
                        if (selectedPlane) {
                            event.preventDefault();
                            // Determine which plane is selected and update its size variable
                            let planeSize = String(selectedPlane.id) === "plane1" ? plane1Size : plane2Size;
                            const delta = event.deltaY > 0 ? -2 : 2;
                            const newSize = Math.max(10, Math.min(50, planeSize + delta));
                            
                            // Update the appropriate size variable
                            if (selectedPlane.id === "plane1") {
                                plane1Size = newSize;

                            } else {
                                plane2Size = newSize;
                            }
                            
                            // Store current position
                            const currentLeft = parseInt(selectedPlane.style.left || 0);
                            const currentTop = parseInt(selectedPlane.style.top || 0);
                            
                            // Calculate center point
                            const centerX = currentLeft + planeSize / 2;
                            const centerY = currentTop + planeSize / 2;
                            
                            // Apply new size
                            selectedPlane.style.width = newSize + "px";
                            selectedPlane.style.height = newSize + "px";
                            selectedPlane.style.lineHeight = newSize + "px";
                            selectedPlane.style.borderRadius = (newSize / 2) + "px";
                            
                            // Adjust position to maintain center point
                            selectedPlane.style.left = (centerX - newSize / 2) + "px";
                            selectedPlane.style.top = (centerY - newSize / 2) + "px";
                        }
                    }

                    // Modified getPlanePositions function
                    function getPlanePositions() {
                        let plane1 = document.getElementById("plane1");
                        let plane2 = document.getElementById("plane2");
                        
                        return {
                            plane1: { 
                                x: plane1.style.left || "0px", 
                                y: plane1.style.top || "0px",
                                size: plane1Size + "px"
                            },
                            plane2: { 
                                x: plane2.style.left || "0px", 
                                y: plane2.style.top || "0px",
                                size: plane2Size + "px"
                            }
                        };
                    }

                    function clearSelection(event) {
                        if (event.target.id === "container") {
                            if (selectedPlane) {
                                selectedPlane.style.border = "none";
                                selectedPlane = null;
                            }
                        }
                    }

                    function runSimulation() {
                        let positions = getPlanePositions();
                        console.log("Saving Positions:", positions);
                        window.pyQtApp.savePositions(positions);
                    }

                    function setupPlaneEventListeners(planeId) {
                        const plane = document.getElementById(planeId);
                        plane.addEventListener('click', planeClick);
                        
                        // Also attach to all children elements
                        const children = plane.querySelectorAll('*');
                        children.forEach(child => {
                            child.addEventListener('click', planeClick);
                        });
                    }

                    window.onload = function() {
                        let container = document.getElementById('container');
                        container.addEventListener('dragover', dragOver);
                        container.addEventListener('drop', drop);
                        container.addEventListener('click', clearSelection);
                        container.addEventListener('wheel', handleWheel);
                        
                        setupPlaneEventListeners('plane1');
                        setupPlaneEventListeners('plane2');
                    };
                </script>
            </head>
            <body>
                <div id="container">
                    <!-- Grid lines for X-axis (horizontal lines) -->
                    <div class="grid-line grid-line-x" style="top: 50px;"></div>
                    <div class="grid-line grid-line-x" style="top: 100px;"></div>
                    <div class="grid-line grid-line-x" style="top: 150px;"></div>
                    <div class="grid-line grid-line-x" style="top: 200px;"></div>
                    <div class="grid-line grid-line-x" style="top: 250px;"></div>
                    <div class="grid-line grid-line-x" style="top: 300px;"></div>
                    
                    <!-- Grid lines for Y-axis (vertical lines) -->
                    <div class="grid-line grid-line-y" style="left: 50px;"></div>
                    <div class="grid-line grid-line-y" style="left: 100px;"></div>
                    <div class="grid-line grid-line-y" style="left: 150px;"></div>
                    <div class="grid-line grid-line-y" style="left: 200px;"></div>
                    <div class="grid-line grid-line-y" style="left: 250px;"></div>
                    <div class="grid-line grid-line-y" style="left: 300px;"></div>
                    
                    <div id="x-axis" class="axis"></div>
                    <div id="y-axis" class="axis"></div>
                    
                    <!-- Origin marker (0,0) -->
                    <div class="marker origin-marker">0</div>
                    
                    <!-- X-Axis Markers -->
                    <div class="marker x-marker" style="left: 50px;">300</div>
                    <div class="marker x-marker" style="left: 100px;">600</div>
                    <div class="marker x-marker" style="left: 150px;">900</div>
                    <div class="marker x-marker" style="left: 200px;">1200</div>
                    <div class="marker x-marker" style="left: 250px;">1500</div>
                    <div class="marker x-marker" style="left: 300px;">1800</div>

                    <!-- Y-Axis Markers -->
                    <div class="marker y-marker" style="top: 50px;">1800</div>
                    <div class="marker y-marker" style="top: 100px;">1500</div>
                    <div class="marker y-marker" style="top: 150px;">1200</div>
                    <div class="marker y-marker" style="top: 200px;">900</div>
                    <div class="marker y-marker" style="top: 250px;">600</div>
                    <div class="marker y-marker" style="top: 300px;">300</div>

                    <div id="plane1" class="draggable" draggable="true" ondragstart="dragStart(event)">
                    <svg viewBox="0 0 24 24" width="100%" height="100%" preserveAspectRatio="xMidYMid meet">
                        <path d="M21,16V14L13,9V3.5A1.5,1.5,0,0,0,11.5,2A1.5,1.5,0,0,0,10,3.5V9L2,14V16L10,13.5V19L8,20.5V22L11.5,21L15,22V20.5L13,19V13.5Z" fill="#007BFF"/>
                    </svg>
                    </div>
                    <div id="plane2" class="draggable" draggable="true" ondragstart="dragStart(event)">
                    <svg viewBox="0 0 24 24" width="100%" height="100%" preserveAspectRatio="xMidYMid meet" style="transform: rotate(90deg);">
                        <path d="M21,16V14L13,9V3.5A1.5,1.5,0,0,0,11.5,2A1.5,1.5,0,0,0,10,3.5V9L2,14V16L10,13.5V19L8,20.5V22L11.5,21L15,22V20.5L13,19V13.5Z" fill="#FF0000"/>
                    </svg>
                    </div>
                </div>
            </body>
        </html>
        """
    def toggle_overlay(self):
        self.overlay_visible = not self.overlay_visible
        
        # Delete and recreate the tab with the new position
        if hasattr(self, 'side_tab'):
            self.side_tab.deleteLater()
        
        # Create new tab in the correct position
        
        self.setup_side_tab()
        # Show/hide overlay as needed
        if self.overlay_visible:
            self.overlay_container.show()
            self.overlay_container.raise_()
        else:
            self.overlay_container.hide()
        self.side_tab.show()


    def run_verse(self,  initial_sets, agent_types, dls, verify ):
        

        script_dir = os.path.realpath(os.path.dirname(__file__))
        input_code_name = os.path.join(script_dir, "controller_3d.py")
        car = CarAgent('car1', file_name=input_code_name)
        car2 = NPCAgent('car2')
        scenario = Scenario(ScenarioConfig(parallel=False))
        car.set_initial(
            # initial_state=[[x1 - r1, y1 - r1, -1, np.pi/3, np.pi/6, 100], [x1 + r1, y1 + r1, 1, np.pi/3, np.pi/6, 100]],
            initial_state=[[-1, -1010, -1, np.pi/3, np.pi/6, 100], [1, -990, 1, np.pi/3, np.pi/6, 100]],
            # initial_state=[[0, -1000, 0, np.pi/3, np.pi/6, 100], [0, -1000, 0, np.pi/3, np.pi/6, 100]],
            # initial_state=[[0, -1000, np.pi/3, 100], [0, -1000, np.pi/3, 100]],
            initial_mode=(AgentMode.COC,   )
        )
        car2.set_initial(
            initial_state=[[-2001, -10, 999, 0,0, 100], [-1999, 10, 1001, 0,0, 100]],
            # initial_state=[[-2000, 0, 1000, 0,0, 100], [-2000, 0, 1000, 0,0, 100]],
            # initial_state=[[x2 - r2,  y2 - r2, 1000,0,0, 100], [x2 + r2, y2 + r2, 1001, 0,0, 100]],
            initial_mode=(AgentMode.COC,   )
        )
        T = 20
        Tv = 1
        ts = 0.01
        # observation: for Tv = 0.1 and a larger initial set of radius 10 in y dim, the number of 

        scenario.config.print_level = 0
        if verify:
            scenario.config.reachability_method = ReachabilityMethod.DRYVR_DISC
            self.plotter.show_grid()
            scenario.add_agent(car)
            scenario.add_agent(car2)
            start = time.perf_counter()
            trace = scenario.verify(Tv, ts, self.plotter) # this is the root
            id = 1+trace.root.id
            models = [[torch.load(f"./examples/simple/acasxu_crown/nets/ACASXU_run2a_{net + 1}_{tau + 1}_batch_2000.pth") for tau in range(9)] for net in range(5)]
            norm = float("inf")

            queue = deque()
            queue.append(trace.root) # queue should only contain ATNs  
            ### begin looping
            while len(queue):
                cur_node = queue.popleft() # equivalent to trace.nodes[0] in this case
                own_state, int_state = get_final_states_verify(cur_node)
                print(own_state, int_state)
                tau_idx_min, tau_idx_max = get_tau_idx(own_state[1], int_state[0]), get_tau_idx(own_state[0], int_state[1]) 
                # print(tau_idx_min, tau_idx_max)
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

                    last_cmd = getattr(AgentMode, cur_node.mode['car1'][0]).value  # cur_mode.mode[.] is some string 
                    for tau_idx in range(tau_idx_min, tau_idx_max+1):
                        lirpa_model = BoundedModule(models[last_cmd-1][tau_idx], (torch.empty_like(x))) 
                        # lirpa_model = BoundedModule(model, (torch.empty_like(x))) 
                        ptb_x = PerturbationLpNorm(norm = norm, x_L=x_l, x_U=x_u)
                        bounded_x = BoundedTensor(x, ptb=ptb_x)
                        lb, ub = lirpa_model.compute_bounds(bounded_x, method='alpha-CROWN')
                        # lb, ub = lirpa_model.compute_bounds(bounded_x, method='backward') # 

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
                        [[own_state[0][1:], own_state[1][1:]], [int_state[0][1:], int_state[1][1:]]], # this should eventually be a range 
                        [(AgentMode(new_m),   ),(AgentMode.COC,   )]
                    )
                    id += 1
                    # new_trace = scenario.simulate(Tv, ts)
                    new_trace = scenario.verify(Tv, ts, self.plotter)
                    temp_root = new_trace.root
                    new_node = cur_node.new_child(temp_root.init, temp_root.mode, temp_root.trace, cur_node.start_time + Tv, id)
                    cur_node.child.append(new_node)
                    print(f'Start time: {new_node.start_time}\nNode ID: {id}\nNew mode: {AgentMode(new_m)}')
                        
                    if new_node.start_time + Tv>=T: # if the time of the current simulation + start_time is at or above total time, don't add
                        continue
                    queue.append(new_node)

            trace.nodes = trace._get_all_nodes(trace.root)
            print(f'Verification time: {time.perf_counter()-start}')
        
        else:
            T = 20
            Tv = 1
            ts = 0.01
            N = 100
            # observation: for Tv = 0.1 and a larger initial set of radius 10 in y dim, the number of 
            self.plotter.show_grid()

            scenario.config.print_level = 0
            # scenario.config.reachability_method = ReachabilityMethod.DRYVR_DISC
            scenario.add_agent(car)
            scenario.add_agent(car2)
            start = time.perf_counter()
            # trace = scenario.simulate(Tv, ts)
            # id = 1+trace.root.id
            # net = 0 # eventually this could be modified in the loop by some cmd_list var
            # model = torch.load(f"./examples/simple/acasxu_crown/ACASXU_run2a_{net + 1}_1_batch_2000.pth")
            models = [[torch.load(f"./examples/simple/acasxu_crown/nets/ACASXU_run2a_{net + 1}_{tau + 1}_batch_2000.pth") for tau in range(9)] for net in range(5)]
            norm = float("inf")

            # queue = deque()
            # queue.append(trace.root) # queue should only contain ATNs  
            ### begin looping
            traces = []
            for i in range(N):
                scenario.set_init(
                    [[[-100, -1100, -1, np.pi/3, np.pi/6, 100], [100, -900, 1, np.pi/3, np.pi/6, 100]],
                    [[-2000, 0, 1000, 0,0, 100], [-2000, 0, 1000, 0,0, 100]]],
                    [(AgentMode.COC,  ), (AgentMode.COC,  )]
                )
                trace = scenario.simulate(Tv, ts, self.plotter) # this is the root
                id = 1+trace.root.id
                # net = 0 # eventually this could be modified in the loop by some cmd_list var
                # model = torch.load(f"./examples/simple/acasxu_crown/ACASXU_run2a_{net + 1}_1_batch_2000.pth")
                queue = deque()
                queue.append(trace.root) # queue should only contain ATNs  
                while len(queue):
                    cur_node = queue.popleft() # equivalent to trace.nodes[0] in this case
                    own_state, int_state = get_final_states_sim(cur_node)
                    acas_state = get_acas_state(own_state[1:], int_state[1:]).float()
                    acas_state = (acas_state-means_for_scaling)/range_for_scaling # normalization
                    # ads = model(acas_state.view(1,5)).detach().numpy()
                    last_cmd = getattr(AgentMode, cur_node.mode['car1'][0]).value  # cur_mode.mode[.] is some string
                    tau_idx = get_tau_idx(own_state[1:], int_state[1:])
                    # print(f'Last Command: {last_cmd}, Tau Index: {tau_idx}')
                    ads = models[last_cmd-1][tau_idx](acas_state.view(1,5)).detach().numpy()
                    new_mode = np.argmin(ads[0])+1 # will eventually be a list
                    scenario.set_init(
                        [[own_state[1:], own_state[1:]], [int_state[1:], int_state[1:]]], # this should eventually be a range 
                        [(AgentMode(new_mode),  ),(AgentMode.COC,  )]
                    )
                    id += 1
                    new_trace = scenario.simulate(Tv, ts, self.plotter)
                    temp_root = new_trace.root
                    new_node = cur_node.new_child(temp_root.init, temp_root.mode, temp_root.trace, cur_node.start_time + Tv, id)
                    cur_node.child.append(new_node)
                    if new_node.start_time + Tv>=T: # if the time of the current simulation + start_time is at or above total time, don't add
                        continue
                    queue.append(new_node)
                
                trace.nodes = trace._get_all_nodes(trace.root)
                traces.append(trace)

            print(f'Total {N} simulations: {(time.perf_counter()-start):.2f} s')
            trace.nodes = trace._get_all_nodes(trace.root)

    def run_button_clicked(self):
        """Callback for the Run button (normal mode)"""
        self.web_view.page().runJavaScript("getPlanePositions();", 
                                        lambda positions: self.handle_positions(positions, True))

    def run_simulate_clicked(self):
        """Callback for the Run Simulate button (simulation mode)"""
        self.web_view.page().runJavaScript("getPlanePositions();", 
                                        lambda positions: self.handle_positions(positions, False))


    def handle_positions(self, positions, verify):

        # Check if file input has content
        file_path = self.file_input.text().strip()
        if file_path:
            # If file specified, load boxes first
            if hasattr(self, 'thread') and self.thread:
                self.thread.join()
            self.plotter.clear()
            self.plotter.show_grid(all_edges=True, n_xlabels = 6, n_ylabels = 6, n_zlabels = 6)
            load_and_plot(self.plotter, log_file=file_path)
            #grid_bounds = [-2400, 300, -1100, 600, 0, 1100]
            #self.plotter.show_grid(axes_ranges=grid_bounds)
        else:
            def position_to_int(pos):
                # Handle case where positions might be empty
                try:
                    x = float(pos['x'].replace('px', ''))
                    y = float(pos['y'].replace('px', ''))
                    s = float(pos['size'].replace('px', ''))
                    return ( 6*(x)+48 , 6*(350-y) - 48, s/.3)
                except (ValueError, KeyError):
                    return (0, 0, 0)
                    
            z1 = self.blue_slider.value() / 10  # Scale as needed
            z2 = self.red_slider.value() / 10  # Scale as needed
            # positions is a JavaScript object containing the current positions of the planes
            print("Positions of planes:", positions)

            # Process the positions and use them in the 3D visualization or other tasks
            self.plane1_position = positions['plane1']
            self.plane2_position = positions['plane2']

            x1, y1, s1 = (position_to_int(self.plane1_position))
            x2, y2, s2 = (position_to_int(self.plane2_position))

            if hasattr(self, 'thread') and self.thread:
                self.thread.join()
            self.plotter.clear()

            self.run_verse([[x1, y1, z1, s1], [x2, y2, z2, s2]], ["vehicle", "pedestrian"], ["vehicle_controller.py", "None"], verify )

            # self.thread = threading.Thread(target=self.run_verse, args=[ [[x1, y1, z1, s1], [x2, y2, z2, s2]], ["vehicle", "pedestrian"], ["vehicle_controller.py", "None"], verify ], daemon=True)
            # self.thread.start()


    def _set_python_bridge(self, result):
        """Sets the python bridge object in JS"""
        self.web_view.page().runJavaScript("window.pyQtApp = pyQtApp;", self._set_python_bridge)

def resizeEvent(self, event):
    """Handle window resize events"""
    super().resizeEvent(event)
    
    # Keep left side tab in position relative to the overlay
    self.update_side_tab_position()
    
    # Keep right overlay at a fixed position regardless of window size
    if hasattr(self, 'right_overlay_container'):
        # Set position to a fixed coordinate rather than relative to window width
        if self.right_overlay_visible:
            self.right_overlay_container.setGeometry(self.width() - 380, 0, 380, 300)
    
    # Update right tab position
    if hasattr(self, 'right_side_tab'):
        self.update_right_tab_position()

# Run the Qt Application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    main_window = MainWindow()
    main_window.showMaximized()
    
    sys.exit(app.exec())













      
        

    
