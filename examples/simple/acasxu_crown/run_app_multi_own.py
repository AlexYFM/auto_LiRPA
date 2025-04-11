from wsgiref.validate import PartialIteratorWrapper
from dubins_3d_agent import CarAgent, NPCAgent
from verse import Scenario, ScenarioConfig
from verse.map.example_map.map_tacas import M1

from verse.plotter.plotter2D import *
from verse.plotter.plotter3D import *
from verse.plotter.plotter3D_new import *

from verse.analysis.verifier import ReachabilityMethod
import plotly.graph_objects as go
import torch
from auto_LiRPA import BoundedTensor
from verse.utils.utils import wrap_to_pi
from collections import deque
from torch import nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

import sys
import numpy as np
import os
import threading
import time
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, 
    QPushButton, QSizePolicy, QStackedLayout, QSlider,
    QLabel, QFrame
)
from PyQt6.QtCore import Qt, QPoint, QRect, QSize
from PyQt6.QtGui import QPainter, QPixmap, QColor, QPalette, QFont, QPen, QPainterPath
from PyQt6.QtSvg import QSvgRenderer
import pyvistaqt as pvqt
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QStyleOptionSlider
from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import QStyle
from enum import Enum, auto

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

class StyledSlider(QSlider):
    """Custom styled slider with modern look"""
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.setStyleSheet("""
            QSlider::groove:vertical {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #2c3e50, stop:1 #34495e);
                width: 1px;
                border-radius: 1px;
            }
            QSlider::handle:vertical {
                background: #E1F4FF;
                border: 2px solid #2980b9;
                border-radius: 10px;
                height: 20px;
                width: 20px;
                margin: -6px 0;
            }
            QSlider::handle:vertical:hover {
                background: #2980b9;
            }
            QSlider::sub-page:vertical {
                background: #E1F4FF;
                border-radius: 4px;
            }
        """)

class StyledButton(QPushButton):
    """Custom styled button with modern look"""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #2472a4;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """)
class OverlayTab(QFrame):
    """Side tab for showing/hiding overlay with rounded corners and smaller size"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(15, 100)  # Smaller size
        self.setStyleSheet("""
            QFrame {
                background-color: #77B1D4;
                border-top-right-radius: 15px;
                border-bottom-right-radius: 15px;
                border: none;
            }
            QFrame:hover {
                background-color: #2980b9;
            }
        """)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Save the current state
        painter.save()
        
        # Set up the pen for drawing
        painter.setPen(QPen(Qt.GlobalColor.black, 2))  # Using black color with thicker line
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get widget dimensions
        width = self.width()
        height = self.height()
        
        # Draw triangular line (two connected lines forming a triangle-like shape)
        path = QPainterPath()
        # Start from top right area
        path.moveTo(width * 0.7, height * 0.2)
        # Draw line to middle left
        path.lineTo(width * 0.3, height * 0.5)
        # Draw line to bottom right area
        path.lineTo(width * 0.7, height * 0.8)
        
        painter.drawPath(path)
        
        # Restore the previous state
        painter.restore()

    def mousePressEvent(self, event):
        # Call the callback function
        if hasattr(self, 'on_click_callback') and self.on_click_callback:
            self.on_click_callback()
        super().mousePressEvent(event)

class InfoPanel(QFrame):
    """Information panel with modern styling"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: rgba(44, 62, 80, 0.9);
                border-radius: 8px;
                padding: 10px;
            }
            QLabel {
                color: white;
                font-size: 12px;
            }
            QLabel#title {
                font-size: 14px;
                font-weight: bold;
                color: #3498db;
                margin-bottom: 5px;
            }
        """)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        title = QLabel("Simulation Controls")
        title.setObjectName("title")
        layout.addWidget(title)
        
        self.status_label = QLabel("Ready to start simulation")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
class SvgPlaneSlider(QSlider):
    """Custom styled slider with plane icon handle and internal markers that match the webview style"""
    def __init__(self, svg_file, parent=None):
        super().__init__(Qt.Orientation.Vertical, parent)
        self.svg_renderer = QSvgRenderer(svg_file)
        self.setMinimum(0)
        self.setMaximum(100)
        self.setValue(50)
        self.setFixedSize(30, 280)
        self.setInvertedAppearance(True)
        
        # Create marker values (every 50 units to match webview)
        self.marker_values = [0, 50, 100, 150, 200, 250, 300]
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # First, clear the background with the same color as the webview
        painter.fillRect(self.rect(), QColor("#E1F4FF"))
        
        # Draw the groove
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        
        # Get groove dimensions
        groove_rect = self.style().subControlRect(QStyle.ComplexControl.CC_Slider, option, QStyle.SubControl.SC_SliderGroove, self)
        slider_length = groove_rect.height()
        slider_start = groove_rect.top()
        slider_center_x = groove_rect.center().x()
        
        # Draw vertical axis line that matches the webview blue color
        #painter.setPen(QPen(QColor("#3498db"), 2))
        #painter.drawLine(slider_center_x, slider_start, slider_center_x, slider_start + slider_length)
        
        # Add marker lines and text with the webview styling
        font = painter.font()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        
        # Draw grid lines matching webview
        painter.setPen(QPen(QColor("rgba(52, 152, 219, 0.15)"), 1))
        line_width = 28
        
        # Draw horizontal grid lines
        for value in self.marker_values:
            if value == 0:
                continue  # Skip the bottom line as we'll draw it separately
                
            # Calculate y position for the marker (inverted to match altitude)
            normalized_value = 300 - value  # Invert to match altitude (0=bottom, 300=top)
            y_pos = slider_start + int(slider_length * normalized_value / 300)
            
            # Draw grid line
            painter.drawLine(2, y_pos, 28, y_pos)
        
        # Draw marker text and ticks with webview blue color
        painter.setPen(QPen(QColor("#3498db"), 1))
        
        for i, value in enumerate(self.marker_values):
            # Calculate y position for the marker (inverted to match altitude)
            normalized_value = 300 - value  # Invert to match altitude (0=bottom, 300=top)
            y_pos = slider_start + int(slider_length * normalized_value / 300)
            
            # Draw ticker mark
            painter.drawLine(slider_center_x - 4, y_pos, slider_center_x + 4, y_pos)
            
            # Draw text with same style as webview
            text = str(value)
            text_rect = QRectF(2, y_pos - 13, 25, 16)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft, text)
        
        # Draw the SVG at the calculated position
        handle_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider, option, QStyle.SubControl.SC_SliderHandle, self)
        
        # Use the plane SVG for the handle
        self.svg_renderer.render(painter, QRectF(0, handle_rect.top(), 30, 30))
        
        painter.end()

def setup_sliders(self):
    """Setup the altitude sliders"""
    self.slider_container = QWidget(self.overlay_container)
    self.slider_container.setGeometry(370, 10, 100, 350)
    
    try:
        script_dir = os.path.realpath(os.path.dirname(__file__))
        input_code_name = os.path.join(script_dir, "plane_blue.svg")
        self.blue_slider = SvgPlaneSlider("plane_blue.svg", self.slider_container)
        # self.blue_slider = SvgPlaneSlider(input_code_name, self.slider_container)
        self.blue_slider.setGeometry(5, 30, 30, 280)
        self.blue_slider.setValue(50)

        input_code_name = os.path.join(script_dir, "plane_red.svg")
        # self.blue_slider = SvgPlaneSlider(input_code_name, self.slider_container)
        self.red_slider = SvgPlaneSlider("plane_red.svg", self.slider_container)
        self.red_slider.setGeometry(55, 30, 30, 280)
        self.red_slider.setValue(50)
        
        # Add labels
        self.blue_label = QLabel("Altitude", self.slider_container)
        self.blue_label.setGeometry(20, 3, 50, 20)
        self.blue_label.setStyleSheet("color: white; font-weight: bold;")
        
    except Exception as e:
        print(f"Error loading SVG sliders: {e}")
        self.setup_fallback_sliders()

def setup_fallback_sliders(self):
    """Setup fallback sliders if SVG loading fails"""
    self.blue_slider = StyledSlider(Qt.Orientation.Vertical, self.slider_container)
    self.blue_slider.setGeometry(5, 30, 30, 280)
    self.blue_slider.setValue(50)
    
    self.red_slider = StyledSlider(Qt.Orientation.Vertical, self.slider_container)
    self.red_slider.setGeometry(55, 30, 30, 280)
    self.red_slider.setValue(50)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic Signal Visualization")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize variables
        self.plane1_position = 0
        self.plane2_position = 0
        self.thread = None
        self.overlay_visible = True
        
        # Setup main UI
        self.setup_main_ui()
        
        # Setup overlay
        self.setup_overlay()
        
        # Setup web view
        self.setup_web_view()
        
        # Setup side tab
        self.setup_side_tab()
        
        # Make sure overlay is visible
        self.overlay_container.raise_()
        self.overlay_container.show()
        
        self.side_tab.raise_()


    def setup_main_ui(self):
        """Setup the main UI components"""
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Setup PyVista plotter
        self.plotter = pvqt.QtInteractor()
        self.plotter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.main_layout.addWidget(self.plotter.interactor)

    def setup_overlay(self):
        """Setup the overlay container and its components"""
        self.overlay_container = QWidget(self.main_widget)
        self.overlay_container.setGeometry(0, 0, 480, 400)
        
        # Create info panel
        self.info_panel = InfoPanel(self.overlay_container)
        self.info_panel.setGeometry(10, 10, 350, 350)
        
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
        
        try:
            self.blue_slider = SvgPlaneSlider("plane_blue.svg", self.slider_container)
            self.blue_slider.setGeometry(5, 30, 30, 280)
            self.blue_slider.setValue(50)

            self.red_slider = SvgPlaneSlider("plane_red.svg", self.slider_container)
            self.red_slider.setGeometry(55, 30, 30, 280)
            self.red_slider.setValue(50)
            
        except Exception as e:
            print(f"Error loading SVG sliders: {e}")
            self.setup_fallback_sliders()
          
            
            #self.setup_slider_markers()
        except Exception as e:
            print(f"Error loading SVG sliders: {e}")
            self.setup_fallback_sliders()

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

    def setup_fallback_sliders(self):
        """Setup fallback sliders if SVG loading fails"""
        self.blue_slider = StyledSlider(Qt.Orientation.Vertical, self.slider_container)
        self.blue_slider.setGeometry(5, 30, 30, 280)
        self.blue_slider.setValue(50)
        
        self.red_slider = StyledSlider(Qt.Orientation.Vertical, self.slider_container)
        self.red_slider.setGeometry(55, 30, 30, 280)
        self.red_slider.setValue(50)

    def setup_buttons(self):
        """Setup the control buttons"""
        self.run_button = StyledButton("Run", self.overlay_container)
        self.run_button.setGeometry(10, 365, 350, 30)
        self.run_button.clicked.connect(self.run_button_clicked)

    def setup_web_view(self):
        """Setup the web view for the visualization"""
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
                    
                    # <!-- X-Axis Markers -->
                    # <div class="marker x-marker" style="left: 50px;">50</div>
                    # <div class="marker x-marker" style="left: 100px;">100</div>
                    # <div class="marker x-marker" style="left: 150px;">150</div>
                    # <div class="marker x-marker" style="left: 200px;">200</div>
                    # <div class="marker x-marker" style="left: 250px;">250</div>
                    # <div class="marker x-marker" style="left: 300px;">300</div>

                    # <!-- Y-Axis Markers -->
                    # <div class="marker y-marker" style="top: 50px;">300</div>
                    # <div class="marker y-marker" style="top: 100px;">250</div>
                    # <div class="marker y-marker" style="top: 150px;">200</div>
                    # <div class="marker y-marker" style="top: 200px;">150</div>
                    # <div class="marker y-marker" style="top: 250px;">100</div>
                    # <div class="marker y-marker" style="top: 300px;">50</div>

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


      
        

    
    def run_verse(self, x1, y1, z1, r1, x2, y2, z2, r2):
        # script_dir = os.path.realpath(os.path.dirname(__file__))
        # input_code_name = os.path.join(script_dir, "vehicle_controller.py")
        # vehicle = VehicleAgent('car', file_name=input_code_name)
        # pedestrian = PedestrianAgent('pedestrian')

        # scenario = Scenario(ScenarioConfig(init_seg_length=1, parallel=False))

        # scenario.add_agent(vehicle) 
        # scenario.add_agent(pedestrian)
        # scenario.set_sensor(VehiclePedestrianSensor())

        # init_car = [[x1 - r1, y1 - r1, 0, 8], [x1 + r1, y1 + r1, 0, 8]]
        # init_pedestrian = [[x2 - r2, y2 - r2, 0, 3], [x2 + r2, y2 + r2, 0, 5]]

        # scenario.set_init_single(
        #     'car', init_car, (VehicleMode.Normal,)
        # )
        # scenario.set_init_single(
        #     'pedestrian', init_pedestrian, (PedestrianMode.Normal,)
        # )

        # self.plotter.show_grid()
        # traces = scenario.verify(80, 0.1, self.plotter)

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
            initial_state=[[-1, -1, -1, np.pi, np.pi/6, 100], [1, 1, 1, np.pi, np.pi/6, 100]],
            initial_mode=(AgentMode.COC, )
        )
        car2.set_initial(
            # initial_state=[[-2000, 0, 1000, 0,0, 100], [-2000, 0, 1000, 0,0, 100]],
            # initial_state=[[-2001, -10, 999, 0,0, 100], [-1999, 10, 1001, 0,0, 100]],
            initial_state=[[-1001, -1, 999, 0,0, 100], [-999, 1, 1000, 0,0, 100]],
            initial_mode=(AgentMode.COC, )
        )
        T = 20
        Tv = 1
        ts = 0.1
        # observation: for Tv = 0.1 and a larger initial set of radius 10 in y dim, the number of 

        scenario.config.print_level = 0
        scenario.config.reachability_method = ReachabilityMethod.DRYVR_DISC
        scenario.add_agent(car)
        scenario.add_agent(car2)
        start = time.perf_counter()
        self.plotter.show_grid()
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
            tau_idx_min, tau_idx_max = get_tau_idx(own_state[1], int_state[0]), get_tau_idx(own_state[0], int_state[1]) 
            # print(tau_idx_min, tau_idx_max)
            
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

                last_cmd = getattr(AgentMode, cur_node.mode['car1'][0]).value  # cur_mode.mode[.] is some string 
                for tau_idx in range(tau_idx_min, tau_idx_max+1):
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
            
            int_modes = set()
            int_reachsets = get_acas_reach(np.array(int_state)[:,1:], np.array(own_state)[:,1:])
            tau_idx_min_int, tau_idx_max_int = get_tau_idx(int_state[1], own_state[0]), get_tau_idx(int_state[0], own_state[1]) 
            # print(reachsets)
            for reachset in int_reachsets:
                if len(int_modes)==5: # if all modes are possible, stop iterating
                    break 
                acas_min, acas_max = reachset
                acas_min, acas_max = (acas_min-means_for_scaling)/range_for_scaling, (acas_max-means_for_scaling)/range_for_scaling
                x_l, x_u = torch.tensor(acas_min).float().view(1,5), torch.tensor(acas_max).float().view(1,5)
                x = (x_l+x_u)/2

                last_cmd = getattr(AgentMode, cur_node.mode['car2'][0]).value  # cur_mode.mode[.] is some string 
                for tau_idx in range(tau_idx_min, tau_idx_max+1):
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
                    new_trace = scenario.verify(Tv, ts, self.plotter)
                    temp_root = new_trace.root
                    new_node = cur_node.new_child(temp_root.init, temp_root.mode, temp_root.trace, cur_node.start_time + Tv, id)
                    cur_node.child.append(new_node)
                    print(f'Start time: {new_node.start_time}\nNode ID: {id}\nNew modes: {(AgentMode(new_m), AgentMode(new_int_m))}')
                        
                    if new_node.start_time + Tv>=T: # if the time of the current simulation + start_time is at or above total time, don't add
                        continue
                    queue.append(new_node)

        trace.nodes = trace._get_all_nodes(trace.root)
        print(f'Verification time: {time.perf_counter()-start}')

    def run_button_clicked(self):
        self.web_view.page().runJavaScript("getPlanePositions();", self.handle_positions)

    def start_animation(self, x1, y1, z1, r1, x2, y2, z2, r2):
        """Starts a background thread to move the sphere."""
        if hasattr(self, 'thread') and self.thread:
            self.thread.join()
        self.plotter.clear()

        self.thread = threading.Thread(target=self.run_verse, args=[x1, y1, z1, r1, x2, y2, z2, r2], daemon=True)
        self.thread.start()

    def handle_positions(self, positions, scaling=1):
        def position_to_int(pos):
            # Handle case where positions might be empty
            try:
                x = float(pos['x'].replace('px', ''))
                y = float(pos['y'].replace('px', ''))
                s = float(pos['size'].replace('px', ''))
                # return (x/scaling, y/scaling, s/scaling)
                return ( 6*(x)+48 , 6*(350-y) - 48, s/.3)
            except (ValueError, KeyError):
                print('Error')
                return (0, 0, 0)
                
        alt1 = self.blue_slider.value() / 10  # Scale as needed
        alt2 = self.red_slider.value() / 10  # Scale as needed
        # positions is a JavaScript object containing the current positions of the planes
        print("Positions of planes:", positions)

        # Process the positions and use them in the 3D visualization or other tasks
        self.plane1_position = positions['plane1']
        self.plane2_position = positions['plane2']

        x1, y1, s1 = (position_to_int(self.plane1_position))
        x2, y2, s2 = (position_to_int(self.plane2_position))

        self.start_animation(x1, y1, alt1, s1, x2, y2, alt2, s2)

    def _set_python_bridge(self, result):
        """Sets the python bridge object in JS"""
        self.web_view.page().runJavaScript("window.pyQtApp = pyQtApp;", self._set_python_bridge)

    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        # Keep side tab in position relative to the overlay
        self.update_side_tab_position()

# Run the Qt Application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    main_window = MainWindow()
    main_window.showMaximized()
    
    sys.exit(app.exec())