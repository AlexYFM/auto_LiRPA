from verse.plotter.plotter3D import *
import verse.plotter
import sys
import numpy as np
import os
import json
import pyvistaqt as pvqt
import time
from ui_components import StyledButton, SvgPlaneSlider, OverlayTab, RightOverlayTab, RightOverlay
from threading import Thread

from PyQt6.QtCore import Qt, QObject, pyqtSlot, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, 
    QSizePolicy, QLabel, QLineEdit, QTextEdit,QComboBox, QSlider
)
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtWebEngineWidgets import QWebEngineView
from ui_components import StyledButton, SvgPlaneSlider, OverlayTab, RightOverlayTab
import pyvistaqt as pvqt

from PyQt6.QtWebChannel import QWebChannel

from verse_bridge_all import VerseBridge
class PlotterWorker(QThread):
    finished = pyqtSignal()
    progress = pyqtSignal(int)  # Optional: for progress updates

    def __init__(self, ax, log_file):
        super().__init__()
        self.log_file = log_file
        self.ax= ax
        

    def run(self):
        # Run the computation in the thread
        preprocess_file(self.ax, self.log_file) 
        
        self.finished.emit()


class VerseWorker(QThread):
    finished = pyqtSignal()
    progress = pyqtSignal(int)  # Optional: for progress updates

    def __init__(self, verse_bridge, x_dim, y_dim, z_dim, time_horizon, time_step, num_sims, verify):
        super().__init__()
        self.verse_bridge = verse_bridge
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.time_horizon = time_horizon
        self.time_step = time_step
        self.num_sims = num_sims
        self.verify = verify

    def run(self):
        # Run the computation in the thread
        self.verse_bridge.run_verse(
            x_dim=self.x_dim, 
            y_dim=self.y_dim, 
            z_dim=self.z_dim, 
            time_horizon=self.time_horizon, 
            time_step=self.time_step, 
            num_sims=self.num_sims, 
            verify=self.verify
        )
        self.finished.emit()
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plane Visualization")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize variables
        self.overlay_visible = True
        self.right_overlay_visible = True
        self.active_slider = None
        # Dictionary to store agent information
        self.agents = {}
        self.total_agent_num = 0
        # Active agent ID (the currently selected plane)
        self.active_agent_id = None
        
        # Setup main UI
        self.setup_main_ui()
        # Setup left overlay
        self.setup_status_text_box()

        self.setup_overlay()
        # Setup right overlay
        self.setup_right_overlay()
        # Setup web view
        self.setup_web_view()
        # Setup side tabs
        self.setup_side_tab()
        self.setup_right_tab()
        
        # Make sure overlays are visible
        self.overlay_container.raise_()
        self.overlay_container.show()
        
        self.right_overlay_container.raise_()
        self.right_overlay_container.show()
        
        self.side_tab.raise_()
        self.right_side_tab.raise_()

        self.verse_worker = None
        self.thread = None
        
    def setup_main_ui(self):
        """Setup the main UI components"""
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = pvqt.QtInteractor()
        self.plotter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.main_layout.addWidget(self.plotter.interactor)
        self.verse_bridge = VerseBridge(self.plotter)

    def setup_timeline_slider(self):
        """Setup the timeline slider at the bottom of the screen"""
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal, self.main_widget)
        
        # Calculate position to span across the bottom of the screen
        slider_height = 30
        slider_y_position = self.height() - slider_height - 10  # 10px padding from bottom
        
        # Make it extend across the screen, leaving space for status box
        self.timeline_slider.setGeometry(350, slider_y_position, self.width() - 350, slider_height)
        
        # Style the slider
        self.timeline_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
                margin: 2px 0;
            }
            
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
            
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3498db, stop:1 #2980b9);
            }
        """)
        
        # Connect the slider to a function to update visualization based on time
        self.timeline_slider.valueChanged.connect(self.on_timeline_changed)
        
        # Initially hide the slider (will be shown when needed)
        self.timeline_slider.hide()

    def on_timeline_changed(self, value):
        """Handle timeline slider value changes"""
        # Update the visualization to show the state at the selected time
        
        # Call the existing visualization update method for the selected time
        # This is a placeholder - call your actual function to update the visualization
        # for the specified time step
        load_and_plot(self.plotter, target_time= float(value)/100)


    def setup_overlay(self):
        """Setup the overlay container and its components"""
        self.overlay_container = QWidget(self.main_widget)
        self.overlay_container.setGeometry(0, 0, 480, 600)
        self.overlay_container.setStyleSheet("background-color: #b7b7b7; border: 2px solid #616161; opacity: 0.8")
        
        # Setup an empty slider container - will add sliders dynamically when plane is selected
        self.setup_sliders()
        
        # Create agent settings container that includes initial set, agent type, and decision logic
        self.agent_settings_container = QWidget(self.overlay_container)
        self.agent_settings_container.setGeometry(10, 400, 460, 115)  # Made taller to fit all three elements
        self.agent_settings_container.setStyleSheet("background-color: #a0a0a0; border: 1px solid #616161; border-radius: 4px;")
        
        # Add text field for initial set inside the container
        self.initial_set_label = QLabel("Initial Set:", self.agent_settings_container)
        self.initial_set_label.setGeometry(10, 10, 100, 25)
        self.initial_set_label.setStyleSheet("color: black; font-weight: bold; border: 0px")
        
        self.initial_set_input = QLineEdit(self.agent_settings_container)
        self.initial_set_input.setGeometry(120, 10, 330, 25)
        self.initial_set_input.setStyleSheet("""
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
        self.initial_set_input.textChanged.connect(self.update_initial_set)
        
        # Add agent type dropdown inside the container
        self.agent_type_label = QLabel("Agent Type:", self.agent_settings_container)
        self.agent_type_label.setGeometry(10, 45, 100, 25)
        self.agent_type_label.setStyleSheet("color: black; font-weight: bold; border: 0px")
        
        self.agent_type_dropdown = QComboBox(self.agent_settings_container)
        self.agent_type_dropdown.setGeometry(120, 45, 330, 25)
        self.agent_type_dropdown.addItems(["Car", "NPC"])
        self.agent_type_dropdown.setStyleSheet("""
            QComboBox {
                background-color: white;
                border: 1px solid #3498db;
                border-radius: 4px;
                padding: 2px 4px;
            }
            QComboBox:focus {
                border: 2px solid #2980b9;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: right;
                width: 20px;
                border-left: 1px solid #3498db;
            }
        """)
        self.agent_type_dropdown.currentTextChanged.connect(self.update_agent_type)
        
        # Add dropdown for decision logic inside the container
        self.decision_logic_label = QLabel("Decision Logic:", self.agent_settings_container)
        self.decision_logic_label.setGeometry(10, 80, 100, 25)
        self.decision_logic_label.setStyleSheet("color: black; font-weight: bold; border: 0px")
        
        self.decision_logic_dropdown = QComboBox(self.agent_settings_container)
        self.decision_logic_dropdown.setGeometry(120, 80, 330, 25)
        self.decision_logic_dropdown.addItems([
            "controller_3d.py", "None"
        ])
        self.decision_logic_dropdown.setStyleSheet("""
            QComboBox {
                background-color: white;
                border: 1px solid #3498db;
                border-radius: 4px;
                padding: 2px 4px;
            }
            QComboBox:focus {
                border: 2px solid #2980b9;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: right;
                width: 20px;
                border-left: 1px solid #3498db;
            }
        """)
        self.decision_logic_dropdown.currentTextChanged.connect(self.update_decision_logic)
        
        # Add text field for file loading - below the agent settings container
        self.file_label = QLabel("Load From File:", self.overlay_container)
        self.file_label.setGeometry(10, 525, 100, 25)
        self.file_label.setStyleSheet("color: black; font-weight: bold; border: 0px")
        
        self.file_input = QLineEdit(self.overlay_container)
        self.file_input.setGeometry(120, 525, 350, 25)
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
        #self.file_input.setText("boxes_multi.txt")
        
        # Setup buttons 
        self.setup_buttons()

    def setup_sliders(self):
        """Setup container for altitude sliders with styling"""
        self.slider_container = QWidget(self.overlay_container)
        self.slider_container.setGeometry(370, 10, 100, 350)
        self.slider_container.setStyleSheet("background-color: gray;")

        # Create a label to show "Altitude" text
        self.altitude_label = QLabel("Altitude", self.slider_container)
        self.altitude_label.setGeometry(20, 3, 60, 20)
        self.altitude_label.setStyleSheet("color: black; font-weight: bold; border:0px solid #3498db")
        
        # We'll add sliders dynamically when planes are selected

    def update_initial_set(self):
        """Update the initial set value for the currently selected agent"""
        if self.active_agent_id and self.active_agent_id in self.agents:
            initial_set = self.initial_set_input.text().strip()
 
            self.agents[self.active_agent_id]['init_set'] =  initial_set
            # Call the bridge method to update the initial set
            self.update_status(f"Updated initial set for {self.active_agent_id}: {initial_set}")

    def update_agent_type(self, agent_type):
        """Update the agent type for the currently selected agent"""
        if self.active_agent_id and self.active_agent_id in self.agents:
            self.agents[self.active_agent_id]['agent_type'] = agent_type
            self.update_status(f"Updated agent type for {self.active_agent_id}: {agent_type}")

    def update_decision_logic(self, decision_logic):
        """Update the decision logic for the currently selected agent"""
        if self.active_agent_id and self.active_agent_id in self.agents:
            self.agents[self.active_agent_id]['dl'] = decision_logic
            self.update_status(f"Updated decision logic for {self.active_agent_id}: {decision_logic}")

    def create_slider_for_agent(self, agent_id):
        """Create a slider for the specified agent"""
        # Remove existing slider if any
        if self.active_slider is not None:
            self.active_slider.deleteLater()
            self.active_slider = None
        
        # Get agent information
        agent = self.agents.get(agent_id)
        if not agent:
            return
        # Create new slider with agent's color
        color = agent.get('color', '#3498db')  # Default to blue if color not set
        plane_svg = f"plane_{color.replace('#', '')}.svg"  # Use color in SVG filename
        
        self.active_slider = SvgPlaneSlider(plane_svg, self.slider_container)
        self.active_slider.setGeometry(30, 30, 30, 280)
        self.active_slider.setValue(agent.get('altitude', 50))

        self.active_slider.valueChanged.connect(lambda value: self.update_agent_altitude(agent_id, value))
        self.active_slider.show()

    def update_agent_yaw(self, agent_id, yaw_radians):
        yaw_radians -= np.pi/2
        # Normalize yaw to be within [-pi, pi]
        if(np.pi < yaw_radians):
            yaw_radians = yaw_radians - 2*np.pi
        if(yaw_radians < -np.pi):
            yaw_radians = yaw_radians + 2*np.pi
        """Update the yaw value for an agent"""
        if agent_id in self.agents:

            self.agents[agent_id]['yaw'] = yaw_radians
            # You might want to update the 3D visualization here
            #print(f"Updated yaw for {agent_id} to {yaw_radians} radians")

    def update_agent_altitude(self, agent_id, value):
        """Update the altitude value for an agent"""
        if agent_id in self.agents:
            self.agents[agent_id]['altitude'] = value
            # Update any necessary visuals or data

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
            self.right_overlay_container.setGeometry(self.width() - overlay_width, 0, overlay_width, 500)
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
            self.right_overlay_container.setGeometry(self.width() - overlay_width, 0, overlay_width, 500)

        self.status_text_box.setGeometry(0, self.height() - 30, 350, 30)
        if hasattr(self, 'timeline_slider'):
                slider_height = 30
                slider_y_position = self.height() - slider_height - 10
                self.timeline_slider.setGeometry(350, slider_y_position, self.width() - 350, slider_height)
        # Update right tab position
        if hasattr(self, 'right_side_tab'):
            self.update_right_tab_position()

    def update_side_tab_position(self):
            """Update the position of the side tab based on overlay visibility"""
            if self.overlay_visible:
                self.side_tab.setGeometry(480, 150, 40, 80)
            else:
                self.side_tab.setGeometry(0, 150, 40, 80)

    def setup_right_overlay(self):
        """Setup the right side overlay container and its components"""
        self.right_overlay_container = RightOverlay(self.main_widget)

    def setup_right_tab(self):
        """Setup the side tab for showing/hiding right overlay"""
        self.right_side_tab = RightOverlayTab(self.main_widget)
        self.right_side_tab.on_click_callback = self.toggle_right_overlay
        self.update_right_tab_position()
        self.right_side_tab.raise_()

    def setup_buttons(self):
        """Setup the control buttons"""
        # Add/Remove plane buttons - above the agent settings
        button_width = 170
        
        self.add_plane_button = StyledButton("Add Plane", self.overlay_container)
        self.add_plane_button.setGeometry(10, 365, button_width, 30)
        self.add_plane_button.clicked.connect(self.add_plane)
        
        self.remove_plane_button = StyledButton("Remove Plane", self.overlay_container)
        self.remove_plane_button.setGeometry(190, 365, button_width, 30)
        self.remove_plane_button.clicked.connect(self.remove_plane)
        
        # Verify, Simulate, and Stop buttons at the bottom
        button_width = 150  # Adjusted for three buttons
        
        self.run_button = StyledButton("Verify", self.overlay_container)
        self.run_button.setGeometry(10, 560, button_width, 30)
        self.run_button.clicked.connect(self.run_button_clicked)
        
        self.run_simulate_button = StyledButton("Simulate", self.overlay_container)
        self.run_simulate_button.setGeometry(10 + button_width + 5, 560, button_width, 30)
        self.run_simulate_button.clicked.connect(self.run_simulate_clicked)
        
        self.stop_button = StyledButton("Stop", self.overlay_container, color='red')
        self.stop_button.setGeometry(10 + 2 * (button_width + 5), 560, button_width, 30)
        self.stop_button.clicked.connect(self.stop)

    def setup_web_view(self):
        """Setup the web view for the visualization with console support"""
        self.web_view = QWebEngineView(self.overlay_container)
        self.web_view.setGeometry(10, 10, 350, 350)
        self.web_view.setHtml(self.get_web_content())
        channel = QWebChannel(self.web_view.page())
    
        class Bridge(QObject):
            @pyqtSlot('QJsonObject')
            def savePositions(self, positions_json):
                self.parent().save_plane_positions(positions_json)
            
            @pyqtSlot(str)
            def planeSelected(self, plane_id):
                #print("Bridge: planeSelected called with:", plane_id)
                self.parent().on_plane_selected(plane_id)
            @pyqtSlot(str, float)
            def saveYawAngle(self, plane_id, yaw_radians):
                #print(f"Saving yaw angle for {plane_id}: {yaw_radians} radians")
                self.parent().update_agent_yaw(plane_id, yaw_radians)

            
        
        self.js_bridge = Bridge()
        self.js_bridge.setParent(self)
        channel.registerObject("interop", self.js_bridge)
        self.web_view.page().setWebChannel(channel)
        
        # Now load the HTML content after the channel is set up
        self.web_view.setHtml(self.get_web_content())
        
        # Configure settings
        self.web_view.page().settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        self.web_view.page().settings().setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanAccessClipboard, True)
        
        

    def add_plane(self):
        """Add a new plane to the visualization"""
        self.total_agent_num+=1
        colors = ["#007BFF", "#FF0000", "#00AA00", "#AA00AA", "#AAAA00"]
        color_idx = (len(self.agents)) % len(colors)
        color = colors[color_idx]
        plane_id = f"plane_{color}_{self.total_agent_num}"

        self.agents[plane_id] = {
            'id': plane_id,
            'color': color,
            'x': 10,
            'y': 10,
            'altitude': 50,
            'size': 20,
            'yaw':0,
            'init_set': '',
            'agent_type': 'Car', 
            'dl': 'controller_3d.py'  
        }
        
        js_code = f"""
        (function() {{
            let container = document.getElementById('plane-container');
            if (!container) return;
            
            let plane = document.createElement('div');
            plane.id = '{plane_id}';
            plane.className = 'draggable';
            plane.setAttribute('draggable', 'true');
            plane.setAttribute('ondragstart', 'dragStart(event)');
            
            let svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            svg.setAttribute('viewBox', '0 0 24 24');
            svg.setAttribute('width', '100%');
            svg.setAttribute('height', '100%');
            svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
            
            let path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            path.setAttribute('d', 'M21,16V14L13,9V3.5A1.5,1.5,0,0,0,11.5,2A1.5,1.5,0,0,0,10,3.5V9L2,14V16L10,13.5V19L8,20.5V22L11.5,21L15,22V20.5L13,19V13.5Z');
            path.setAttribute('fill', '{color}');
            
            svg.appendChild(path);
            plane.appendChild(svg);
            
            plane.style.left = '10px';
            plane.style.top = '10px';
            plane.style.width = '20px';
            plane.style.height = '20px';
            plane.style.lineHeight = '20px';
            plane.style.borderRadius = '10px';
            
            container.appendChild(plane);
            
            // Initialize plane size
            if (typeof planesSizes !== 'undefined') {{
                planesSizes['{plane_id}'] = 20;
            }}
            
            // Add event listeners
            plane.addEventListener('click', planeClick);
            const children = plane.querySelectorAll('*');
            children.forEach(child => {{
                child.addEventListener('click', planeClick);
            }});
        }})();
        """
        
        self.web_view.page().runJavaScript(js_code)

    def remove_plane(self):
        """Remove the selected plane"""
        if not self.active_agent_id:
            self.update_status("No plane selected for removal")
            return
            
        # Remove from agents dictionary
        if self.active_agent_id in self.agents:
            del self.agents[self.active_agent_id]
            
        # Remove from web view
        js_code = f"""
        var plane = document.getElementById('{self.active_agent_id}');
        if (plane) {{
            plane.parentNode.removeChild(plane);
            
            // Clear selection if it was the selected plane
            if (selectedPlane && selectedPlane.id === '{self.active_agent_id}') {{
                removeRotationHandle();

                selectedPlane = null;
            }}
        }}
        """
        self.web_view.page().runJavaScript(js_code)
        
        # Remove active slider
        if self.active_slider:
            self.active_slider.deleteLater()
            self.active_slider = None
            
        # Clear active agent ID
        self.update_status(f"Removed {self.active_agent_id}")

        self.verse_bridge.removePlane(self.active_agent_id)

        self.active_agent_id = None

    def on_plane_selected(self, plane_id):
        """Handle plane selection from JavaScript"""
        self.active_agent_id = plane_id
        
        # Create/update slider for the selected plane
        if plane_id in self.agents:
            self.create_slider_for_agent(plane_id)

            init_set = self.agents[plane_id].get('init_set', '')
            self.initial_set_input.setText(init_set)
            
            self.update_status(f"Selected plane: {plane_id}")

            # Update agent type dropdown
            agent_type = self.agents[plane_id].get('agent_type', 'Car')
            index = self.agent_type_dropdown.findText(agent_type)
            if index >= 0:
                self.agent_type_dropdown.setCurrentIndex(index)
            
            # Update decision logic dropdown
            decision_logic = self.agents[plane_id].get('dl', 'controller_3d.py')
            index = self.decision_logic_dropdown.findText(decision_logic)
            if index >= 0:
                self.decision_logic_dropdown.setCurrentIndex(index)
                
            
    def save_plane_positions(self, positions):
        """Save the plane positions from JavaScript to the agents dictionary."""

        for plane_id, position in positions.items():
            if plane_id in self.agents:
                # Convert QJsonValue to a real Python dict
                position_data = position.toVariant()
                # Now you can safely use .get()
                x_px = int(position_data.get('x', '0px').replace('px', ''))
                y_px = int(position_data.get('y', '0px').replace('px', ''))
                size_px = int(position_data.get('size', '20px').replace('px', ''))

                # Rotation: ensure it's a float
                rotation = float(position_data.get('rotation', self.agents[plane_id].get('yaw', 0)))

                # Update the agent's data
                self.agents[plane_id].update({
                    'x': 6*(x_px)+48,
                    'y': 6*(350-y_px) - 48,
                    'size': size_px/2,
                    'yaw': rotation,
                })
                
    def on_web_view_loaded(self):
        """Handle web view load events"""

        self.web_view.page().runJavaScript("""
            // Don't need to define both interop and pyQtApp
            // Just define what we'll use after the channel is established
            console.log("Setting up bridge connections...");
        """)

        # Enable necessary web settings
        self.web_view.page().settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        self.web_view.page().settings().setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanAccessClipboard, True)
        
        # Create and register the web channel
        channel = QWebChannel(self.web_view.page())
        
        class Bridge(QObject):
            @pyqtSlot(str)
            def savePositions(self, positions_json):
                self.parent().save_plane_positions(positions_json)
            
            @pyqtSlot(str)
            def planeSelected(self, plane_id):
                #print("planeSelected called with:", plane_id)  # Add debug print
                self.parent().on_plane_selected(plane_id)
        self.js_bridge = Bridge()
        self.js_bridge.setParent(self)
        channel.registerObject("interop", self.js_bridge)
        self.web_view.page().setWebChannel(channel)

    def get_web_content(self):
        """Get the HTML content for the web view with support for dynamic planes"""
        script_dir = os.path.realpath(os.path.dirname(__file__))

        with open( os.path.join(script_dir, "web_content.html" ),  'r') as file:  # r to open file in READ mode
            return file.read()
         
        
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

    def setup_status_text_box(self):
        """Setup the status text box in the bottom left"""
        self.status_text_box = QTextEdit(self.main_widget)
        self.status_text_box.setGeometry(0, self.height() - 30, 350, 30)
        self.status_text_box.setReadOnly(True)
        self.status_text_box.setStyleSheet("""
            QTextEdit {
                background-color: rgba(255,255,255,1);
                color: #000;
                font-size: 12px;
            }
        """)
        self.update_status("Initialized and ready")

    def update_status(self, message):
        self.status_text_box.setText(f"{message}")

    def run_button_clicked(self):
        """Callback for the Run button (normal mode)"""
        # self.web_view.page().runJavaScript("getPlanePositions();", 
        #                                 lambda positions: self.handle_inputs(positions, True))
        self.stop()
        self.plotter.clear()

        self.handle_inputs(True)

    def run_simulate_clicked(self):
        """Callback for the Run Simulate button (simulation mode)"""
        self.stop()
        self.plotter.clear()

        self.handle_inputs(False)

    def stop(self):
        if(self.verse_worker):
            self.verse_worker.terminate()
            self.verse_worker.wait()
        if(self.thread):
            self.thread.join()
            self.thread = None
        self.plotter.clear()
        self.update_status("Stopped Verse")
    def get_max_time_from_data(self, file_path, verify):
        """Get the maximum time value from the file (last value of last line)"""
        max_time = 0
        
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                if lines:
                    # Get the last line
                    if(verify):
                        last_line = lines[-1].strip()
                    else:
                        last_line = lines[-3].strip()
                    if last_line:
                        # Split the line and get the last value
                        parts = last_line.split()
                        if parts:
                            try:
                                max_time = float(parts[0])
                            except ValueError:
                                # If the first value isn't a time value, try the last value
                                try:
                                    max_time = float(parts[-1])
                                except ValueError:
                                    self.update_status("Could not determine max time from file")
        except Exception as e:
            self.update_status(f"Error reading file: {str(e)}")
        
        return max_time


    def handle_inputs(self, verify):
        # Check if file input has content
        file_path = self.file_input.text().strip()
        if file_path:
            # If file specified, load boxes
            self.stop()

            self.plotter.show_grid(all_edges=True)


            max_time = self.get_max_time_from_data(file_path, verify)
            print(f"Max time from file: {max_time}")
            
            # Setup the slider with the appropriate range
            if not hasattr(self, 'timeline_slider'):
                self.setup_timeline_slider()

            self.timeline_slider.setMinimum(0)
            self.timeline_slider.setMaximum(int(max_time)*100)
            self.timeline_slider.setValue(int(max_time)*100)  # Start at end
            self.timeline_slider.setTickInterval(1)  # Single keyboard step is 1/100th of a time unit
            self.timeline_slider.setPageStep(100 // 10) 

            #self.thread = Thread(target=preprocess_file, args = [self.plotter, file_path])


            #self.run_plotter_in_thread( file_path)
            preprocess_file(self.plotter, file_path)

            self.timeline_slider.show()
            self.timeline_slider.raise_()
            verse.plotter.plotter3D.load_time = float(max_time)



        else:
            if  hasattr(self, 'timeline_slider'):

                self.timeline_slider.hide()


            if os.path.exists('plotter_config.json'):
                with open('plotter_config.json', 'r') as f:
                    config = json.load(f)
                    x_dim = config['x_dim']
                    y_dim = config['y_dim']
                    z_dim = config['z_dim']
                    time_horizon = config['time_horizon']
                    time_step = config['time_step']
                    num_sims = config['num_sims']
                    save_to_file = config['save']
                    log_file = config['log_file']
            verse.plotter.plotter3D.node_rect_cache ={}
            verse.plotter.plotter3D.node_idx =0

            if save_to_file:
                with open(log_file, "w") as f:
                    f.write("")  # Optional: just clears the file

            for id in self.agents:
                d = self.agents[id]
                if(d['init_set'] == ''):
                    self.verse_bridge.updatePlane( id= id, x =d['x'],y =d['y'], z= 300 - 3*d['altitude']-10, radius= d['size'], pitch=0,yaw=d['yaw'], v=100, agent_type=d["agent_type"], dl = (None if d["dl"] == "None" else d["dl"]) )
                else:

                    initial_set_str = d['init_set']
                    if initial_set_str.startswith('[') and initial_set_str.endswith(']'):
                        # Remove the outer brackets
                        initial_set_str = initial_set_str[1:-1]
                    
                    # Split into rows
                    rows = initial_set_str.split('], [')
                    
                    # Clean up the rows
                    rows = [row.replace('[', '').replace(']', '') for row in rows]
                    
                    result = []
                    for row in rows:
                        # Split the row into elements
                        elements = row.split(',')
                        # Parse each element with eval (with numpy context)
                        parsed_row = []
                        for elem in elements:
                            elem = elem.strip()
                            if elem:
                                # Use eval with numpy context
                                value = eval(elem, {"np": np})
                                parsed_row.append(value)
                        result.append(parsed_row)

                    self.verse_bridge.updatePlane( id= id, agent_type=d["agent_type"], dl = (None if d["dl"] == "None" else d["dl"]) )
                    self.verse_bridge.addInitialSet(id, result)

            self.plotter.clear()
            self.run_in_thread( x_dim, y_dim, z_dim, time_horizon, time_step, num_sims, verify)


    def run_in_thread(self, x_dim, y_dim, z_dim, time_horizon, time_step, num_sims, verify):
        
        # Create and set up the worker
        self.verse_worker = VerseWorker(
            self.verse_bridge, x_dim, y_dim, z_dim, time_horizon, time_step, num_sims, verify
        )
        
        # Connect signals
        self.verse_worker.finished.connect(self.on_verse_calculation_complete)
        
        # Start the thread
        self.verse_worker.start()
    def run_plotter_in_thread(self, log_file):
       
        
        # Create and set up the worker
        self.plotter_worker = PlotterWorker(
            self.plotter, log_file
        )
        
        # Connect signals
        self.plotter_worker.finished.connect(self.on_plotter_complete)
        
        # Start the thread
        self.plotter_worker.start()

    def on_plotter_complete(self):
        
        self.plotter_worker.deleteLater()
        self.plotter_worker = None

    def on_verse_calculation_complete(self):
        # This will be called when the thread finishes
        plotRemaining(self.plotter, self.verse_worker.verify)
        
        # Re-enable UI elements
        # self.some_button.setEnabled(True)
        
        # Clean up
        self.verse_worker.deleteLater()
        self.verse_worker = None


        # def _set_python_bridge(self, result):
        #     """Sets the python bridge object in JS"""
        #     self.web_view.page().runJavaScript("window.pyQtApp = pyQtApp;", self._set_python_bridge)


# Run the Qt Application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    main_window = MainWindow()
    main_window.showMaximized()
    
    sys.exit(app.exec())