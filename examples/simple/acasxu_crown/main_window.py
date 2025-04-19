from verse.plotter.plotter3D import *

import sys
import numpy as np
import os
import json
import pyvistaqt as pvqt
import ast
import re

from ui_components import StyledButton, SvgPlaneSlider, OverlayTab, RightOverlayTab, RightOverlay


from PyQt6.QtCore import Qt, QObject, pyqtSlot, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, 
    QSizePolicy, QLabel, QLineEdit, QTextEdit,
)
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtWebEngineWidgets import QWebEngineView
from ui_components import StyledButton, SvgPlaneSlider, OverlayTab, RightOverlayTab,RightInfoPanel
import pyvistaqt as pvqt

from PyQt6.QtWebChannel import QWebChannel

from verse_bridge_all import VerseBridge
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

    def setup_overlay(self):
        """Setup the overlay container and its components"""
        self.overlay_container = QWidget(self.main_widget)
        self.overlay_container.setGeometry(0, 0, 480, 500)
        self.overlay_container.setStyleSheet("background-color: #b7b7b7; border: 2px solid #616161; opacity: 0.8")


        
        # Add text field for file loading
        self.file_label = QLabel("Load From File:", self.overlay_container)
        self.file_label.setGeometry(10, 450, 100, 25)
        self.file_label.setStyleSheet("color: black; font-weight: bold; border:0px")
        
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

        # Add text field for initial set
        self.initial_set_label = QLabel("Initial Set:", self.overlay_container)
        self.initial_set_label.setGeometry(10, 480, 100, 25)
        self.initial_set_label.setStyleSheet("color: black; font-weight: bold; border:0px")
        
        self.initial_set_input = QLineEdit(self.overlay_container)
        self.initial_set_input.setGeometry(110, 480, 240, 25)
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
        
        # Setup an empty slider container - will add sliders dynamically when plane is selected
        self.setup_sliders()
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
        """Update the yaw value for an agent"""
        if agent_id in self.agents:
            self.agents[agent_id]['yaw'] = yaw_radians
            # You might want to update the 3D visualization here
            #print(f"Updated yaw for {agent_id} to {yaw_radians} radians")

    def update_agent_altitude(self, agent_id, value):
        """Update the altitude value for an agent"""
        #print(300- 3*value-15)
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
        button_width = 170  # Half the width of the original button
    
        self.run_button = StyledButton("Verify", self.overlay_container)
        self.run_button.setGeometry(10, 365, button_width, 30)
        self.run_button.clicked.connect(self.run_button_clicked)
        
        self.run_simulate_button = StyledButton("Simulate", self.overlay_container)
        self.run_simulate_button.setGeometry(10 + button_width + 10, 365, button_width, 30)  # Position next to first button with a small gap
        self.run_simulate_button.clicked.connect(self.run_simulate_clicked)
        
        # Add/Remove plane buttons
        self.add_plane_button = StyledButton("Add Plane", self.overlay_container)
        self.add_plane_button.setGeometry(10, 400, button_width, 30)
        self.add_plane_button.clicked.connect(self.add_plane)
        
        self.remove_plane_button = StyledButton("Remove Plane", self.overlay_container)
        self.remove_plane_button.setGeometry(10 + button_width + 10, 400, button_width, 30)
        self.remove_plane_button.clicked.connect(self.remove_plane)

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

            # Add this method to the MainWindow class
            def update_agent_yaw(self, agent_id, yaw_radians):
                """Update the yaw value for an agent"""
                if agent_id in self.agents:
                    self.agents[agent_id]['yaw'] = yaw_radians
                    # You might want to update the 3D visualization here
                    #print(f"Updated yaw for {agent_id} to {yaw_radians} radians")
            
        
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
            'init_set': '' 
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
                
                # Display agent information in the right panel
                # altitude = self.agents[plane_id].get('altitude', 50)
                # x_pos = self.agents[plane_id].get('x', 0)
                # y_pos = self.agents[plane_id].get('y', 0)
            
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
                    'size': size_px,
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
                print("planeSelected called with:", plane_id)  # Add debug print
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
        self.status_text_box.setGeometry(10, self.height() - 30, 350, 100)
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
        self.handle_inputs(True)

    def run_simulate_clicked(self):
        """Callback for the Run Simulate button (simulation mode)"""
        self.handle_inputs(False)



    def handle_inputs(self, verify):
        # Check if file input has content
        file_path = self.file_input.text().strip()
        if file_path:
            # If file specified, load boxes first
            self.plotter.clear()
            self.plotter.show_grid(all_edges=True, n_xlabels = 6, n_ylabels = 6, n_zlabels = 6)
            load_and_plot(self.plotter, log_file=file_path)
            #grid_bounds = [-2400, 300, -1100, 600, 0, 1100]
            #self.plotter.show_grid(axes_ranges=grid_bounds)
        else:

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

            if(verify):
                num_sims = 0
            global node_rect_cache
            global node_idx
            node_rect_cache ={}
            node_idx =0

            if save_to_file:
                with open(log_file, "w") as f:
                    f.write("")  # Optional: just clears the file

            for id in self.agents:
                d = self.agents[id]
                print(d)
                if(d['init_set'] == ''):
                    self.verse_bridge.updatePlane( id= id, x =d['x'],y =d['y'], z=d['altitude'], radius= d['size'], pitch=np.pi/3,yaw=d['yaw'], v=100, agent_type="Car", dl ="controller_3d.py")
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
                                print(elem)
                                # Use eval with numpy context
                                value = eval(elem, {"np": np})
                                parsed_row.append(value)
                        result.append(parsed_row)
                    self.verse_bridge.updatePlane( id= id, agent_type="Car", dl ="controller_3d.py")
                    self.verse_bridge.addInitialSet(self.active_agent_id, result)


            #self.verse_bridge.run_verse(x_dim=x_dim, y_dim=y_dim, z_dim=z_dim, time_horizon=time_horizon, time_step=time_step, num_sims=num_sims, verify=verify)
            self.run_in_thread( x_dim, y_dim, z_dim, time_horizon, time_step, num_sims, verify)
            #plotRemaining(self.plotter, verify)

            #self.update_status("Finished Running Verse")

    def run_in_thread(self, x_dim, y_dim, z_dim, time_horizon, time_step, num_sims, verify):
        # Disable UI elements if needed
        # self.some_button.setEnabled(False)
        
        # Create and set up the worker
        self.verse_worker = VerseWorker(
            self.verse_bridge, x_dim, y_dim, z_dim, time_horizon, time_step, num_sims, verify
        )
        
        # Connect signals
        self.verse_worker.finished.connect(self.on_verse_calculation_complete)
        
        # Start the thread
        self.verse_worker.start()

    def on_verse_calculation_complete(self):
        # This will be called when the thread finishes
        plotRemaining(self.plotter, self.verse_worker.verify)
        
        # Re-enable UI elements
        # self.some_button.setEnabled(True)
        
        # Clean up
        self.verse_worker.deleteLater()


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