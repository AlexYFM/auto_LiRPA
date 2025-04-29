from PyQt6.QtWidgets import (QVBoxLayout, QWidget, 
    QPushButton, QSlider,
    QLabel, QFrame,QSpinBox,QCheckBox, QStyleOptionSlider, QStyle, QDoubleSpinBox, QLineEdit
)
from PyQt6.QtGui import QPainter, QColor, QPen, QPainterPath
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtSvg import QSvgRenderer
import os
import json
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



class RightOverlayTab(QFrame):
    """Side tab for showing/hiding right overlay with rounded corners and smaller size"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(15, 100)  # Smaller size
        self.setStyleSheet("""
            QFrame {
                background-color: #D477B1;  /* Different color from left panel */
                border-top-left-radius: 15px;
                border-bottom-left-radius: 15px;
                border: none;
            }
            QFrame:hover {
                background-color: #b92980;
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
        painter.setPen(QPen(Qt.GlobalColor.black, 2))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get widget dimensions
        width = self.width()
        height = self.height()
        
        # Draw triangular line (two connected lines forming a triangle-like shape)
        # For right panel, make the arrow point left
        path = QPainterPath()
        path.moveTo(width * 0.3, height * 0.2)
        path.lineTo(width * 0.7, height * 0.5)
        path.lineTo(width * 0.3, height * 0.8)
        
        painter.drawPath(path)
        
        # Restore the previous state
        painter.restore()

    def mousePressEvent(self, event):
        # Call the callback function
        if hasattr(self, 'on_click_callback') and self.on_click_callback:
            self.on_click_callback()
        super().mousePressEvent(event)
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

class StyledButton(QPushButton):
    """Custom styled button with modern look"""
    def __init__(self, text, parent=None, color='#3498db'):
        super().__init__(text, parent)
        self.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #2980b9;
            }}
            QPushButton:pressed {{
                background-color: #2472a4;
            }}
            QPushButton:disabled {{
                background-color: #bdc3c7;
            }}
            """
        )

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



class RightInfoPanel(QFrame):
    """Right side information panel with modern styling"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: rgba(80, 44, 62, 0.9);  /* Different color from left panel */
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
                color: #D477B1;  /* Different color from left panel */
                margin-bottom: 5px;
            }
        """)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        title = QLabel("Configuration")
        title.setObjectName("title")
        layout.addWidget(title)
        
        layout.addStretch()



class RightOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set up dimensions
        overlay_width = 380
        overlay_height = 500  # Increased height
        self.setGeometry(parent.width() - overlay_width, 0, overlay_width, overlay_height)
        
        # Initialize components
        self.setup_ui()
        
        # Load config on startup
        self.load_config()
        
        self.visible = True

    def setup_ui(self):
        """Create and arrange the widgets inside the overlay"""
        # Info Panel
        self.right_info_panel = RightInfoPanel(self)
        self.right_info_panel.setGeometry(20, 10, 340, 450)

        # Labels and Inputs
        self.dimensions_label = self.create_label("Dimensions:", 30, 110, bold=True)
        
        self.x_dim_label = self.create_label("X:", 110, 110)
        self.x_dim_input = self.create_spinbox(130, 110, 0, 9999, 0)
        
        self.y_dim_label = self.create_label("Y:", 190, 110)
        self.y_dim_input = self.create_spinbox(210, 110, 0, 9999, 1)
        
        self.z_dim_label = self.create_label("Z:", 270, 110)
        self.z_dim_input = self.create_spinbox(290, 110, 0, 9999, 2)

        self.speed_label = self.create_label("Plot Speed:", 30, 150, bold=True)
        self.speed_input = self.create_spinbox(130, 150, 1, 99999, 100)

        self.time_step_label = self.create_label("Time Step:", 30, 70, bold=True)
        self.time_step_input = self.create_doublespinbox(130, 70, 0.01, 10.0, 0.1, 0.1, 2)

        self.time_horizon_label = self.create_label("Time Horizon:", 30, 270, bold=True)
        self.time_horizon_input = self.create_doublespinbox(130, 270, 0.1, 100.0, 5.0, 0.5, 1)

        self.num_sims_label = self.create_label("Number of Simulations:", 30, 310, bold=True)
        self.num_sims_input = self.create_spinbox(180, 310, 0, 500, 0)

        self.node_batching_label = self.create_label("Node Level Batching:", 30, 350, bold=True)
        self.node_batching_checkbox = self.create_checkbox(180, 355)

        self.save_to_file_label = self.create_label("Save to File:", 30, 180, bold=True)
        self.save_to_file_checkbox = self.create_checkbox(130, 185)

        self.log_file_label = self.create_label("Log File:", 30, 225, bold=True)
        self.log_file_input = self.create_lineedit(130, 225, "boxes.txt")

        self.save_config_button = StyledButton("Save Config", self)
        self.save_config_button.setGeometry(120, 400, 150, 30)
        self.save_config_button.clicked.connect(self.save_config)

    def create_label(self, text, x, y, bold=False):
        label = QLabel(text, self)
        label.setGeometry(x, y, 150, 30)
        style = "color: white;" + (" font-weight: bold;" if bold else "")
        label.setStyleSheet(style)
        return label

    def create_spinbox(self, x, y, min_val, max_val, default_val):
        spinbox = QSpinBox(self)
        spinbox.setGeometry(x, y, 50, 30)
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(default_val)
        spinbox.setStyleSheet(self.spinbox_style())
        return spinbox

    def create_doublespinbox(self, x, y, min_val, max_val, default_val, step, decimals):
        doublespinbox = QDoubleSpinBox(self)
        doublespinbox.setGeometry(x, y, 100, 30)
        doublespinbox.setRange(min_val, max_val)
        doublespinbox.setValue(default_val)
        doublespinbox.setSingleStep(step)
        doublespinbox.setDecimals(decimals)
        doublespinbox.setStyleSheet(self.spinbox_style())
        return doublespinbox

    def create_lineedit(self, x, y, default_text):
        lineedit = QLineEdit(self)
        lineedit.setGeometry(x, y, 160, 30)
        lineedit.setText(default_text)
        lineedit.setStyleSheet(self.lineedit_style())
        return lineedit

    def create_checkbox(self, x, y):
        checkbox = QCheckBox(self)
        checkbox.setGeometry(x, y, 20, 20)
        checkbox.setStyleSheet(self.checkbox_style())
        return checkbox

    def spinbox_style(self):
        return """
            QSpinBox, QDoubleSpinBox {
                background-color: white;
                border: 1px solid #D477B1;
                border-radius: 4px;
                padding: 2px 4px;
            }
            QSpinBox:focus, QDoubleSpinBox:focus {
                border: 2px solid #b92980;
            }
        """

    def lineedit_style(self):
        return """
            QLineEdit {
                background-color: white;
                border: 1px solid #D477B1;
                border-radius: 4px;
                padding: 2px 4px;
            }
            QLineEdit:focus {
                border: 2px solid #b92980;
            }
        """

    def checkbox_style(self):
        return """
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
        """

    def load_config(self):
        """Load configuration from a JSON file if it exists"""
        if os.path.exists('plotter_config.json'):
            with open('plotter_config.json', 'r') as f:
                config = json.load(f)
                
            # Update UI elements with loaded values
            self.x_dim_input.setValue(config.get('x_dim', 0))
            self.y_dim_input.setValue(config.get('y_dim', 1))
            self.z_dim_input.setValue(config.get('z_dim', 2))
            self.speed_input.setValue(int(config.get('speed', 100)))
            self.time_step_input.setValue(float(config.get('time_step', 0.1)))
            # Load new fields with default values if not present
            self.time_horizon_input.setValue(float(config.get('time_horizon', 5.0)))
            self.num_sims_input.setValue(int(config.get('num_sims', 0)))
            self.node_batching_checkbox.setChecked(bool(config.get('node_batch', False)))
            self.save_to_file_checkbox.setChecked(bool(config.get('save', False)))
            self.log_file_input.setText(config.get('log_file', 'boxes.txt'))


    def save_config(self):
            """Save the current configuration to a JSON file"""
            config = {
                'x_dim': self.x_dim_input.value(),
                'y_dim': self.y_dim_input.value(),
                'z_dim': self.z_dim_input.value(),
                'speed': self.speed_input.value(),
                "time_step": self.time_step_input.value(),
                'time_horizon': self.time_horizon_input.value(),
                'num_sims': self.num_sims_input.value(),
                'node_batch': self.node_batching_checkbox.isChecked(),
                'save': self.save_to_file_checkbox.isChecked(),
                'log_file': self.log_file_input.text()
            }
            with open('plotter_config.json', 'w') as f:
                json.dump(config, f, indent=4)
           