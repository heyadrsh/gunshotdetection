import sys
import os
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                           QProgressBar, QFrame, QSizePolicy, QSpacerItem)
from PyQt6.QtCore import Qt, QMimeData, QThread, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import librosa
import librosa.display
from test_model import load_model, predict_gunshot, load_and_preprocess_audio

class PlotWidget(QWidget):
    def __init__(self, title, height=2, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(6, height))
        self.canvas = FigureCanvas(self.figure)
        self.title = title
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Title label
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            QLabel {
                color: #444444;
                font-size: 13px;
                font-weight: bold;
                padding: 8px 0;
            }
        """)
        
        layout.addWidget(title_label)
        
        # Plot container
        plot_container = QFrame()
        plot_container.setStyleSheet("""
            QFrame {
                background: white;
                border-radius: 8px;
                padding: 10px;
                border: 1px solid #EEEEEE;
            }
        """)
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(10, 10, 10, 10)
        plot_layout.addWidget(self.canvas)
        
        layout.addWidget(plot_container)
        self.setLayout(layout)
        
    def clear(self):
        self.figure.clear()
        self.canvas.draw()

class DropZone(QFrame):
    dropped = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #CCCCCC;
                border-radius: 10px;
                background: #FFFFFF;
                padding: 25px;
            }
            QFrame:hover {
                border-color: #2196F3;
                background: #F5F5F5;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(12)
        
        # Text
        text = QLabel("Drop WAV file here\nor click to browse")
        text.setStyleSheet("""
            color: #666666;
            font-size: 14px;
            font-weight: bold;
        """)
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(text)
        self.setLayout(layout)
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()
            
    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        for f in files:
            if f.lower().endswith('.wav'):
                self.dropped.emit(f)
                break
                
    def mousePressEvent(self, event):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open WAV File", "", 
                                                 "WAV Files (*.wav)")
        if file_name:
            self.dropped.emit(file_name)

class ResultsWidget(QFrame):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QFrame {
                background: #FFFFFF;
                border-radius: 10px;
                padding: 20px;
                border: 1px solid #EEEEEE;
            }
        """)
        
        self.layout = QVBoxLayout()
        self.layout.setSpacing(15)
        self.setLayout(self.layout)
        
        # Title
        title = QLabel("Results")
        title.setStyleSheet("""
            font-size: 16px;
            color: #444444;
            font-weight: bold;
            margin-bottom: 5px;
        """)
        self.layout.addWidget(title)
        
        # Initialize empty labels
        self.detection = QLabel()
        self.confidence = QLabel()
        self.timing = QLabel()
        
        for label in [self.detection, self.confidence, self.timing]:
            label.setStyleSheet("""
                color: #555555;
                font-size: 14px;
                padding: 5px 0;
                font-weight: 500;
            """)
            self.layout.addWidget(label)
            
    def update_results(self, predicted_type, confidence, timings):
        self.detection.setText(f"Detected: {predicted_type}")
        
        # Color-coded confidence
        confidence_color = "#4CAF50" if confidence > 80 else "#FFA726" if confidence > 50 else "#EF5350"
        self.confidence.setStyleSheet(f"""
            color: {confidence_color};
            font-size: 14px;
            padding: 5px 0;
            font-weight: bold;
        """)
        self.confidence.setText(f"Confidence: {confidence:.1f}%")
        
        self.timing.setText(f"Processing Time: {timings['total']:.3f}s")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gunshot Detection")
        self.setFixedSize(1000, 650)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F8F9FA;
            }
            QWidget {
                font-family: 'Segoe UI', Arial;
            }
        """)
        
        # Load model
        self.label_encoder_classes = np.load('label_encoder_classes.npy')
        self.model, self.device = load_model('best_model.pth', 
                                           len(self.label_encoder_classes))
        
        # Main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(25)
        
        # Left panel (Input and Results)
        left_panel = QVBoxLayout()
        left_panel.setSpacing(20)
        
        # Title
        title = QLabel("Gunshot Detection")
        title.setStyleSheet("""
            font-size: 24px;
            color: #2196F3;
            font-weight: bold;
            margin-bottom: 10px;
        """)
        left_panel.addWidget(title)
        
        # Drop zone
        self.drop_zone = DropZone()
        self.drop_zone.dropped.connect(self.process_file)
        left_panel.addWidget(self.drop_zone)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: none;
                background: #EEEEEE;
                height: 4px;
                border-radius: 2px;
            }
            QProgressBar::chunk {
                background: #2196F3;
                border-radius: 2px;
            }
        """)
        self.progress.hide()
        left_panel.addWidget(self.progress)
        
        # Results widget
        self.results = ResultsWidget()
        left_panel.addWidget(self.results)
        
        left_panel.addStretch()
        
        # Add left panel to main layout
        left_container = QWidget()
        left_container.setLayout(left_panel)
        left_container.setFixedWidth(320)
        layout.addWidget(left_container)
        
        # Right panel (Visualizations)
        right_panel = QVBoxLayout()
        right_panel.setSpacing(20)
        
        # Plots
        self.waveform = PlotWidget("Waveform", height=1.5)
        self.spectrogram = PlotWidget("Mel Spectrogram", height=2)
        self.confidence_plot = PlotWidget("Prediction Confidence", height=2)
        
        right_panel.addWidget(self.waveform)
        right_panel.addWidget(self.spectrogram)
        right_panel.addWidget(self.confidence_plot)
        
        # Add right panel to main layout
        right_container = QWidget()
        right_container.setLayout(right_panel)
        layout.addWidget(right_container)

    def plot_waveform(self, audio_path):
        self.waveform.figure.clear()
        audio, sr = librosa.load(audio_path, sr=22050)
        ax = self.waveform.figure.add_subplot(111)
        librosa.display.waveshow(audio, sr=sr, ax=ax, color='#2196F3')
        ax.set_facecolor('none')
        self.waveform.figure.patch.set_alpha(0.0)
        ax.grid(True, color='#F5F5F5')
        ax.set_xlabel('Time (s)', fontsize=9, color='#666666')
        ax.set_ylabel('Amplitude', fontsize=9, color='#666666')
        ax.tick_params(colors='#666666', labelsize=8)
        self.waveform.canvas.draw()
        
    def plot_spectrogram(self, features):
        self.spectrogram.figure.clear()
        ax = self.spectrogram.figure.add_subplot(111)
        img = librosa.display.specshow(features, y_axis='mel', x_axis='time', 
                                     ax=ax, cmap='viridis')
        self.spectrogram.figure.colorbar(img, format='%+2.0f dB', ax=ax)
        ax.set_facecolor('none')
        ax.set_xlabel('Time (s)', fontsize=9, color='#666666')
        ax.set_ylabel('Frequency (Hz)', fontsize=9, color='#666666')
        ax.tick_params(colors='#666666', labelsize=8)
        self.spectrogram.figure.patch.set_alpha(0.0)
        self.spectrogram.canvas.draw()
        
    def plot_confidence(self, probabilities, classes):
        self.confidence_plot.figure.clear()
        ax = self.confidence_plot.figure.add_subplot(111)
        y_pos = np.arange(len(classes))
        
        # Create background bars (light gray)
        ax.barh(y_pos, [100] * len(classes), align='center', color='#F5F5F5', 
               zorder=1)
        
        # Create main bars
        colors = ['#4CAF50' if p > 80 else '#FFA726' if p > 50 else '#EF5350' 
                 for p in probabilities]
        bars = ax.barh(y_pos, probabilities, align='center', color=colors, 
                      zorder=2)
        
        # Set axis labels and ticks
        ax.set_yticks(y_pos)
        ax.set_yticklabels(classes, fontsize=10, color='#444444')
        ax.set_xticks(np.arange(0, 101, 20))
        ax.set_xlim(0, 100)
        ax.invert_yaxis()
        
        # Add percentage labels
        for i, v in enumerate(probabilities):
            # Add gun type label on the left
            ax.text(-5, i, classes[i], 
                   ha='right', va='center',
                   fontsize=10, color='#444444',
                   fontweight='bold')
            
            # Add percentage on the right of each bar
            if v > 0:
                percentage_text = f'{v:.0f}%'
                text_color = '#FFFFFF' if v > 30 else '#444444'
                text_x = min(v + 2, 98)  # Cap position at 98%
                
                ax.text(text_x, i, percentage_text,
                       ha='left' if v < 80 else 'right',
                       va='center',
                       color=text_color,
                       fontsize=9,
                       fontweight='bold',
                       zorder=3)
        
        # Customize grid and appearance
        ax.set_facecolor('white')
        self.confidence_plot.figure.patch.set_alpha(0.0)
        
        # Remove all spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Add subtle horizontal grid lines
        ax.yaxis.grid(True, linestyle='-', color='#EEEEEE', zorder=0)
        ax.xaxis.grid(False)
        
        # Hide y-axis labels since we're showing them inside the plot
        ax.set_yticklabels([])
        
        # Style x-axis
        ax.tick_params(axis='x', colors='#666666', labelsize=8)
        
        # Add title
        ax.set_title('Prediction Confidence', 
                    pad=10, 
                    fontsize=11, 
                    color='#444444',
                    fontweight='bold')
        
        self.confidence_plot.canvas.draw()
        
    def process_file(self, file_path):
        self.progress.show()
        self.progress.setRange(0, 0)
        
        # Update plots
        self.plot_waveform(file_path)
        features = load_and_preprocess_audio(file_path)
        if features is not None:
            self.plot_spectrogram(features)
        
        # Process in thread
        self.thread = QThread()
        self.worker = Worker(self.model, self.device, 
                           self.label_encoder_classes, file_path)
        self.worker.moveToThread(self.thread)
        
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.update_results)
        
        self.thread.start()

    def update_results(self, result):
        if result[0] is None:
            return
            
        predicted_type, confidence, probabilities, timings = result
        
        # Update results widget
        self.results.update_results(predicted_type, confidence, timings)
        
        # Update confidence plot
        self.plot_confidence(probabilities, self.label_encoder_classes)
        
        self.progress.hide()

class Worker(QThread):
    finished = pyqtSignal(tuple)
    
    def __init__(self, model, device, label_encoder_classes, audio_path):
        super().__init__()
        self.model = model
        self.device = device
        self.label_encoder_classes = label_encoder_classes
        self.audio_path = audio_path
        
    def run(self):
        result = predict_gunshot(self.model, self.audio_path, 
                               self.label_encoder_classes, self.device)
        self.finished.emit(result)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main() 