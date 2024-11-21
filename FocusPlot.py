# Maurizio Monti 2024
import sys
import numpy as np
import json
import warnings as wn

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QCheckBox,
    QFileDialog,
    QPushButton,
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt

import PlotWidgets as pw
import Utilities as ut


class DragAndDropPlotter(QMainWindow):
    def __init__(self):
        super().__init__()

        # ## Defintitions
        self.setWindowTitle("Drag and Drop Plotter")
        self.setGeometry(100, 100, 1200, 800)

        self.label = QLabel("Drag and drop a file here to plot", self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.result_label_x = QLabel("Width in x", self)
        self.result_label_y = QLabel("Width in y", self)
        self.result_label_x.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label_y.setAlignment(Qt.AlignmentFlag.AlignCenter)
        custom_font = QFont("Arial", 30)
        self.result_label_x.setFont(custom_font)
        self.result_label_y.setFont(custom_font)

        self.subtract_bg = False  # Initial mode: no bg
        self.check = QCheckBox("Subtract background", self)
        self.check.setChecked(False)
        self.check.stateChanged.connect(self.toggle_bg_subtraction)

        self.default_folder = self.load_folder()

        # self.selected_file_path
        self.button = QPushButton("Load bg")
        self.button.clicked.connect(self.open_file_dialog)
        self.bg_label = QLabel("No background", self)
        self.bg_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Set up plot widgets
        self.plot_data = pw.PlotWidget("Data", self)
        self.plot_fit = pw.PlotWidget("Fit", self)
        self.plot_res = pw.PlotWidget("residuals", self)

        self.plot_x = pw.PlotWidget("Sum x", self)
        self.plot_y = pw.PlotWidget("Sum y", self)

        plot_layout_2D = QHBoxLayout()
        plot_layout_2D.addWidget(self.plot_data)
        plot_layout_2D.addWidget(self.plot_fit)
        plot_layout_2D.addWidget(self.plot_res)

        plot_layout_sums = QHBoxLayout()
        plot_layout_sums.addWidget(self.plot_x)
        plot_layout_sums.addWidget(self.plot_y)

        bg_layout = QHBoxLayout()
        bg_layout.addWidget(self.check)
        bg_layout.addWidget(self.button)
        bg_layout.addWidget(self.bg_label)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.label)
        main_layout.addLayout(bg_layout)
        main_layout.addLayout(plot_layout_2D)
        main_layout.addLayout(plot_layout_sums)
        main_layout.addWidget(self.result_label_x)
        main_layout.addWidget(self.result_label_y)

        # Putting all together
        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Enable drag-and-drop
        self.setAcceptDrops(True)

    def load_folder(self):
        try:
            with open("Paths.json") as f:
                config = json.load(f)
                return config.get("default_folder", "")
        except FileNotFoundError:
            wn.warn(
                "Default background folder not found,"
                + " file dialog will open on widget directory.",
                category=UserWarning,
            )
            return ""

    def toggle_bg_subtraction(self):
        self.subtract_bg = not self.subtract_bg
        try:
            self.plot_file(self.last_file)
        except AttributeError:
            wn.warn(
                "Attempting to subtract a background without having loaded"
                + " a file.",
                category=UserWarning,
            )

    def open_file_dialog(self):
        initial_folder = self.default_folder
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select a File", initial_folder, "All Files (*.*)"
        )
        if file_path:
            self.bg_path = file_path
            self.bg_label.setText(f"Selected File: {file_path}")
        else:
            self.bg_label.setText("No file selected")

    # Override dragEnterEvent to accept files
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    # Override dropEvent to handle dropped files
    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[
                0
            ].toLocalFile()  # Get the file path of the dropped file
            self.last_file = file_path
            self.plot_file(file_path)

    def plot_file(self, file_path):
        try:
            data = ut.ReadImg(file_path)

            x = data[0]  # Assume first column as x-axis
            y = data[1]  # Assume second column as y-axis
            img = data[2]

            if self.subtract_bg:
                try:
                    _, _, bg = ut.ReadImg(self.bg_path)
                    img -= bg
                except AttributeError:
                    wn.warn(
                        "Attempting to subtract a background without having"
                        + " loaded one. Background subtraction ignored.",
                        category=UserWarning,
                    )
                    pass

            self.plot_data.clear_plot()
            self.plot_fit.clear_plot()
            self.plot_res.clear_plot()
            self.plot_x.clear_plot()
            self.plot_y.clear_plot()

            self.plot_data.canvas.ax.set_xlabel("x[mm]")
            self.plot_data.canvas.ax.set_ylabel("y[mm]")

            self.plot_data.plot_img(
                x,
                y,
                img,
                vmin=0,
                vmax=10000,
                # label=f"{file_path.split('/')[-1]}",
            )

            # Fitting
            out, _, fitted = ut.FitSpot(img, x, y)
            wx, wy, swx, swy = (
                np.rint(out.params["sx"].value * 2),
                np.rint(out.params["sy"].value * 2),
                np.rint(out.params["sx"].stderr * 2),
                np.rint(out.params["sy"].stderr * 2),
            )

            self.plot_fit.canvas.ax.set_xlabel("x[mm]")
            self.plot_fit.canvas.ax.set_ylabel("y[mm]")

            self.plot_fit.plot_img(
                x,
                y,
                fitted,
                vmin=0,
                vmax=10000,
            )

            # Residuals

            self.plot_res.canvas.ax.set_xlabel("x[mm]")
            self.plot_res.canvas.ax.set_ylabel("y[mm]")

            self.plot_res.plot_img(
                x,
                y,
                img - fitted,
                vmin=-1000,
                vmax=1000,
                cmap="coolwarm",
            )

            # Sum x

            self.plot_x.canvas.ax.set_xlabel("x[mm]")
            self.plot_x.canvas.ax.set_ylabel("Sum along y")
            self.plot_x.plot_data(
                x,
                img.sum(0),
                label=f"data",
            )
            self.plot_x.plot_data(
                x,
                fitted.sum(0),
                label="Fit",
            )

            # Sum y
            self.plot_y.canvas.ax.set_xlabel("y[mm]")
            self.plot_y.canvas.ax.set_ylabel("Sum along x")
            self.plot_y.plot_data(
                y,
                img.sum(1),
                label=f"data",
            )
            self.plot_y.plot_data(
                y,
                fitted.sum(1),
                label="Fit",
            )

            self.label.setText(f"Plotting data from: {file_path}")
            self.result_label_x.setText(f"x width=({wx}\u00B1{swx})\u00B5m")
            self.result_label_y.setText(f"y width=({wy}\u00B1{swy})\u00B5m")

        except Exception as e:
            self.label.setText(f"Failed to plot data: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DragAndDropPlotter()
    window.show()
    sys.exit(app.exec())
