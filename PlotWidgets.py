import matplotlib.pyplot as plt
import numpy as np
import typing as tp
from PyQt6.QtWidgets import (
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT,
)


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig, self.ax = plt.subplots()
        super().__init__(fig)


class PlotWidget(QWidget):
    def __init__(self, title: str, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.canvas = MplCanvas(self)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.canvas.ax.set_title(title)

    def clear_plot(self) -> None:
        self.canvas.ax.clear()
        self.canvas.draw()

    def plot_img(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        *args: tp.Any,
        **kwargs: tp.Any
    ):
        self.canvas.ax.imshow(
            z,
            aspect="equal",
            extent=[x[0] * 1e-3, x[-1] * 1e-3, y[0] * 1e-3, y[-1] * 1e-3],
            *args,
            **kwargs
        )
        # self.canvas.ax.legend(loc="upper center")
        self.canvas.draw()

    def plot_data(
        self, x: np.ndarray, y: np.ndarray, label: str, color: str = None
    ):
        self.canvas.ax.plot(x, y, label=label, color=color)
        self.canvas.ax.legend(loc="upper left")
        self.canvas.draw()


if __name__ == "__main__":
    ...
