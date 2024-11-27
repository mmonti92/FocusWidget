# FocusWidget

## Overview

The **FocusWidget** repository provides a set of Python tools for visualizing, analyzing, and fitting spot-like intensity distributions in images. This project is designed with PyQt6 and integrates Matplotlib for interactive plotting and `lmfit` for 2D Gaussian model fitting. The application allows for drag-and-drop functionality, background subtraction, and visualization of raw data, fitted data, residuals, and projections along different axes.

This repository is particularly useful for analyzing image data in scientific applications such as microscopy, optical measurements, and beam profiling.

## Features

- **Drag-and-Drop Plotting**: Easily load image data by dragging files into the application window.
- **Interactive Visualization**: Display data with Matplotlib, with support for zooming, panning, and adjusting plot elements.
- **2D Gaussian Fitting**: Fit 2D Gaussian models to the image data and visualize the fit results.
- **Background Subtraction**: Subtract background data to enhance the analysis of foreground signals.
- **Summed Projections**: View summed projections along the x and y axes to better understand the intensity distributions.

## Installation

### Prerequisites

To run this project, you'll need the following Python libraries:

- **PyQt6**: For the graphical user interface.
- **Matplotlib**: For plotting the data.
- **NumPy**: For handling arrays and numerical operations.
- **lmfit**: For nonlinear least squares fitting (2D Gaussian fitting).
- **DataAnalysis**: Custom package for reading, fitting, and residual calculation (assumed to be present).

### Install Dependencies

You can install the required libraries using `pip`:

```bash
pip install PyQt6 matplotlib numpy lmfit
```


## License

[MIT](https://choosealicense.com/licenses/mit/
