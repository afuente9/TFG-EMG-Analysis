# TFG-EMG-Analysis
Description:
This repository contains a Python-based framework for processing, filtering, and analyzing electromyography (EMG) signals collected from two different devices: Delsys Trigno Avanti (gold standard) and mDurance (low-cost alternative). The project focuses on signal synchronization, filtering (using Butterworth filters), normalization, and statistical comparison through RMSE, FastDTW, and cross-correlation analysis.

Features:

Data Processing: Functions to load, clean, and process EMG signals from CSV and Excel files.
Signal Filtering: Implements Butterworth filtering for noise reduction.
Synchronization: Aligns signals from Delsys and mDurance based on RMS envelope thresholds.
Normalization: Scales signal envelopes for direct comparison.
Statistical Analysis: Calculates RMSE, Dynamic Time Warping (DTW), and cross-correlation to evaluate the agreement between the two systems.
Visualization: Plots EMG signals, filtered signals, and statistical results using Matplotlib.
Interactive CLI Menu: Provides an interactive terminal interface for data loading, processing, and analysis.
Usage:
Run the script in Python to access an interactive menu that allows you to process and analyze EMG signals step by step.
