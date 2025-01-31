# Import necessary libraries
import pyemgpipeline as pep
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.figure import SubplotParams
from scipy.signal import filtfilt
from scipy.interpolate import interp1d
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Butterworth filter configuration
butter_b = [0.5139044, 0.0000000, -2.0556177, 0.0000000, 3.0834266, 0.0000000, -2.0556177, 0.0000000, 0.5139044]
butter_a = [1.0000000, -0.6755365, -2.4805579, 1.1749712, 2.6492236, -0.7716328, -1.3369744, 0.1783695, 0.2641028]

# General dictionary to store all DataFrames organized by repetition type and patient
patient_data = {
    "Delsys Continuous Repetition": {},   # Stores Delsys data (Continuous Repetition)
    "Delsys 5 Repetitions": {},        # Stores Delsys data (5 Repetitions)
    "mDurance Continuous Repetition": {}, # Stores mDurance data (Continuous Repetition)
    "mDurance 5 Repetitions": {}       # Stores mDurance data (5 Repetitions)
}

# Lista de tipos de Repeticion
repetition_type = ["Continuous Repetition", "5 Repetitions"]

# Function to filter a signal using a Butterworth filter
def filter_signal(signal, b, a):
    return filtfilt(b, a, signal)
    
# Function to load and process mDurance data
def process_mdurance(file_path, fs):
    df = pd.read_csv(file_path, delimiter=';')
    df['time'] = np.arange(len(df))/fs
    df.columns = ['window_angles', 'angles_biceps', 'signal', 'time']
    # Convert microvolts to volts
    df['signal'] = df['signal'] * 1e-6  
    df['filtered_signal'] = filter_signal(df['signal'], butter_b, butter_a)
    
    signal_array = df['filtered_signal'].to_numpy()
    trial_name = "mDurance Test"
    channel_names = ['biceps brachii']
    emg_plot_params = pep.plots.EMGPlotParams(n_rows=1, fig_kwargs={'figsize': (12, 8), 'dpi': 80, 'subplotpars': SubplotParams(wspace=0, hspace=0.6)}, line2d_kwargs={'color': 'blue'})
    
    emg = pep.wrappers.EMGMeasurement(data=signal_array, hz=fs, trial_name=trial_name, channel_names=channel_names, emg_plot_params=emg_plot_params)
    
    emg.apply_dc_offset_remover()
    emg.apply_full_wave_rectifier()
    df['envelope'] = emg.data
    emg.apply_linear_envelope()
    df['rms_envelope'] = emg.data
    
    return df

# Function to load and process Delsys data
def process_delsys(file_path, fs):
    df = pd.read_excel(file_path)
    df.columns = ['time', 'signal']
    df['filtered_signal'] = df['signal']
    
    signal_array = df['filtered_signal'].to_numpy()
    trial_name = "Delsys Test"
    channel_names = ['biceps brachii']
    emg_plot_params = pep.plots.EMGPlotParams(n_rows=1, fig_kwargs={'figsize': (12, 8), 'dpi': 80, 'subplotpars': SubplotParams(wspace=0, hspace=0.6)}, line2d_kwargs={'color': 'blue'})
    
    emg = pep.wrappers.EMGMeasurement(data=signal_array, hz=fs, trial_name=trial_name, channel_names=channel_names, emg_plot_params=emg_plot_params)
    
    emg.apply_dc_offset_remover()
    emg.apply_full_wave_rectifier()
    df['envelope'] = emg.data
    emg.apply_linear_envelope()
    df['rms_envelope'] = emg.data
    
    return df

# Function to synchronize signals between mDurance and Delsys
def synchronize_signals(mdurance_df, delsys_df, target_fs, threshold):
    """
    Synchronize Delsys and mDurance data by removing the first peak in each signal and adjusting start and end times.
    """
    t_cross_mdurance = mdurance_df['time'][np.argmax(mdurance_df['rms_envelope'] > threshold)]
    t_cross_delsys = delsys_df['time'][np.argmax(delsys_df['rms_envelope'] > threshold)]
    
    mdurance_df['time'] -= t_cross_mdurance
    delsys_df['time'] -= t_cross_delsys
    
    start_time = max(mdurance_df['time'].iloc[0], delsys_df['time'].iloc[0])
    end_time = min(mdurance_df['time'].iloc[-1], delsys_df['time'].iloc[-1])
    num_samples = int(round(target_fs * (end_time - start_time)))
    common_time = np.linspace(start_time, end_time, num_samples)
    
    mdurance_interpolator = interp1d(mdurance_df['time'], mdurance_df['rms_envelope'], kind='linear', fill_value="extrapolate")
    delsys_interpolator = interp1d(delsys_df['time'], delsys_df['rms_envelope'], kind='linear', fill_value="extrapolate")
    
    mdurance_df = mdurance_df.iloc[:len(common_time)].copy()
    delsys_df = delsys_df.iloc[:len(common_time)].copy()
    
    mdurance_df['time'] = common_time
    mdurance_df['rms_envelope'] = mdurance_interpolator(common_time)
    
    delsys_df['time'] = common_time
    delsys_df['rms_envelope'] = delsys_interpolator(common_time)
    
    return mdurance_df, delsys_df

# Function to recalculate the features of a signal
def recalculate_features(df, fs, system_type):
    if system_type == 'mdurance':
        signal_column = 'filtered_signal'
    elif system_type == 'delsys':
        signal_column = 'filtered_signal'
    else:
        raise ValueError("Invalid system type. Use 'delsys' or 'mdurance'")
    
    # Convert the signal to ndarray
    signal_array = df[signal_column].to_numpy()
    trial_name = f"test {system_type}"
    channel_names = ['biceps brachii']
    emg_plot_params = pep.plots.EMGPlotParams(n_rows=1,fig_kwargs={'figsize': (12, 8), 'dpi': 80, 'subplotpars': SubplotParams(wspace=0, hspace=0.6)}, line2d_kwargs={'color': 'blue'})
    
    # Initialize the EMGMeasurement object with data and parameters
    emg = pep.wrappers.EMGMeasurement(data = signal_array, hz = fs, trial_name = trial_name, channel_names = channel_names, emg_plot_params = emg_plot_params)
    
    # Apply signal processing steps
    emg.apply_dc_offset_remover()
    emg.apply_full_wave_rectifier()
    envelope = emg.data
    df['envelope']  = envelope
    emg.apply_linear_envelope()
    rms = emg.data
    df['rms_envelope'] = rms
    
    return df

# Function to load patient data
def load_data():
    print("Loading and processing data...")
    
    fs_delsys = 2048
    fs_mdurance = 1024

    base_path = os.path.join(os.getcwd(), 'Datos')  # Path to data

    if os.path.exists(base_path):
        patients = [folder for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]
        for patient in patients:
            patient_path = os.path.join(base_path, patient)
            try:
                # Use os.scandir to list files in the folder
                with os.scandir(patient_path) as entries:
                    for entry in entries:
                        if entry.is_file():
                            file_path = entry.path
                            file_name = entry.name
                            
                            # Process files based on their extension
                            try:
                                if file_name.endswith('_rep_cont.xlsx'):
                                    df = process_delsys(file_path, fs_delsys)
                                    patient_data["Delsys Continuous Repetition"][patient] = df
                                    print(f"Data for {patient} (Delsys Continuous Repetition) loaded from {file_name}")

                                elif file_name.endswith('_5_rep.xlsx'):
                                    df = process_delsys(file_path, fs_delsys)
                                    patient_data["Delsys 5 Repetitions"][patient] = df
                                    print(f"Data for {patient} (Delsys 5 Repetitions) loaded from {file_name}")

                                elif file_name.endswith('_rep_cont.csv'):
                                    df = process_mdurance(file_path, fs_mdurance)
                                    patient_data["mDurance Continuous Repetition"][patient] = df
                                    print(f"Data for {patient} (mDurance Continuous Repetition) loaded from {file_name}")

                                elif file_name.endswith('_5_rep.csv'):
                                    df = process_mdurance(file_path, fs_mdurance)
                                    patient_data["mDurance 5 Repetitions"][patient] = df
                                    print(f"Data for {patient} (mDurance 5 Repetitions) loaded from {file_name}")
                            except Exception as e:
                                print(f"Error processing file {file_name} for {patient}: {e}")
            except Exception as e:
                print(f"Error processing folder for {patient}: {e}")
    else:
        print("Data folder not found.")
    return True

# Function to normalize patient data
def normalize_data(patient):
    if patient == 'all':
        patients = list_patients()
    else:
        patients = [patient]
        
    for patient in patients:
        types = ["Delsys Continuous Repetition", "Delsys 5 Repetitions", "mDurance Continuous Repetition", "mDurance 5 Repetitions"]
        
        for data_type in types:
            if patient in patient_data[data_type]:
                print(f"Normalizing data for {patient} in category {data_type}...")
                df = patient_data[data_type][patient]
                if 'envelope' in df.columns:
                    df['envelope'] = normalize_signal(df['envelope'])

                if 'rms_envelope' in df.columns:
                    df['rms_envelope'] = normalize_signal(df['rms_envelope'])

                patient_data[data_type][patient] = df
                print(f"Data normalized for {patient} in {data_type}.")
            else:
                print(f"No data found for {patient} in category {data_type}.")
    
    return True

# Function to normalize a signal
def normalize_signal(signal):
    return (signal - signal.min()) / (signal.max() - signal.min())


# Function to remove initial peak from a signal
def remove_initial_data(df, start_time=3):
    """
    Removes all data from the initial time up to the specified time
    and adjusts the timestamps to start from 0.
    
    Args:
    - df (pd.DataFrame): DataFrame containing the signal data.
    - start_time (float): Time in seconds from which data will be retained.
    
    Returns:
    - pd.DataFrame: Adjusted DataFrame.
    """
    # Filter data from the specified time onward
    df = df[df['time'] >= start_time].copy()
    
    # Adjust timestamps to start from 0
    df['time'] -= df['time'].iloc[0]
    
    return df

# Function to synchronize patient data
def synchronize_data(patient, repetition_type):
    print(f"Synchronizing data for {patient}...")
    
    if patient == 'all':
        patients = list_patients()
    else:
        patients = [patient]

    for patient in patients:
        for rep_type in repetition_type:
            t = f'Delsys {rep_type}'
            t1 = f'mDurance {rep_type}'
            if patient in patient_data[t] and patient in patient_data[t1]:
                df_delsys = patient_data[t][patient]
                df_mdurance = patient_data[t1][patient]
                # Verify if signals are normalized
                if not (df_delsys['rms_envelope'].min() >= 0 and df_delsys['rms_envelope'].max() <= 1):
                    print(f"Error: RMS signal from Delsys is not normalized for {patient} in {rep_type}.")
                    print("Returning to main menu.")
                    return False
                if not (df_mdurance['rms_envelope'].min() >= 0 and df_mdurance['rms_envelope'].max() <= 1):
                    print(f"Error: RMS signal from mDurance is not normalized for {patient} in {rep_type}.")
                    return False
                if 'time' in df_delsys.columns and 'time' in df_mdurance.columns:
                    if rep_type == "Continuous Repetition":
                        threshold = 0.35
                    elif rep_type == "5 Repetitions":
                       threshold = 0.25
                    df_mdurance, df_delsys = synchronize_signals(df_mdurance, df_delsys, 1024, threshold)
                                                                  
                    patient_data[t][patient] = df_delsys
                    patient_data[t1][patient] = df_mdurance
                    print(f"Data synchronized for {patient} in {rep_type}.")
                else:
                    print(f"No temporal data found for {patient} in category {rep_type}.")
            else:
                print(f"No data found for {patient} in category {rep_type}.")
    
    return True

# Function to plot the data
def plot_data():
    to_all = input("Do you want to plot data for all patients? (yes/no): ").strip().lower()
    
    if to_all in ['yes', 'y']:
        for rep_type in repetition_type:
            # Iterate over all patients for each repetition type
            for patient in list_patients():
                if patient in patient_data[f'Delsys {rep_type}'] and patient in patient_data[f'mDurance {rep_type}']:
                    df_delsys = patient_data[f'Delsys {rep_type}'][patient]
                    df_mdurance = patient_data[f'mDurance {rep_type}'][patient]
                    if 'rms_envelope' in df_delsys.columns and 'rms_envelope' in df_mdurance.columns:
                        plt.figure(figsize=(12, 8))
                        plt.plot(df_delsys['time'], df_delsys['rms_envelope'], label=f'{patient} Delsys')
                        plt.plot(df_mdurance['time'], df_mdurance['rms_envelope'], label=f'{patient} mDurance')
                        plt.xlabel('Time (s)')
                        plt.ylabel('RMS Envelope')
                        plt.title(f'Comparison of RMS Envelopes between Delsys and mDurance ({rep_type})')
                        plt.legend()
                        plt.show()
                    else:
                        print(f'Error: No RMS envelope data found for {patient} in category {rep_type}.')
            
    else:
        while True:
            patient = input("Enter the patient's name: ").strip().lower()
        
            if patient not in [p.lower() for p in list_patients()]:
                print(f"Error: The patient '{patient}' doen't exist in the data base.")
            else:
                patient = next(p for p in list_patients() if p.lower() == patient)
                break

        # Function to validate repetition type
        while True:
            rep_type = input("Enter the repetition type (Continuous Repetition or 5 Repetitions): ").strip().lower()

            # Validate the repetition type
            if rep_type in ["continuous repetition", "5 repetitions"]:
                # Convert rep_type to the correct format for use in the dictionary
                rep_type = "Continuous Repetition" if rep_type == "continuous repetition" else "5 Repetitions"
                break
            else:
                print("Error: The repetition type must be 'Continuous Repetition' or '5 Repetitions'. Please try again.")
        
        # Check if the patient exists in both categories based on the repetition type
        if rep_type == "Continuous Repetition":
            if not (patient in patient_data["Delsys Continuous Repetition"] and patient in patient_data["mDurance Continuous Repetition"]):
                print(f"Error: The patient '{patient}' does not exist in both categories of Continuous Repetition.")
                return
        elif rep_type == "5 Repetitions":
            if not (patient in patient_data["Delsys 5 Repetitions"] and patient in patient_data["mDurance 5 Repetitions"]):
                print(f"Error: The patient '{patient}' does not exist in both categories of 5 Repetitions.")
                return

        # Select the correct DataFrame based on the repetition type
        df_delsys = patient_data[f'Delsys {rep_type}'][patient]
        df_mdurance = patient_data[f'mDurance {rep_type}'][patient]

        
        # Ask the user if they want a single graph or two stacked graphs
        while True:
            option = input("Do you want to see 1 or 2 graphs? (1 for a single plot, 2 for two stacked plots): ").strip()

            if option == "1":
                # Plot a single graph
                plot_single_graph(df_delsys, df_mdurance)
                break
            elif option == "2":
                # Plot two stacked graphs
                plot_two_graphs(df_delsys, df_mdurance)
                break
            else:
                print("Invalid option. Please choose 1 or 2.")

    return True

# Function to plot a single graph
def plot_single_graph(df_delsys, df_mdurance):
    while True:
        num_var = input("How many variables do you want to plot? (1, 2, 3, or 4): ").strip()
        
        if num_var in ["1", "2", "3", "4"]:
            break
        else:
            print("Invalid option. Please choose 1, 2, 3, or 4.")
            
    # Request column names
    if num_var == '1':
        c1 = input("Enter the name of the variable to plot (signal/filtered_signal/envelope/rms_envelope): ").strip().lower()
        if c1 in df_delsys.columns and c1 in df_mdurance.columns:
            plt.figure(figsize=(12, 8))
            plt.plot(df_delsys['time'], df_delsys[c1], label='Delsys')
            plt.plot(df_mdurance['time'], df_mdurance[c1], label='mDurance')
            plt.xlabel('Time (s)')
            plt.ylabel(c1)
            plt.title(f'Comparison of {c1} between Delsys and mDurance')
            plt.legend()
            plt.show()
    elif num_var in ['2', '3']:
        variables = []
        for i in range(int(num_var)):
            var = input(f"Enter the name of variable {i+1} to plot (signal/filtered_signal/envelope/rms_envelope): ").strip().lower()
            if var in df_delsys.columns and var in df_mdurance.columns:
                variables.append(var)
            else:
                print(f"Error: The variable '{var}' does not exist in both DataFrames.")
                return
        
        plt.figure(figsize=(12, 8))
        for var in variables:
            plt.plot(df_delsys['time'], df_delsys[var], label=f'Delsys {var}')
            plt.plot(df_mdurance['time'], df_mdurance[var], label=f'mDurance {var}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f"Graph of {', '.join(variables)} in Delsys and mDurance")
        plt.legend()
        plt.show()
    elif num_var == '4':
        plt.figure(figsize=(12, 8))
        for var in ['signal', 'filtered_signal', 'envelope', 'rms_envelope']:
            plt.plot(df_delsys['time'], df_delsys[var], label=f'Delsys {var}')
            plt.plot(df_mdurance['time'], df_mdurance[var], label=f'mDurance {var}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title("Graph of all variables in Delsys and mDurance")
        plt.legend()
        plt.show()
    
    return True

# Function to plot two graphs
def plot_two_graphs(df_delsys, df_mdurance):
    while True:
        num_var = input("How many variables do you want to plot? (1, 2, 3, or 4): ").strip()
        
        if num_var in ["1", "2", "3", "4"]:
            break
        else:
            print("Invalid option. Please choose 1, 2, 3, or 4.")
            
    if num_var == '1':
        c1 = input("Enter the name of the variable to plot (signal/filtered_signal/envelope/rms_envelope): ").strip().lower()
        if c1 in df_delsys.columns and c1 in df_mdurance.columns:
            plt.figure(figsize=(12, 8))
            plt.plot(df_delsys['time'], df_delsys[c1], label='Delsys')
            plt.xlabel('Time (s)')
            plt.ylabel(c1)
            plt.title(f'Graph of {c1} in Delsys')
            plt.legend()
            
            plt.figure(figsize=(12, 8))
            plt.plot(df_mdurance['time'], df_mdurance[c1], label='mDurance')
            plt.xlabel('Time (s)')
            plt.ylabel(c1)
            plt.title(f'Graph of {c1} in mDurance')
            plt.legend()
            plt.show()
    elif num_var in ['2', '3']:
        variables = []
        for i in range(int(num_var)):
            var = input(f"Enter the name of variable {i+1} to plot (signal/filtered_signal/envelope/rms_envelope): ").strip().lower()
            if var in df_delsys.columns and var in df_mdurance.columns:
                variables.append(var)
            else:
                print(f"Error: The variable '{var}' does not exist in both DataFrames.")
                return
        
        plt.figure(figsize=(12, 8))
        for var in variables:
            plt.plot(df_delsys['time'], df_delsys[var], label=f'Delsys {var}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f"Graph of {', '.join(variables)} in Delsys")
        plt.legend()
        plt.show()
        
        plt.figure(figsize=(12, 8))
        for var in variables:
            plt.plot(df_mdurance['time'], df_mdurance[var], label=f'mDurance {var}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f"Graph of {', '.join(variables)} in mDurance")
        plt.legend()
        plt.show()
    elif num_var == '4':
        plt.figure(figsize=(12, 8))
        for var in ['signal', 'filtered_signal', 'envelope', 'rms_envelope']:
            plt.plot(df_delsys['time'], df_delsys[var], label=f'Delsys {var}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title("Graph of all variables in Delsys")
        plt.legend()
        plt.show()
        
        plt.figure(figsize=(12, 8))
        for var in ['signal', 'filtered_signal', 'envelope', 'rms_envelope']:
            plt.plot(df_mdurance['time'], df_mdurance[var], label=f'mDurance {var}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title("Graph of all variables in mDurance")
        plt.legend()
        plt.show()
            
    return True

# Function to calculate statistics of patient data
def calculate_statistics(patient):
    """Perform RMSE, FastDTW & cross-correlation analysis between two RMS envelopes"""
    # Lists to store maximum correlations
    max_corr_continuous = []
    max_corr_repetitions = []
    rmse_continuous = []
    rmse_repetitions = []
    dtw_continuous = []
    dtw_repetitions = []
    output_file = 'complete_statistical_results2.txt'
            
    with open(output_file, 'w') as f:
        if patient == 'all':
            patients = list_patients()
        else:
            patients = [patient]
        
        for patient in patients:
            f.write(f"\nProcessing statistics for patient: {patient}\n")
            print(f"\nProcessing statistics for patient: {patient}")

            # Iterate through repetition types
            for rep_type in repetition_type:
                f.write(f"\nProcessing repetition type: {rep_type}\n")
                print(f"\nProcessing repetition type: {rep_type}")

                # Get the correct DataFrame based on the repetition type
                df_delsys = patient_data[f'Delsys {rep_type}'][patient]
                df_mdurance = patient_data[f'mDurance {rep_type}'][patient]
    
                # Calculate RMSE
                rmse = calculate_rmse(df_delsys['rms_envelope'], df_mdurance['rms_envelope'])
                if rep_type == "Continuous Repetition":
                    rmse_continuous.append(rmse)
                else:
                    rmse_repetitions.append(rmse)
                f.write(f"RMSE between Delsys and mDurance signals: {rmse}\n")
                print(f"RMSE between Delsys and mDurance signals: {rmse}")

                # Calculate DTW
                dtw_distance = calculate_dtw(df_delsys['rms_envelope'], df_mdurance['rms_envelope'])
                if rep_type == "Continuous Repetition":
                    dtw_continuous.append(dtw_distance)
                else:
                    dtw_repetitions.append(dtw_distance)
                f.write(f"DTW distance between Delsys and mDurance signals: {dtw_distance}\n")
                print(f"DTW distance between Delsys and mDurance signals: {dtw_distance}")
                    
                # Calculate cross-correlation
                f.write("Calculating cross-correlation between Delsys and mDurance...\n")
                print('Calculating cross-correlation between Delsys and mDurance...')
                correlation = calculate_cross_correlation(df_delsys, df_mdurance)
                if rep_type == "Continuous Repetition":
                    max_corr_continuous.append(correlation)
                else:
                    max_corr_repetitions.append(correlation)
                f.write(f"Maximum cross-correlation between Delsys and mDurance signals: {correlation}\n")
                print(f"Maximum cross-correlation between Delsys and mDurance signals: {correlation}")

    # Prepare data for bar plot
    metrics_data = {
        "RMSE Continuous": rmse_continuous,
        "RMSE Repetitions": rmse_repetitions,
        "FastDTW Continuous": dtw_continuous,
        "FastDTW Repetitions": dtw_repetitions,
        "Cross-Correlation Continuous": max_corr_continuous,
        "Cross-Correlation Repetitions": max_corr_repetitions
    }
    
    # Calculate mean and standard deviation for each metric
    means = {metric: np.mean(values) for metric, values in metrics_data.items()}
    std_devs = {metric: np.std(values) for metric, values in metrics_data.items()}

    # Define metric groups
    rmse_metrics = ["RMSE Continuous", "RMSE Repetitions"]
    dtw_metrics = ["FastDTW Continuous", "FastDTW Repetitions"]
    corr_metrics = ["Cross-Correlation Continuous", "Cross-Correlation Repetitions"]

    # Set up the figure with 3 horizontal plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot RMSE
    axes[0].bar(rmse_metrics, [means[m] for m in rmse_metrics], yerr=[std_devs[m] for m in rmse_metrics], capsize=5, color=['blue', 'green'])
    axes[0].set_title("RMSE (Mean ± Std)")
    axes[0].set_ylabel("Metric Value")
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Plot FastDTW
    axes[1].bar(dtw_metrics, [means[m] for m in dtw_metrics], yerr=[std_devs[m] for m in dtw_metrics], capsize=5, color=['red', 'orange'])
    axes[1].set_title("FastDTW (Mean ± Std)")
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # Plot Cross-Correlation
    axes[2].bar(corr_metrics, [means[m] for m in corr_metrics], yerr=[std_devs[m] for m in corr_metrics], capsize=5, color=['purple', 'brown'])
    axes[2].set_title("Cross-Correlation (Mean ± Std)") 
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)

    # Show plot
    plt.tight_layout()
    plt.show()

    print('All statistics have been calculated.')

    return True


# Function to calculate the cross-correlation between two signals
def calculate_cross_correlation(df_delsys, df_mdurance):
    # Ensure that the signals in the dataframes are normalized before calculating correlation
    env_delsys = df_delsys['rms_envelope'].values  # Assuming 'rms_envelope' is the relevant column
    env_mdurance = df_mdurance['rms_envelope'].values  # Assuming 'rms_envelope' is the relevant column

    # Use the ccf function from statsmodels to calculate cross-correlation
    ccf_values = sm.tsa.ccf(env_delsys, env_mdurance, adjusted=False)

    # Find the maximum cross-correlation (maximum value of the ccf function)
    max_ccf = np.max(ccf_values)

    return max_ccf

# Function to perform Dynamic Time Warping (DTW)
def calculate_dtw(signal1, signal2):
    """Calculate the Fast Dynamic Time Warping (FDTW) distance between two signals."""
    # Ensure signals are converted to NumPy arrays and reshaped
    signal1 = signal1.to_numpy().reshape(-1, 1)
    signal2 = signal2.to_numpy().reshape(-1, 1)
    distance, _ = fastdtw(signal1, signal2, dist=euclidean)
    return distance

# Function to calculate Root Mean Square Error (RMSE)
def calculate_rmse(signal1, signal2):
    """Calculate the Root Mean Square Error (RMSE) between two signals."""
    return np.sqrt(np.mean((signal1 - signal2) ** 2))

# Function to list patients in the database
def list_patients():
    l1 = list(patient_data["Delsys Continuous Repetition"].keys())
    l2 = list(patient_data["Delsys 5 Repetitions"].keys())
    l3 = list(patient_data["mDurance Continuous Repetition"].keys())
    l4 = list(patient_data["mDurance 5 Repetitions"].keys())
    
    patients = sorted(list(set(l1) & set(l2) & set(l3) & set(l4)))
    
    return patients

# Function to display patients in the database
def show_patients():
    print("Patients in the database:")
    print(list_patients())
    
    return True

# Function to exit the program
def exit_program():
    print('Exiting the program...')
    return False

# Function to remove peaks from a signal
def remove_peaks(patient, repetition_type):
    """
    Removes initial data (before start_time) for the selected patients.
    """
    print(f"Removing initial data for {patient}...")
    
    if patient == 'all':
        patients = list_patients()
    else:
        patients = [patient]

    for patient in patients:
        for rep_type in repetition_type:
            t = f'Delsys {rep_type}'
            t1 = f'mDurance {rep_type}'
            if patient in patient_data[t] and patient in patient_data[t1]:
                df_delsys = patient_data[t][patient]
                df_mdurance = patient_data[t1][patient]
                
                # Remove initial data
                df_delsys = remove_initial_data(df_delsys, start_time=3)
                df_mdurance = remove_initial_data(df_mdurance, start_time=3)
                
                # Save the trimmed data
                patient_data[t][patient] = df_delsys
                patient_data[t1][patient] = df_mdurance
                print(f"Initial data removed for {patient} in {rep_type}.")
            else:
                print(f"No data found for {patient} in category {rep_type}.")
    
    return True
    

# Function to calculate the cross-correlation between two signals
def calculate_cross_correlation(df_delsys, df_mdurance):
    # Ensure that the signals in the dataframes are normalized before calculating correlation
    env_delsys = df_delsys['rms_envelope'].values  # Assuming 'rms_envelope' is the relevant column
    env_mdurance = df_mdurance['rms_envelope'].values  # Assuming 'rms_envelope' is the relevant column

    # Use the ccf function from statsmodels to calculate cross-correlation
    ccf_values = sm.tsa.ccf(env_delsys, env_mdurance, adjusted=False)

    # Find the maximum cross-correlation (maximum value of the ccf function)
    max_ccf = np.max(ccf_values)

    return max_ccf

# Function to list patients in the database
def list_patients():
    l1 = list(patient_data["Delsys Continuous Repetition"].keys())
    l2 = list(patient_data["Delsys 5 Repetitions"].keys())
    l3 = list(patient_data["mDurance Continuous Repetition"].keys())
    l4 = list(patient_data["mDurance 5 Repetitions"].keys())
    
    patients = sorted(list(set(l1) & set(l2) & set(l3) & set(l4)))
    
    return patients

# Function to display patients in the database
def show_patients():
    print("Patients in the database:")
    print(list_patients())
    
    return True

# Function to exit the program
def exit_program():
    print('Exiting the program...')
    return False

# Function to remove peaks from a signal
def remove_peaks(patient, repetition_type):
    """
    Removes initial data (before start_time) for the selected patients.
    """
    print(f"Removing initial data for {patient}...")
    
    if patient == 'all':
        patients = list_patients()
    else:
        patients = [patient]

    for patient in patients:
        for rep_type in repetition_type:
            t = f'Delsys {rep_type}'
            t1 = f'mDurance {rep_type}'
            if patient in patient_data[t] and patient in patient_data[t1]:
                df_delsys = patient_data[t][patient]
                df_mdurance = patient_data[t1][patient]
                
                # Remove initial data
                df_delsys = remove_initial_data(df_delsys, start_time=3)
                df_mdurance = remove_initial_data(df_mdurance, start_time=3)
                
                # Save the trimmed data
                patient_data[t][patient] = df_delsys
                patient_data[t1][patient] = df_mdurance
                print(f"Initial data removed for {patient} in {rep_type}.")
            else:
                print(f"No data found for {patient} in category {rep_type}.")
    
    return True

# Function to create a menu to choose loading data, filtering, normalizing, synchronizing, plotting, calculating statistics, and displaying patient names in the database
def menu():
    options = {
        1: load_data,
        2: show_patients,
        3: plot_data,
        4: normalize_data,
        5: synchronize_data,
        6: remove_peaks,
        7: calculate_statistics,
        0: exit_program
    }
    
    while True:
        print("|----------------------------|")
        print("|            Menu:           |")
        print("|----------------------------|")
        print("| 1. Load data               |")
        print("| 2. Show patients           |")
        print("| 3. Plot data               |")
        print("| 4. Normalize data          |")
        print("| 5. Synchronize data        |")
        print("| 6. Remove peaks            |")
        print("| 7. Calculate statistics    |")
        print("| 0. Exit                    |")
        print("|----------------------------|")
        
        try:
            choice = int(input('Enter an option: '))
            if choice in options:
                if choice == 0:
                    print("Exiting the program...")
                    break
                if choice in [5, 6]:
                    to_all = input('Apply to all patients? (yes/no): ').strip().lower()
                    if to_all in ['yes', 'y']:
                        continue_execution = options[choice]('all', repetition_type)
                    else:
                        while True:
                            patient = input("Enter the patient's name: ")
                            if patient in list_patients():
                                break
                            else:
                                print(f"Error: The patient '{patient}' does not exist in the database.")
                        continue_execution = options[choice](patient, repetition_type)
                elif choice in [4, 7]:
                    to_all = input('Apply to all patients? (yes/no): ').strip().lower()
                    if to_all in ['yes', 'y']:
                        continue_execution = options[choice]('all')
                    else:
                        while True:
                            patient = input("Enter the patient's name: ")
                            if patient in list_patients():
                                break
                            else:
                                print(f"Error: The patient '{patient}' does not exist in the database.")
                        continue_execution = options[choice](patient)
                else:
                    continue_execution = options[choice]()
                if continue_execution is False:
                    print("Returning to the main menu...")
                    continue
            else:
                print('Invalid option. Please choose a valid option.')
        except ValueError:
            print('Please enter an integer.')
    
    print('Program finished.')


# Main function
if __name__ == "__main__":
    menu()