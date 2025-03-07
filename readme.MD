# Root Folder - LSTM-GA techniques
            - Go to Scripts folder to activate (.\activate)
            - This venv will be used for all future iterations for this technique

# Folder Navigation
        The following screenshot is for reference only. It describes the folder structure of this project.
        <insert_screenshot_here> once all folders have been populated.

# Required Libraries and packages
        The required libraries and packages information is provided in `requirements.txt` file

# File Execution order
**In some modules (files are imported into other) for functionality, the filenames must not start with numbers. Otherwise the order would make more sense, but because we are importing them and there are file naming restrictions, therefore the following order must be read carefully. There is a work around for that, but keeping this ReadME updated can easily walk through** 

## File Names and description

1. **0_addingTimestamps.py** -> only to be run once IFF raw dataset is used without the timestamps
2. **filemanagement.py** -> reads all files into multiple dataframes (`based on battery chemistry`) and performs basic dataframe description on each. Then it is stored under `raw_stats` folder.
3. **correlation.py** -> reads dataframes, performs correlation analysis and makes `correlograms` using Seaborn library for each dataframe. The graphs are then saved in `raw_stats/graphs/correlation_plots` folder with `EPS` and `PNG` file extensions.
4. **rawVoltageCurves.py** -> groups dataframes by 'Current' and 'Temperature' and plots raw voltage curves, which is stored under `raw_stats/graphs/raw_voltage_curves` folder
5. **smoothing.py** -> Applies Savitzky-Golay filter for smoothing the voltages. Although this is only needed for visual appearence at this stage, however, if missing values are to be padded (due to variance in recorded values as experiments ran for different times based on battery types and conditions), smoothing can help in interpolation. 
Right now, it is only intended for side-by-side comparison. The output of this file stores graphs in `raw_stats/graphs/smoothing` folder.
6. **interpolate.py** -> uses linear interpolation to padd missing values. the code then compares and plots the values with smoothed values in the previous step. The output is stored in `raw_stats/graphs/interpolation` folder.
7. **normalization.py** -> normalized {temperature_num,soc,capacity,final_voltage,currents} columns using `MinMaxScaler()`. Dataframes are updated, where new columns are created using _norm suffix for normalized values. The Dataframes are the stored as pickle files separately in `data/normalized` folder, so that the rest of the model development can utilize these dataframes without needing to  perform the above steps.
8. **LiIon_LSTM_base.py** -> Base Keras LSTM model with 80/20 data split, and early stopping. Padded data to match any mismatch length in the recorded values of timestamps. Model training results and evaluation metrics are stored in `basic_model_stats/untuned/LiIon`
9. **LiPo_LSTM_base.py** -> Base Keras LSTM model with 80/20 data split, and early stopping. Padded data to match any mismatch length in the recorded values of timestamps. Model training results and evaluation metrics are stored in `basic_model_stats/untuned/LiPo`
10. **NiMH_LSTM_base.py** -> Base Keras LSTM model with 80/20 data split, and early stopping. Padded data to match any mismatch length in the recorded values of timestamps. Model training results and evaluation metrics are stored in `basic_model_stats/untuned/NiMH`
11. **LiIon_LSTM_KTuner.py** -> takes the base model developed in previous steps and uses Keras Tuner for hyper parameter optimization. records the model performance statistics and comparison plots in `basic_model_stats/tuned/KerasTuner/LiIon` folder.
12. **LiPo_LSTM_KTuner.py** -> takes the base model developed in previous steps and uses Keras Tuner for hyper parameter optimization. records the model performance statistics and comparison plots in `basic_model_stats/tuned/KerasTuner/LiPo` folder.
13. **NiMH_LSTM_KTuner.py** -> takes the base model developed in previous steps and uses Keras Tuner for hyper parameter optimization. records the model performance statistics and comparison plots in `basic_model_stats/tuned/KerasTuner/NiMH` folder.
14. **LiIon_LSTM_GA.py** -> takes the base model developed in previous steps and uses Genetic Algorithm for hyper parameter tuning and optimization. Records the model performance statistics and comparison plots in `basic_model_stats/tuned/GA/LiIon` folder.
15. **LiPo_LSTM_GA.py** -> takes the base model developed in previous steps and uses Genetic Algorithm for hyper parameter tuning and optimization. Records the model performance statistics and comparison plots in `basic_model_stats/tuned/GA/LiPo` folder.
16. **NiMH_LSTM_GA.py** -> takes the base model developed in previous steps and uses Genetic Algorithm for hyper parameter tuning and optimization. Records the model performance statistics and comparison plots in `basic_model_stats/tuned/GA/NiMH` folder.
17. **LiIon_LSTM_BestFit.py** -> Reads the optimized Genetic Algorithm parameters stored in the previos step to build and fit the model. The model is evaluated by plotting all the values in the dataframe. This code on its own could be useful if you **DO NOT** want to run the code from the scratch. Because, at this stage, we have already computed Best GA parameters and it takes them directly to build the best fit LSTM. This however struggles with academic style plots, and for that its extension in the form of `LiIon_LSTM_BestFit_A.py` is created. The results from BestFit are stored under `socEstimation/final_analysis/LiIon`.
