from neurokit2 import (ecg_clean, ecg_segment, epochs_to_df, ecg_process)
from polars import (DataFrame, Series, from_pandas)
import polars as pl
import numpy as np
import logging

from utils.util_functions import ecg_normalize

class ECGEvents():
    def __init__(self, sampling_rate: int = 500) -> None:
        self.ecg_channel: int = 1
        self.sampling_rate = sampling_rate
    
    def run_qrs(self,
        data: DataFrame
    ) -> DataFrame:
        """Computed all QRS-complexes of the input ECG, and returns them on a unified DataFrame.

        Args:
            ecg (DataFrame): The input ECG to use.
            sampling_rate (int, optional): The used sampling rate of the ECG. Defaults to 500.

        Returns:
            DataFrame: All QRS-complexes as an overlay.
        """
        try:
            cleaned_ecg = ecg_clean(
                data[:, self.ecg_channel],
                sampling_rate=self.sampling_rate
            )
            qrs_epochs = ecg_segment(
                cleaned_ecg,
                rpeaks=None,
                sampling_rate=self.sampling_rate,
                show=False,
            )
        except Exception as e:
            logging.warning(f'Failure in Neurokit {e}')
            return DataFrame()
        
        qrs_epochs = epochs_to_df(qrs_epochs)
        qrs_epochs = qrs_epochs[['Signal', 'Time', 'Label']].rename({'Label': int})
        qrs_epochs.dropna()

        check = qrs_epochs[['Signal', 'Label']].groupby(['Label']).sum()
        check = check[check['Signal'] == 0.0].reset_index()['Label']

        if not check.empty:
            qrs_epochs = qrs_epochs[~qrs_epochs['Label'].isin(check.values)]


        return from_pandas(qrs_epochs)
    
    def run_events(self,
        data: DataFrame
    ) -> DataFrame:
        events = ecg_process(
            data[:, self.ecg_channel],
            sampling_rate=self.sampling_rate
        )[0][[
            'ECG_Clean',
            'ECG_R_Peaks',
            'ECG_R_Onsets',
            'ECG_R_Offsets',
            'ECG_P_Peaks',
            'ECG_P_Onsets',
            'ECG_P_Offsets',
            'ECG_Q_Peaks',
            'ECG_S_Peaks',
            'ECG_T_Peaks',
            'ECG_T_Onsets',
            'ECG_T_Offsets',
        ]]

        events = from_pandas(events)

        # Remove prefixes
        events = events.rename(lambda colname: str(colname).removeprefix('ECG_'))

        # Normalize ECG
        ecg_raw = np.expand_dims(events['Clean'].to_numpy(), axis=0)
        events = events.with_columns(Series('Clean', ecg_normalize(ecg_raw, sampling_rate=self.sampling_rate)[0]))

        # Filter only necessary points and align to signal-Y-axis
        events = events.select([
            pl.when(pl.col(col) == 0.0)
            .then(np.nan)
            .otherwise(pl.col('Clean'))
            .alias(col)
            for col in events.columns
        ])

        events = events.rename({'Clean': 'ECG_Clean'})
        return events
    
