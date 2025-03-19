
import json
import logging
import os
from io import BytesIO
from hashlib import md5
from typing import Any, Generator, List, Union

import numpy as np
from numpy import load as np_load
from numpy import frombuffer as np_frombuffer
from numpy import ndarray as np_ndarray
from numpy import vstack
from scipy.io import loadmat
import xml.etree.ElementTree as ET
from werkzeug.datastructures import FileStorage

from polars import (DataFrame, Float64, concat, read_csv)
from polars import col as pl_col
from pydicom import dcmread
from pydicom.waveforms import multiplex_array


from utils.session_data import ECGData_Dummy, SessionData
from utils.ecg_events import ECGEvents
from utils.onnx_runtimes import ONNX_Runtimes
from utils.rlign_wrapper import Rlign_wrapper
from utils.util_functions import checkScale, normalize_scale, remove_outliers, resample_multichannel

class ECGData(ECGData_Dummy):
    def __init__(
        self,
        target_sampling_rate: int = 100,
        lead_layout_list: List[str] = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        batch_size = 8
    ) -> None:
        super().__init__(target_sampling_rate, batch_size) 

        self.lead_layout_list = lead_layout_list

        self.onnx_runtimes = ONNX_Runtimes(
            sampling_rate=self.target_sampling_rate
        )
        self.ecg_events = ECGEvents(
            sampling_rate=self.target_sampling_rate
        )
        self.rlign = Rlign_wrapper(
            sampling_rate=self.target_sampling_rate
        )

        # storages
        self.initialize_storages()


    def upload_demo_data(self, session_key=None):
        try:
            self.upload_labels(labels=["example_data/BBB.csv"], session_key=session_key)
            for file_path in os.listdir("example_data/"):
                if file_path == "BBB.csv": continue
                with open(f"example_data/{file_path}", 'rb') as file:
                    self.upload(
                        files = (
                            file.name,
                            ".csv",
                            FileStorage(
                                file,
                                filename=file.name,
                                content_type='application/octet-stream'
                            ).read()
                        ),
                        sampling_rate=100,
                        session_key=session_key
                    )
        except Exception as e:
            logging.warning(e)

    def register_prediction_task(self, run_prediction_task):
        self.run_prediction_task = run_prediction_task
    def register_upload_data_task(self, upload_data_task):
        self.upload_data_task = upload_data_task

    def clear_session_storage(self, session_key=None) -> None:
        self.loaded_ecgs.clear_session_storage(session_key=session_key)
        self.loaded_ecgs_qrs.clear_session_storage(session_key=session_key)
        self.loaded_ecgs_events.clear_session_storage(session_key=session_key)
        self.loaded_ecgs_Rlign.clear_session_storage(session_key=session_key)
        self.loaded_labels.clear_session_storage(session_key=session_key)
        self.loaded_prediction_statistics.clear_session_storage(session_key=session_key)

    def initialize_storages(self) -> None:
        
        self.loaded_ecgs_qrs = SessionData({}, "loaded_ecgs_qrs")
        self.loaded_ecgs_events = SessionData({}, "loaded_ecgs_events")
        self.loaded_ecgs_Rlign = SessionData({}, "loaded_ecgs_Rlign")
        self.loaded_prediction_statistics = SessionData({}, "loaded_prediction_statistics")

        # load default model list
        self.selected_model_list = SessionData({}, "selected_model_list")
        

    def save_selected_model_list(self, model_list):
        model_list = json.dumps(model_list)
        self.selected_model_list.set(model_list)
        
    def available_prediction_models(self) -> List[str]:
        checked_models = json.loads(self.selected_model_list.get("selected_model_list"))

        prediction_models = [m for m in checked_models['prediction_models'] if m is not None and m.endswith(".onnx")]
        exchange_models = [f"*{m}" for m in checked_models['exchange_models'] if m is not None and m.endswith(".onnx")]

        all_checked_models = [*prediction_models, *exchange_models]
        aval_models = [
            m.removesuffix('.onnx') for m in 
            all_checked_models
        ]
        return [] if len(aval_models) == 0 else sorted(aval_models) 
    
    def available_xai_models(self) -> List[str]:
        return [
            m.removesuffix('.onnx') for m in 
            os.listdir(self.onnx_runtimes.xai_prediction_model_path)
        ]
    
    def available_training_models(self) -> List[str]:
        checked_models = json.loads(self.selected_model_list.get("selected_model_list"))

        training_models = [m for m in checked_models['training_models'] if m is not None and m.endswith(".onnx")]
        exchange_models = [f"*{m}" for m in checked_models['exchange_models'] if m is not None and m.endswith(".onnx")]

        all_checked_models = [*training_models, *exchange_models]

        aval_models = [
            ".".join(m.split('.')[:-1]) for m in 
            all_checked_models
        ]

        return [] if len(aval_models) == 0 else sorted(aval_models) 

    def upload(
        self,
        files,
        sampling_rate: int = 500,
        adc_gain: int = 1000,
        lead_layout: str = 'leads_default',
        session_key = None
    ) -> None:
        filename, filetype, file_content = files
        try:
            data = self.__read_by_filetype(filetype, file_content, sampling_rate, adc_gain, lead_layout)
            if data is not None:
                self.loaded_ecgs.append(filename, data, session_key=session_key)
        except Exception as e:
            logging.warning(e)
        

    def upload_labels(self, labels) -> None:
        labels = concat([read_csv(file, infer_schema=False) for file in labels])

        if len(labels.columns) != 3 or (labels.columns != ["file_name", "label", "ICD-10"]):
            raise Exception("Wrong data format of labels")
        
        self.loaded_labels.set(labels)

    def get_by_name(
        self,
        name: str,
        require_transform: str = None,
        session_key=None,
        **kwargs: Any
    ) -> Union[DataFrame, None]:
        if isinstance(name, list):
            data = [self.get_by_name(n, require_transform, session_key=session_key, **kwargs) for n in name]
        else:
            if name == 'No data': return None
            data = self.loaded_ecgs.get(name, session_key=session_key)

            if data is None: return None

            # resample if required
            for key, value in kwargs.items():
                match key:
                    case 'target_sampling_rate':
                        if value != self.target_sampling_rate:
                            data = DataFrame(resample_multichannel(
                                xs=data.to_numpy(),
                                fs=self.target_sampling_rate,
                                fs_target=value
                            ))
                    case 'xai':
                        return self.rlign.run_Rlign(data, median=True, xai=value)
                    case _:
                        pass
            
            # transform if required
            match require_transform:
                case 'qrs':
                    data_ = self.loaded_ecgs_qrs.get(name, session_key=session_key)
                    if data_ is not None: return data_
                    data = self.ecg_events.run_qrs(data)
                    self.loaded_ecgs_qrs.append(name, data)
                
                case 'Rlign':
                    data_ = self.loaded_ecgs_Rlign.get(name, session_key=session_key)
                    if data_ is not None: return data_
                    data = self.rlign.run_Rlign(data)[0]
                    self.loaded_ecgs_Rlign.append(name, data)

                case 'Rlign-MedianBeats':
                    #data_ = self.loaded_ecgs_Rlign.get(name)
                    #if data_ is not None: return data_
                    data = self.rlign.run_Rlign(data, median=True, xai=False)
                    #self.loaded_ecgs_Rlign.append(name, data)

                case 'events':
                    data_ = self.loaded_ecgs_events.get(name, session_key=session_key)
                    if data_ is not None: return data_
                    data = self.get_by_name(name, 'Rlign', session_key=session_key)
                    if data is not None:
                        data = data.fill_nan(0.0)
                        data = self.ecg_events.run_events(data)
                        self.loaded_ecgs_events.append(name, data)

                case _:
                    pass

        return data 
        
    def __read_by_filetype(
        self,
        filetype,
        file_content,
        sampling_rate: int = 500,
        adc_gain: int = 1000,
        lead_layout: str = 'leads_default'
    ) -> Union[DataFrame, None]:
        try:
            header = None
            match filetype:
                case '.csv':
                    # This may has a header
                    data = read_csv(file_content, has_header=False)
                    if all(isinstance(dt, str) for dt in data.row(0)):
                        header = data.row(0)
                        data = data[1:]
                        data.columns = header
                        data = data.with_columns([pl_col(col).cast(Float64) for col in data.columns])
                        if 'column' in header[0]:  header = None
                case '.npy':
                    data = DataFrame(np_load(file_content))
                case '.dcm':
                    data = DataFrame(load_dicom(file_content))
                case '.mat':
                    data = loadmat(BytesIO(file_content), mat_dtype=True)
                    key = [key for key in data.keys() if not key.startswith('__')][0]
                    data = DataFrame(data[key])
                case '.dat':
                    data = np_frombuffer(
                        file_content,
                        dtype=np.int16
                    ).astype(np.float32).reshape(-1, 12)
                    data = DataFrame(data).interpolate().fill_nan(0)
                case '.xml':
                    data = load_xml(file_content, self.lead_layout_list).reshape(-1, 12)
                    data = DataFrame(data).interpolate().fill_nan(0)
                case _:
                    # should never happen
                    raise Exception("Not able to load data!")
            
            data = data * (1000/adc_gain)

            if 12 not in data.shape:
                raise Warning(f"Not supported data format of ECGs: {data.shape}")
            
            if data.shape[0] == 12:
                data = data.transpose()
            
            auto_sampling_rate = data.shape[0] // 10
            # Xhz to target_sampling_rate upsampling
            if sampling_rate != self.target_sampling_rate:
                # upsample to target_sampling_rate
                data = DataFrame(resample_multichannel(
                    xs=data.to_numpy(),
                    fs=sampling_rate,
                    fs_target=self.target_sampling_rate
                ))


            # crop to normal 10s interval data
            data_len_offset = max(data.shape) - self.target_sampling_rate * 10
            if data_len_offset > 0:
                data = data[:self.target_sampling_rate * 10]
            elif data_len_offset < 0:
                padding = data[:data_len_offset].to_numpy()
                data = vstack((data.to_numpy(), padding))
                data = data.transpose()
                data = DataFrame(data)

            if data.shape == (12, self.target_sampling_rate * 10):
                data = data.transpose()

            # reorder leads
            if lead_layout != 'leads_default' and header is None:
                match lead_layout:
                    case 'leads_mimic':
                        data = data[:, [0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11]]
                    case _:
                        raise Exception('No such lead order')
            elif header is not None:
                data = data[:, [self.lead_layout_list.index(h) for h in header]]


            data = normalize_scale(data, checkScale(data.to_numpy()), "milliVolt")
            data = remove_outliers(data)
            return data
        except Exception as e:
            logging.warning(e)
    
    def prediction_statistics(self, prediction_model, exchange_name="default", xai=False, session_key=None) -> dict[str, str]:
        # generate hash and check if existing
        if self.loaded_ecgs.is_empty(session_key=session_key):
            return DataFrame()
        
        prediction_model_data_hash = md5(prediction_model.encode()).hexdigest() + self.loaded_ecgs.compute_hash(session_key=session_key)
        statistics = self.loaded_prediction_statistics.get(prediction_model_data_hash, session_key=session_key)

        if statistics is not None:
            return statistics
        
        statistics = DataFrame(
            run_prediction_model(
                self.run_prediction_task,
                prediction_model=prediction_model,
                names=self.names(session_key=session_key),
                xai=xai,
                get_class=True,
                session_key=session_key,
                exchange_name=exchange_name
            )
        )
        if not xai and statistics is not None and not statistics.is_empty(): 
            self.loaded_prediction_statistics.set_hashed(prediction_model_data_hash, statistics, session_key=session_key)
        return statistics


    
def load_dicom(file) -> np_ndarray:
    data = dcmread(BytesIO(file.read()), force=True)
    return multiplex_array(data, 0, as_raw=True)



def load_xml(file, lead_layout_list) -> np_ndarray:
    data = ET.parse(file).getroot().find('StripData')
    leads_data = {lead: [] for lead in lead_layout_list}
    meta_data = {}

    # Navigate through the XML structure and extract data
    for record in data:  # Update tag based on actual structure
        tag = record.tag
        meta_data[tag] = record.text

        if tag == 'WaveformData':
            lead = record.attrib.values().mapping['lead']
            leads_data[lead] = np.asarray(record.text.replace('\t','').split(','), dtype=np.int32)
    return np.asarray([x for x in leads_data.values()])

def batch_list(
    data: List,
    batch_size: int = 8
) -> Generator:
    batch = []
    for d in data:
        batch.append(d)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch



def run_prediction_model(
    run_prediction_task,
    prediction_model: str,
    names: List,
    xai: bool = False,
    session_key = None,
    batch_size=32,
    get_class=False,
    exchange_name="default"
) -> DataFrame:
    
    def predict_ecg(batched_names, get_class=False, prediction_model=None, session_key=session_key, exchange_name="default"):
        ft = run_prediction_task.delay(batched_names, get_class=get_class, prediction_model=prediction_model, xai=xai, session_key=session_key, exchange_name=exchange_name)
        return ft
    
    predictions = {}
    futures = [predict_ecg(batch, get_class=get_class, prediction_model=prediction_model, session_key=session_key, exchange_name=exchange_name) for batch in batch_list(names, batch_size=batch_size)]
    for ft in futures:
        if isinstance(ft, list):
            for f in ft:
                predictions.update( f.get() )
        else:
            predictions.update( ft.get() )
        
    return DataFrame(predictions)