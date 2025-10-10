import logging
import os
from typing import Any, List, Union

from werkzeug.utils import secure_filename
import numpy as np
from onnxruntime import (get_available_providers, InferenceSession, SessionOptions)
#from onnxruntime.capi.onnxruntime_pybind11_state import SessionOptions
from polars import (DataFrame)
from utils.util_functions import ecg_normalize, softmax

class ONNX_Runtimes():
    def __init__(self,
        exchange_model_path: str = 'exchange_models/',
        prediction_model_path: str = 'models/',
        xai_prediction_model_path: str = 'xai_models/',
        training_model_path: str = 'training_models/',
        sampling_rate: int = 100
    ) -> None:
        self.exchange_model_path = exchange_model_path
        self.prediction_model_path = prediction_model_path
        self.xai_prediction_model_path = xai_prediction_model_path
        self.training_model_path = training_model_path
        self.prediction_session_model = ''
        self.sampling_rate = sampling_rate
        self.standardizer = None
        self.scale = "milliVolt"

        self.sess_options = SessionOptions()
        self.sess_options.log_severity_level = 3

    def upload_models(self, models) -> None:
        for file in models:
            file.save(os.path.join(self.prediction_model_path, secure_filename(file.filename)))

    def load_prediction_model(self, prediction_model: str, xai: bool, exchange_name: str):
        if xai:
            model_path = f'{self.xai_prediction_model_path}{prediction_model}.onnx'
        elif prediction_model.startswith("*"):
            model_path = f'{self.exchange_model_path}{exchange_name}/{prediction_model[1:]}.onnx'
        else:
            model_path = f'{self.prediction_model_path}{prediction_model}.onnx'

        if not os.path.isfile(model_path):
            raise Exception(f"No such prediction model {model_path}, {prediction_model},!")
    
        if model_path == self.prediction_session_model:
            return
        
        # load if different model
        self.prediction_session_model = model_path

        
        self.prediction_session = InferenceSession(
            self.prediction_session_model,
            providers=get_available_providers(),
            sess_options=self.sess_options
        )
        session_data = self.prediction_session.get_inputs()[0]
        self.prediction_input_name = session_data.name
        self.prediction_input_shape = tuple(session_data.shape[1:]) #ignore batchsize

        output_shape = self.prediction_session.get_outputs()[0].shape[1:]
        if not xai:
            try:
                self.classes = np.asarray(eval(self.prediction_session.get_modelmeta().custom_metadata_map.get('target_keys', None)))
            except Exception as e:
                logging.warning(f'Failed to extract class names from model: {e}; Using generic classes')
                
                num_classes = output_shape[0]
                if len(output_shape) == 1: #only classes, no probabilities
                    self.class_wo_prop = True
                else:
                    self.class_wo_prop = False
                    if self.prediction_input_shape[1] > self.prediction_input_shape[0]:
                        num_classes = output_shape[1]
                self.classes = np.asarray([f"generic_class_{i}" for i in range(num_classes)])

            try: self.standardizer = self.prediction_session.get_modelmeta().custom_metadata_map.get('standardizer', None)
            except Exception as e: self.standardizer = None

            try: self.scale = self.prediction_session.get_modelmeta().custom_metadata_map.get('scale', "milliVolt")
            except Exception as e: self.scale = "milliVolt"
    
    def run_prediction(
        self,
        data: Union[DataFrame, List[DataFrame]],
        get_class=False,
        **kwargs: Any
    ) -> Union[dict[str, str], None]:
        prediction_model = kwargs.get('prediction_model')
        xai = kwargs.get('xai', False)
        exchange_name = kwargs.get('exchange_name', "default")
        
        self.load_prediction_model(prediction_model, xai, exchange_name)
        
        if not xai:
            data = ecg_normalize(
                data,
                sampling_rate = self.sampling_rate,
                trend_removal ="movingMedian",
                standardizer = self.standardizer,
                scale = self.scale
            )
        else:
            data = ecg_normalize(
                data,
                trend_removal=None,
                standardizer="meanStd"
            )

        if isinstance(data, list):
            data = np.asarray(data)

        # add batch_size if necessary
        if data.ndim < 3:
            data = np.expand_dims(data, axis=0)

        if data.shape[1:] != self.prediction_input_shape:
            data = np.transpose(data, axes=(0,2,1))
       
        try:
            outputs_logits = self.prediction_session.run(None, {self.prediction_input_name: data})[0]
            
            if not xai:
                outputs = np.apply_along_axis(softmax, axis=-1, arr=outputs_logits)
                if get_class:
                    return list(self.classes[np.argmax(outputs, axis=-1)])
                else:
                    return [
                        {self.classes[idx]: str(round(o, 3)) for idx, o in enumerate(sample_probs)}
                        for sample_probs in outputs
                    ]
                    #return {self.classes[idx]: str(round(o, 3)) for idx, o in enumerate(outputs[0])}
            else:
                #SVM directly predicts!
                return list(outputs_logits)
        except Exception as e:
            logging.warning(f'Failure to predict and map to classes: {e}')
            return None