import logging
import os
from typing import Any, List, Union

import yaml
import json
from hashlib import md5
from polars import DataFrame


from utils.util_functions import serialize, deserialize, resample_multichannel
from utils.decorators import with_session_key

from redis import Redis
redis_client = Redis(host='localhost', port=6379, decode_responses=False)


class SessionData():
    def __init__(self, _data_type, _name):
        self._data_type = _data_type
        self._name = _name
        self._called = False

    def init_model_list(self):
        with open("default_models.yml", 'r') as stream:
            try:
                default_loaded_models = yaml.safe_load(stream)
                self.set(json.dumps(default_loaded_models))
            except yaml.YAMLError:
                logging.warning("No default models loaded in session.")

    @with_session_key()
    def get(self, name, session_key=None):
        if self._name == "selected_model_list":
            models = redis_client.lrange(f"{session_key}_{self._name}:selected_model_list", 0, -1)
            models = [name.decode('utf-8') for name in models]
            if len(models) > 0:
                return models[0]
            else:
                self.init_model_list()
                return self.get("", session_key=session_key)
        
        parquet_bytes = redis_client.get(f"{session_key}_{self._name}:{name}")
        if parquet_bytes is None:
                return None
        return deserialize(parquet_bytes)

    @with_session_key()
    def __len__(self, session_key=None):
        if self._name == "loaded_labels":
            #labels = self.names(session_key=session_key)
            labels = redis_client.get(f"{session_key}_{self._name}")
            if labels is None:
                return 0
            labels = deserialize(labels)
            return len(labels)
        return len(self.keys(session_key=session_key))
    
    @with_session_key()
    def get_all(self, session_key=None):
        if self._name != "loaded_labels":
            keys = self.keys(session_key=session_key)
            keys_with_prefix = [f"{session_key}_{self._name}:{k}" for k in keys]
            values = redis_client.mget(keys_with_prefix)
            return {key: deserialize(df) for key, df in values}
        else:
            values = redis_client.get(f"{session_key}_{self._name}")
            if values is None:
                return DataFrame()
            return deserialize(values)
        
    
    @with_session_key()
    def __or__(self, data, session_key=None):
        data_with_prefix = { f"{session_key}_{self._name}:{k}": v for k, v in data.items() }
        _ = redis_client.mset(data_with_prefix)
        redis_client.lpush(f"{session_key}_{self._name}:keys", list(data.keys()))

    @with_session_key()
    def set(self, data, session_key=None):
        if isinstance(data, DataFrame):
            data = serialize(data)
        if self._name == "selected_model_list":
            redis_client.delete(f"{session_key}_{self._name}:selected_model_list")
            redis_client.lpush(f"{session_key}_{self._name}:selected_model_list", data)
        else:
            redis_client.set(f"{session_key}_{self._name}", data)

    @with_session_key()
    def keys(self, session_key=None):
        return self.names(session_key=session_key)
    
    @with_session_key()
    def names(self, session_key=None):
        names = redis_client.lrange(f"{session_key}_{self._name}:keys", 0, -1)
        return [name.decode('utf-8') for name in names]
    
    @with_session_key()
    def append(self, name, data, session_key=None):
        redis_client.set(f"{session_key}_{self._name}:{name}", serialize(data))
        
        if self._name == "loaded_ecgs":
            all_keys = set(self.names(session_key=session_key))
            if name not in all_keys:
                redis_client.lpush(f"{session_key}_{self._name}:keys", name)
        
    @with_session_key()
    def extend(self, df_dict: dict[str, DataFrame], session_key=None):
        ser_df_dict = {f"{session_key}_{self._name}:{key}": serialize(x) for key, x in df_dict.items()}
        redis_client.mset(ser_df_dict)
        
        if self._name == "loaded_ecgs":
            all_keys = set(self.names(session_key=session_key))
            append_keys = set(df_dict.keys()) - all_keys
            if append_keys is not None:
                redis_client.lpush(f"{session_key}_{self._name}:keys", *append_keys)

    @with_session_key()
    def is_empty(self, session_key=None):
        if self._name != "loaded_labels":
            return not redis_client.exists(f"{session_key}_{self._name}:keys")
        else:
            return not redis_client.exists(f"{session_key}_{self._name}")
    
    @with_session_key()
    def clear_session_storage(self, session_key=None):
        for key in redis_client.scan_iter(f"{session_key}_{self._name}*"):
            redis_client.delete(key)
    
    @with_session_key()
    def compute_hash(self, session_key=None):
        """Compute a hash to detect any new data added."""
        keys = self.keys(session_key=session_key)  # Get all keys
        # Sort keys and join them into a single string to maintain consistency
        keys_string = ",".join(sorted(keys))
        return md5(keys_string.encode(), usedforsecurity=False).hexdigest()
    
    @with_session_key()
    def set_hashed(self, hash, data, session_key=None):
        if isinstance(data, DataFrame):
            data = serialize(data)
        redis_client.set(f"{session_key}_{self._name}:{hash}", data)


class ECGData_Dummy():
    def __init__(
        self,
        target_sampling_rate: int = 100,
        batch_size = 8
    ) -> None:
        self.num_cores = os.cpu_count()
        self.batch_size = batch_size
        self.target_sampling_rate = target_sampling_rate

        self.loaded_ecgs = SessionData({}, "loaded_ecgs")
        self.loaded_labels = SessionData(DataFrame(), "loaded_labels")

    def get_all(self, session_key, **kwargs):
        if not self.loaded_ecgs.is_empty(session_key=session_key):
            for name in self.names(session_key=session_key):
                yield name, self.get_by_name(name, session_key=session_key, **kwargs)
        else: return {}

    def get_all_labels(self, session_key) -> DataFrame:
        return self.loaded_labels.get_all(session_key=session_key)
                
    def get_by_name(
        self,
        name: str,
        session_key,
        **kwargs: Any
    ) -> Union[DataFrame, None]:
        if isinstance(name, list):
            data = [self.get_by_name(n, session_key=session_key, **kwargs) for n in name]
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
                    case _:
                        pass
            
        return data
    
    def statistics(self, session_key) -> dict:
        return {
            'ecgs': self.loaded_ecgs.__len__(session_key=session_key),
            'labels': self.loaded_labels.__len__(session_key=session_key)
        }
    
    def label_statistics(self, session_key, icd_codes=False) -> List[str]:
        if self.loaded_labels.is_empty(session_key=session_key): 
            return list({})
        if icd_codes:
            return list(self.loaded_labels.get_all(session_key=session_key)["ICD-10"]) 
        else:
            return list(self.loaded_labels.get_all(session_key=session_key)["label"])

    def names(self, session_key) -> List[str]:
        return sorted(list(self.loaded_ecgs.keys(session_key=session_key)))
    