
import numpy as np
from polars import (DataFrame, Float32)

from rlign import Rlign, Template


class Rlign_wrapper():
    def __init__(
        self,
        sampling_rate: int,
        xai_model_path: str = "xai_models/"
    ) -> None:
        self.normalizer = Rlign(
            sampling_rate=sampling_rate,
            agg_beat="none",
            num_workers=1,
            template_bpm=60
        )

        self.median_normalizer = Rlign(
            sampling_rate=sampling_rate,
            agg_beat="median",
            num_workers=1,
            template_bpm=60
        )

        self.xai_model_path = xai_model_path
        self.xai_session_model = ''
    

    def run_Rlign(self, data: np.ndarray, median=False, xai=False) -> list[DataFrame]:
        if len(data.shape) < 3: data = np.expand_dims(data, 0)
        if data.shape[1] > data.shape[2]: data = data.transpose((0,2,1))
        if median:
            normalized_ecg = self.median_normalizer.transform(data)
            if xai:
                normalized_ecg = normalized_ecg.reshape(1, normalized_ecg.shape[0], -1)
                normalized_ecg = normalized_ecg.astype(np.float32)
            else:
                normalized_ecg = DataFrame(normalized_ecg[0])
        else:
            normalized_ecg = self.normalizer.transform(data)
            normalized_ecg = normalized_ecg.transpose((0,2,1))
            normalized_ecg = [DataFrame(out).cast(Float32()) for out in normalized_ecg]
        return normalized_ecg