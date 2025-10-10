import json
import os
import tempfile
import zipfile

from celery import Task
from polars import Series

from utils.data import ECGData_Dummy
from .finetuning import Finetune


class FinetuneWrapper(Task):
    name = 'finetune_wrapper_task'
        
    def run(
        self,
        session_key,
        base_model: str,
        model_name: str,
        train_method,
        optimizer,
        epochs,
        lr,
        lr_gamma,
        batchsize,
        show_logs,
        target_sampling_rate,
        exchange_name
    ):
        model_name = model_name
        session_key = session_key
        task_id = self.request.id

        ecg_data = ECGData_Dummy(target_sampling_rate, batchsize)
        target_sampling_rate = target_sampling_rate

        label_stats = Series(ecg_data.label_statistics(session_key=session_key)).value_counts().to_dict(as_series=False)
        label_stats_icd10 = Series(ecg_data.label_statistics(session_key=session_key, icd_codes=True)).value_counts().to_dict(as_series=False)
        label_stats_full = ecg_data.label_statistics(session_key=session_key)
        data_stats = ecg_data.statistics(session_key=session_key)
        
        session_storage = {
            "stats": {
                'model_name': model_name,
                'base_model': base_model,
                'optimizer': optimizer,
                'label_stats': label_stats,
                'label_stats_icd10': label_stats_icd10,
                'data_stats': data_stats,
            },
            "show_logs": show_logs
        }

        
        loaded_ecgs = {key: x for key,x in ecg_data.get_all(session_key=session_key, kwargs={'target_sampling_rate': target_sampling_rate})}
        loaded_labels = ecg_data.get_all_labels(session_key=session_key)


        self.update_state(state="RUNNING")
        try:
            finetune_stats = Finetune(sampling_rate=target_sampling_rate).run_finetune(
                base_model = base_model,
                model_name = model_name,
                loaded_ecgs = loaded_ecgs,
                loaded_labels = loaded_labels,
                label_statistics = label_stats_full,
                task_id = task_id,
                train_method = train_method,
                num_epochs = int(epochs),
                optimizer = str(optimizer),
                lr = float(lr),
                lr_gamma = float(lr_gamma),
                batch_size = int(batchsize),
                exchange_name = exchange_name
            )
        except Exception as e:
            self.update_state(state="ERROR")
            return

        self.update_state(state="FINALIZING")

        eval_model = f'models/{model_name}.onnx'
        stats_file = f"{model_name}_statistics.json"

        train_stats = session_storage["stats"] | finetune_stats

        if train_stats is None:
            train_stats = {}

        temp_dir = tempfile.mkdtemp()
        
        stats_file = os.path.join(temp_dir, stats_file)
        zip_filename = os.path.join(temp_dir, f"{model_name}.zip")

        with open(stats_file, 'w') as f:
            json.dump(train_stats, f)

        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            zipf.write(eval_model, os.path.basename(eval_model))
            zipf.write(stats_file, os.path.basename(stats_file))

        return zip_filename
