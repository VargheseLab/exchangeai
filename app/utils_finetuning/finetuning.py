import copy
import importlib
import json
import logging
import math
import os
import sys
from threading import Event
from typing import List

import onnx
from onnxruntime import (InferenceSession, get_available_providers)
import onnxruntime as ort
from onnxruntime_extensions import get_library_path as _lib_path
from onnx2torch import convert

import redis
from torch.optim import AdamW, SGD, Adam, Adamax, NAdam, Adafactor, RMSprop
from torch.optim.lr_scheduler import ExponentialLR
from torch.cuda import device_count, is_available
from torch import (argmax, enable_grad, no_grad, tensor)
from torch import save as torch_save
from torch import load as torch_load
from torch.nn import CrossEntropyLoss, DataParallel
from torch.nn.init import zeros_, xavier_uniform_
from torch.nn.functional import softmax

from .utils import *
from .onnx_modifier import OnnxModifier
from .torch_modifier import TorchModifier
from .customConverters import *
from model_definitions import *

SAMPLING_RATE = int(os.environ.get('SAMPLING_RATE', 100))

class Finetune():
    def __init__(self, sampling_rate: int = 500) -> None:
        self.stats = {}
        self.sampling_rate = sampling_rate
        self._stop_event = None
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)

    def initialize_stats(self):
        self.stats = {'train_loss': {}}
        self.stats |= {'valid_loss': {}}
        self.stats |= {'valid_f1_macro': {}}
        self.stats |= {'valid_f1_weighted': {}}

    def stop(self):
        self._stop_event.set()

    # This logs to logging and sends it to the frontend
    def _log_message(self, message):
        logging.info(message)
        self.redis_client.publish(f"{self.task_id}:logs", json.dumps(message))

    def run_finetune(
        self,
        base_model: str,
        model_name: str,
        loaded_ecgs: dict[str, DataFrame],
        loaded_labels,
        label_statistics,
        task_id: str,
        train_method: str,
        exchange_name: str,
        num_epochs: int = 100,
        optimizer: str = "AdamW",
        batch_size: int = 8,
        lr: float = 1e-3,
        lr_gamma: float = 0.8,
        artifact_dir: str = 'training_artifacts',
        is_onnx: bool = True,
        min_dp_samples: int = 8,  # minimum number of samples per batch per GPU for DataParallel
    ) -> None:

        self.initialize_stats()
        self._stop_event = Event()
        self.lr = lr
        self.optimizer = optimizer
        self.task_id = task_id
        self.device = 'cuda' if is_available() else 'cpu'
        self._log_message(f"Training on {self.device}!")
        

        self.num_epochs = max(num_epochs,3)
        logging.debug('Transfering data...')
        self.data =  loaded_ecgs
        self.labels = loaded_labels

        if len(self.labels) == 0:
            logging.exception('No labels provided!')
            return
        
        if len(set(self.labels["file_name"]) - set(self.data.keys())) != 0 :
            logging.exception('No matching labels provided!')
            return
        
        
        # generate artifact directory
        self.artifact_dir = artifact_dir
        os.makedirs(self.artifact_dir, exist_ok=True)

        # get prediction model keys
        prediction_keys = sorted(list(set(label_statistics)))
        new_head_dim = len(prediction_keys)
        
        if base_model in [model.split('.')[0] for model in os.listdir("training_models/")]:
            model_path = "training_models/"
        elif base_model in [model.split('.')[0] for model in os.listdir("models/")]:
            model_path = "models/"
        elif base_model in [model.split('.')[0] for model in os.listdir(f"exchange_models/{exchange_name}/")]:
            model_path = f"exchange_models/{exchange_name}/"
        else:
            raise Exception("Model not found")

        is_onnx = any([_model_name.endswith('.onnx') for _model_name in os.listdir(model_path) if base_model in _model_name])

        self._log_message("Modify Model...")
        if is_onnx:
            modified_nodes = self.load_and_convert_onnx(model_path, base_model, new_head_dim)
        else:
            modified_nodes = self.load_and_modify_torch(model_path, base_model, new_head_dim)

        logging.info("Set parameters states...")
        try:
            named_params_with_grad = self.set_param_states(self.model, train_method, modified_nodes)
            self.stats |= {'method': train_method}
            self.stats |= {'finetuned_params': named_params_with_grad}
        except Exception as e:
            raise Exception(e)
        
        self._log_message("Generate Dataloaders...")
        # This shall load custom standardizer if necessary
        try: standardizer = self.model.standardizer
        except Exception: standardizer = None
        try: scale = self.model.scale
        except Exception: scale = "milliVolt"
        try: self.sampling_rate_overwrite = int(self.model.sampling_rate)
        except Exception: 
            if "untrained" in base_model or "DSAIL" in base_model: self.sampling_rate_overwrite = SAMPLING_RATE
            else: self.sampling_rate_overwrite = 100

        self.train_loader, self.eval_loader, self.class_weight = generate_loaders(
                                                                        self.data,
                                                                        self.labels,
                                                                        prediction_keys,
                                                                        device=self.device,
                                                                        sampling_rate=self.sampling_rate_overwrite,
                                                                        standardizer=standardizer,
                                                                        scale=scale
                                                                    )
        
        del self.labels, self.data
        

        self._log_message("Running lr-finder...")
        self.loss_fn = CrossEntropyLoss(weight=self.class_weight.to(self.device))

        model_copy = copy.deepcopy(self.model)
        loss_copy = copy.deepcopy(self.loss_fn)
        dataloader_copy = copy.deepcopy(self.train_loader)
        dataloader_copy.extend(self.eval_loader)

        self.set_param_states(model_copy, train_method, modified_nodes)
        self.suggested_lr = lr_finder(model_copy, dataloader_copy, loss_copy, device=self.device, final_lr=lr)
        if self.suggested_lr is not None:
            self.lr = self.suggested_lr
            self._log_message(f"Suggested lr: {self.suggested_lr}")
        else:
            self.lr = lr
            self._log_message(f"Lr-finder returned None - Fallback to maximum provided lr: {lr}")
        del model_copy, loss_copy, dataloader_copy

        # enable DataParallel if requirements are met
        if self.device=='cuda':
            cuda_device_count = device_count()
            if cuda_device_count > 1 and batch_size >= min_dp_samples*cuda_device_count:
                self._log_message(f"Running DataParallel on {cuda_device_count} CUDA devices.")
                self.model = DataParallel(self.model)

        self._log_message(f"Using {self.optimizer} optimizer.")
        match self.optimizer:
            case "AdamW":
                self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
            case "Adam":
                self.optimizer = Adam(self.model.parameters(), lr=self.lr)
            case "NAdam":
                self.optimizer = NAdam(self.model.parameters(), lr=self.lr)
            case "Adamax":
                self.optimizer = Adamax(self.model.parameters(), lr=self.lr)
            case "Adafactor":
                self.optimizer = Adafactor(self.model.parameters(), lr=self.lr)
            case "RMSprop":
                self.optimizer = RMSprop(self.model.parameters(), lr=self.lr)
            case "SGD":
                self.optimizer = SGD(self.model.parameters(), lr=self.lr)
            case _:
                self._log_message("Unknown Optimizer, using default AdamW.")
                self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = ExponentialLR(self.optimizer, gamma=lr_gamma)
        
        self._log_message("Start finetune")
        best_loss = math.inf
        best_f1_weighted = 0.0
        best_epoch = 0
        overfit_eps = 1.25
        best_loss_delta = 1 - 0.01 # at least 1% better
        best_f1_delta = 1 + 0.02 # at least in range of 2% F1-score
        epoch_eps = 10
        min_delta_epoch = 10

        import time
        
        for epoch in range(self.num_epochs):
            start_time = time.perf_counter() 
            # interrupt thread
            if self._stop_event.is_set(): 
                logging.warning("Interrupting Thread!")
                self.cleanup()
                return "INTERRUPTED"

            self.train(epoch, batch_size)
            eval_loss, eval_f1_weighted = self.eval(epoch, batch_size)
            self.scheduler.step()
            logging.debug(f"Learning Rate: {self.scheduler.get_last_lr()}")


            if (eval_loss < best_loss * best_loss_delta) and (eval_f1_weighted > best_f1_weighted * best_f1_delta):
                best_loss = eval_loss
                best_f1_weighted = eval_f1_weighted
                best_epoch = epoch

                try: # DataParallel
                    state_dict = self.model.module.state_dict()
                except AttributeError:
                    state_dict = self.model.state_dict()

                torch_save(state_dict, f'{self.artifact_dir}/{model_name}_chkpt.pth')
            
            elif (eval_loss > (overfit_eps * best_loss) and epoch > min_delta_epoch) or ( (best_epoch + epoch_eps) < epoch ):
                logging.info("Early stopping!")
                break

            #self.queue.appendleft(epoch / self.num_epochs)
            self.redis_client.publish(f"{self.task_id}:progress", f"{(epoch + 1) / self.num_epochs}")
            end_time = time.perf_counter()
            self.stats[f'train_time_epoch_{epoch + 1}'] = end_time - start_time

        if os.path.exists(f'{self.artifact_dir}/{model_name}_chkpt.pth'):
            #shutil.copy(f'{self.artifact_dir}/{model_name}_chkpt.pth', f'{self.artifact_dir}/{model_name}.pth')
            state_dict = torch_load(f'{self.artifact_dir}/{model_name}_chkpt.pth')
            try: # DataParallel
                model = self.model.module
            except AttributeError:
                model = self.model
            model.load_state_dict(state_dict)
            export2onnx(model, model_name, prediction_keys, self.sampling_rate_overwrite, standardizer)
        else:
            logging.warning("No finetuned model generated, please retry!")

        self.redis_client.publish(f"{self.task_id}:progress", f"{1.0}")
        self.redis_client.close()
        return self.stats


    def set_param_states(
        self,
        model,
        train_method: str,
        modified_nodes: List[str]
    ) -> None:
        match train_method:
            case "Finetuning (classification head)":
                # Only unfreeze alls heads and modified nodes
                for name, param in model.named_parameters():
                    if ('head' not in name.lower()) and not any([mn in name for mn in modified_nodes]):
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

            case "Finetuning (all layers)":
                # unfreeze complete model
                for param in model.parameters():
                    param.requires_grad = True

            case "Scratch":
                # unfreeze complete model and initilize with new weights
                for name, param in model.named_parameters():
                    param.requires_grad = True
                    if 'batch' in name.lower():
                        continue
                    elif param.ndimension() >= 2:
                        xavier_uniform_(param.data)
                    else:  # Bias is typically a 1D tensor
                        zeros_(param.data)
            case _:
                raise Exception(f"No such trainings method supported: {train_method}")
        named_params_with_grad = [name for name, param in model.named_parameters() if param.requires_grad]
        logging.debug(named_params_with_grad)
        return named_params_with_grad
        
        
    def train(
        self,
        epoch: int,
        batch_size: int
    ) -> None:
        '''
        Training loop
        '''
        self.model.train()
        losses = []

        for (data, target) in batch(self.train_loader, batch_size, training=True):
            with enable_grad():
                self.optimizer.zero_grad()
                logits = self.model(data.to(self.device))
                train_loss = self.loss_fn(logits, target.to(self.device))
                train_loss.backward()

                self.optimizer.step()
                losses.append(train_loss.item())

        if losses: 
            train_loss = sum(losses)/len(losses)
            self._log_message(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}')
            self.stats['train_loss'] |= {f'epoch_{epoch+1:03}': train_loss}
        else: logging.warning("No loss generated by finetuning!")


    def eval(
        self,
        epoch: int,
        batch_size: int
    ) -> float:
        '''
        Evaluation loop
        '''
        losses = []
        valid_loss = 9999.9
        f1_score = F1ScoreEvaluator()

        self.model.eval()
        for (data, target) in batch(self.eval_loader, batch_size, training=False, noise=False):
            with no_grad():
                logits = self.model(data.to(self.device))

                test_loss = self.loss_fn(logits, target.to(self.device))

                f1_score.add_batch(references=target.cpu(), predictions=self.get_pred(logits).cpu())
                losses.append(test_loss.item())

        if losses: 
            valid_loss = np.nansum(losses)/len(losses)
            self._log_message(f'Epoch: {epoch+1}, Eval Loss: {valid_loss:.4f}') 
            self.stats['valid_loss'] |= {f'epoch_{epoch+1:03}': valid_loss}

            f1_weighted = f1_score.compute_weighted_f1()
            f1_macro = f1_score.compute_f1()
            self._log_message(f'Epoch: {epoch+1}, Eval-F1 : {f1_weighted:.4f} (weighted) {f1_macro:.4f} (macro)')
            self.stats['valid_f1_macro'] |= {f'epoch_{epoch+1:03}': f1_macro}
            self.stats['valid_f1_weighted'] |= {f'epoch_{epoch+1:03}': f1_weighted}
            return valid_loss, f1_weighted
        else: 
            logging.warning("No loss generated by finetuning!")
            return valid_loss, None

    def load_and_convert_onnx(self, model_path, base_model, new_head_dim):
        # add extra operators
        so = ort.SessionOptions()
        so.register_custom_ops_library(_lib_path())

        prediction_session = InferenceSession(f'{model_path}/{base_model}.onnx', providers=get_available_providers(), sess_options=so)
        prediction_model_keys = prediction_session.get_modelmeta().custom_metadata_map.get('target_keys', None)

        if prediction_model_keys is not None:
            prediction_model_keys = eval(prediction_model_keys)
            head_dim = len(prediction_model_keys)
        else:
            head_dim = prediction_session.get_outputs()[0].shape[1]

        logging.info(f'Old head dimension: {head_dim}')
        logging.info(f'New head dimension: {new_head_dim}')
        logging.debug(f'Current prediction keys: {prediction_model_keys}')
        
        onnx_model = onnx.load(f"{model_path}/{base_model}.onnx")
        onnx_model.opset_import[0].version = 17

        self._log_message(f'Modifying Model to new classification task...')
        onnx_model, modified_nodes = OnnxModifier(
            head_dim,
            new_head_dim
        ).modify_onnx(onnx_model)
        logging.info(f"Modified nodes: {modified_nodes}")

        modified_model_path = f'{self.artifact_dir}/{base_model}_modified.onnx'
        if os.path.isfile(modified_model_path):
            os.remove(modified_model_path)
        onnx.save(onnx_model, modified_model_path)

        try:
            onnx.checker.check_model(onnx_model, full_check=True)
        except onnx.onnx_cpp2py_export.shape_inference.InferenceError as e:
            self._log_message(f'Failed at modifying model, please check compability!')
            raise Exception(e)
        
        logging.info("Convert to torch...")
        try:
            self.model = convert(onnx_model).to(self.device)
        except Exception as e:
            logging.info(e)

        return modified_nodes
    
    def load_and_modify_torch(self, model_path, base_model, new_head_dim):
        files = {
            f.split(".")[0]: f.split(".")[1]
            for f in os.listdir("training_models/")
            if os.path.isfile(f"training_models/{f}")
        }

        file_suffix = files.get(base_model, None)
        
        match file_suffix:
            case 'pt':
                torch_model = torch_load(f'{model_path}/{base_model}.pt', map_location=self.device)
            case 'pth':
                torch_model = torch_load(f'{model_path}/{base_model}.pth', map_location=self.device)
                if isinstance(torch_model, dict):
                    torch_model = torch_model['state_dict']

                torch_save(torch_model, f'training_artifacts/{base_model}.pt')
                torch_model = torch_load(f'training_artifacts/{base_model}.pt', map_location=self.device)
            case _:
                raise Exception(f"Model mismatch {base_model}")

        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        sys.path.insert(0, parent_dir)
   
        # List all Python files in the given directory
        for filename in os.listdir("model_definitions/"):
            if filename.endswith('.py') and filename != '__init__.py':
                module_name = filename.removesuffix(".py")  # Strip the .py extension

                if module_name == base_model:
                    try:
                        module = importlib.import_module(f"model_definitions.{module_name}")
                        cls = getattr(module, module_name)
                        state_dict = torch_model
                    except Exception as e:
                        raise Exception(e)
                    
                    torch_model = cls()
                    try:
                        torch_model.load_state_dict(state_dict, strict=False)
                        torch_model.to(self.device)
                    except Exception as e:
                        raise Exception(e)
                    
        head_dim = torch_model(torch_randn((1,12,1000), device=self.device)).shape[1]
        torch_model, modified_nodes = TorchModifier(
            head_dim,
            new_head_dim
        ).modify_torch(torch_model)
                        
        logging.info(f"Modified nodes: {modified_nodes}")
        self.model = torch_model.to(self.device)
        return modified_nodes


    # Util function to convert logits to predictions.
    @staticmethod
    def get_pred(logits: tensor) -> tensor:
        return argmax(logits, axis=1)
    
    # Util function to convert logits to predictions.
    @staticmethod
    def softmax(logits: tensor) -> tensor:
        return softmax(logits, dim=1)
    




