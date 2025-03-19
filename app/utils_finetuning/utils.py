import logging
from typing import List
import numpy as np
from pandas import DataFrame
import polars as pl
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import f1_score
from torch import tensor, ones_like, float32
from torch import randn as torch_randn
from torch import onnx as torch_onnx
from torch import normal as torch_normal
from torch.optim import AdamW
import onnx

from tqdm import tqdm

from utils.util_functions import ecg_normalize

def create_single_loader(file_names, data, labels, label_converter, sampling_rate, standardizer, scale):
    # Reduce labels dataframe to only the necessary labels
    label_map = labels.filter(pl.col("file_name").is_in(file_names))
    
    # Convert to dictionary for fast lookup
    try:
        label_map = label_map.with_columns((
            pl.col("label").map_elements(lambda x: label_converter.get(x, None), return_dtype=pl.Int32).alias("label")
        ))
    except Exception as e:
        logging.info(e)

    label_map = dict(zip(label_map["file_name"], label_map["label"]))

    def process_file(data, label, standardizer = None, scale = "milliVolt"):
        normalized_data = ecg_normalize(
            data.to_numpy(),
            sampling_rate = sampling_rate,
            trend_removal = "movingMedian",
            standardizer = standardizer,
            scale = scale
        )
        label = np.asarray(label, dtype=np.int64)
        return (normalized_data, label)

    loader = []
    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, data.get(file_name), label_map.get(file_name), standardizer, scale) for file_name in file_names}
        for future in tqdm(as_completed(future_to_file), total=len(future_to_file)):
            try:
                data = future.result()
                loader.append(data)
            except Exception as exc:
                logging.warning(f'Error while processing ECGs in finetune: {exc}')

    return loader

def generate_loaders(
        data,
        labels: DataFrame,
        prediction_model_keys,
        frac: float = 0.8,
        device: str = 'cuda',
        smoothing_factor_bin: float = 0.00, # set for binary distribution, reduces overconfidentiality
        sampling_rate: int = 100,
        standardizer = None,
        scale = "milliVolt"
    ):
    '''
    Generate dataloaders
    '''
    labels_ = labels.to_pandas()
    train_data = labels_.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=frac))
    class_weight = train_data['label'].value_counts().to_dict()
    class_weight = np.asarray([class_weight.get(key) for key in prediction_model_keys], dtype=np.float32) 
    eval_data = set(labels['file_name']) - set(train_data['file_name'])


    keys_to_index = {key: key_idx for key_idx, key in enumerate(prediction_model_keys)}
    
    train_loader = create_single_loader(train_data['file_name'], data, labels, keys_to_index, sampling_rate, standardizer, scale)
    eval_loader = create_single_loader(eval_data, data, labels, keys_to_index, sampling_rate, standardizer, scale)

    class_weight = tensor(1 / (class_weight / np.min(class_weight)), device=device)
    ones_tensor = ones_like(class_weight, device=device, dtype=float32)
    smoothing_factor = smoothing_factor_bin * len(class_weight)
    class_weight = (1 - smoothing_factor) * class_weight + smoothing_factor * ones_tensor

    logging.info(f"Class Weights: {class_weight} for {prediction_model_keys}")

    return train_loader, eval_loader, class_weight

def export2onnx(
    pytorch_model,
    model_name: str,
    prediction_model_keys,
    sampling_rate: int,
    standardizer = None
) -> None:

    dummy_input = torch_randn((5, 12, 10 * sampling_rate), requires_grad=True, device='cpu')
    
    training_mode = torch_onnx.TrainingMode.EVAL

    pytorch_model.eval()
    pytorch_model = pytorch_model.to("cpu")
    try:
        torch_onnx.export(
            model = pytorch_model, 
            args = dummy_input,
            f = f"models/{model_name}.onnx",
            export_modules_as_functions = False,
            operator_export_type = torch_onnx.OperatorExportTypes.ONNX,
            dynamic_axes = {
                "arg0": {0: "s0"},
                "self_1_1": {0: "s0"}
            },
            input_names = ["arg0"],
            output_names = ["self_1_1"],
            opset_version = 17,
            export_params = True,
            training = training_mode,
            do_constant_folding = False,
            keep_initializers_as_inputs = False,
        )
    except Exception as e:
        raise Exception(e)

    # load model and add to metadata header
    onnx_model = onnx.load(f"models/{model_name}.onnx")
    if len(onnx_model.metadata_props) == 1:
        onnx_model.metadata_props.pop(0)
    meta = onnx_model.metadata_props.add()
    meta.key = "target_keys"
    meta.value = str(prediction_model_keys)

    meta2 = onnx_model.metadata_props.add()
    meta2.key = "standardizer"
    meta2.value = str(standardizer)
    onnx.save(onnx_model, f'models/{model_name}.onnx')



def add_gaussian_noise(ecg_data, mean=0, std=0.01):
    noise = torch_normal(mean, std, ecg_data.shape, device=ecg_data.device)
    return ecg_data + noise


class F1ScoreEvaluator:
    def __init__(self):
        self.true_labels = []
        self.pred_labels = []

    def add_batch(self, references: List[int], predictions: List[int]) -> None:
        """Add a batch of true labels and predictions."""
        self.true_labels.extend(references)
        self.pred_labels.extend(predictions)

    def compute_f1(self) -> float:
        """Compute the F1 score."""
        return f1_score(self.true_labels, self.pred_labels, average='macro')

    def compute_weighted_f1(self) -> float:
        """Compute the weighted F1 score."""
        return f1_score(self.true_labels, self.pred_labels, average='weighted')
    


def lr_finder(model, data_loader, criterion, init_lr=1e-4, final_lr=1e-2, device:str = 'cuda', step_mode='linear', smooth_f=0.05):
    assert init_lr <= final_lr

    num_samples = len(data_loader)
    min_steps = 200
    batch_size = max((num_samples // min_steps), 4)
    num_iter = num_samples // batch_size
    model = model.to(device)
    optimizer = AdamW(model.parameters())

    # set start lr
    def set_lr(lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    set_lr(init_lr)
    
    # set scheduler
    if step_mode.lower() == "exp":
        scheduler = lambda curr_iter: init_lr * (final_lr / init_lr) ** (curr_iter / num_iter)
    elif step_mode.lower() == "linear":
        scheduler = lambda curr_iter: (curr_iter / num_iter) * (final_lr - init_lr)

    history = {"lr": [], "loss": []}
    best_loss = None
    curr_iter = 0
    curr_lr = init_lr
    diverge_th = 5

    for (data, target) in batch(data_loader, batch_size=batch_size, noise=False):
        optimizer.zero_grad()
        try:
            outputs = model(data.to(device))
        except Exception as e:
            logging.info(e)
        loss = criterion(outputs, target.to(device))
        loss.backward()

        history["lr"].append(curr_lr)
        loss = loss.item()

        if curr_iter > 0:
            loss = smooth_f * loss + (1 - smooth_f) * history["loss"][-1]

            # Record the best loss and corresponding LR
            if loss < best_loss:
                best_loss = loss
        else:
            best_loss = loss

        history["loss"].append(loss)

        # Stop if the loss is exploding
        if loss > diverge_th * best_loss:
            break

        # Do a step in the optimizer
        optimizer.step()
        curr_iter += 1
        curr_lr = scheduler(curr_iter)
        set_lr(curr_lr)

    
    lrs = np.asarray(history["lr"])[1:-1]
    losses = np.asarray(history["loss"])[1:-1]
    min_grad_idx = None
    best_lr = None

    try:
        min_grad_idx = np.gradient(losses).argmin()
        best_lr = lrs[min_grad_idx]
    except ValueError:
        logging.warning("Failed to compute the gradients, there might not be enough points.")

    return best_lr

def batch(
    loader: List[tuple[np.ndarray, str]],
    batch_size: int = 4,
    training = True,
    noise = True
) -> tensor:
    indices = np.arange(len(loader), dtype=np.int64) 

    X, Y = zip(*loader)
    
    # shuffle data before each epoch
    np.random.shuffle(indices)
    X = np.asarray(X)
    Y = np.asarray(Y, dtype=np.int64)
    for idx in range(0, len(indices), batch_size):
        idx_end = min(idx+batch_size, len(indices))

        ts = tensor(X[indices[idx : idx_end]], requires_grad=training)
        if noise:
            pass
            #ts = add_gaussian_noise(ts)
        yield ts, tensor(Y[indices[idx : idx_end]])