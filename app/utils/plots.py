from typing import List

from polars import (DataFrame)
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from flask import jsonify
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc, precision_recall_curve
import numpy as np

header = ["I", "II", "III", "aVF", "aVR", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]

def plot(
    data: DataFrame,
    sampling_rate: int = 500
) -> str:
    
    """Generates several plots for each lead individually of the input ecg.

    Args:
        data (DataFrame): The input ECG to use.
        sampling_rate (int, optional): The used sampling rate of the ECG. Defaults to 500.

    Returns:
        Any: The plotly-json of the ECG.
    """
    fig = make_subplots(
        rows=6, cols=2,
        shared_xaxes=True,
        vertical_spacing=0.01,
        horizontal_spacing=0.025,
        row_heights=[1/2]*6,
    )
    fig.update_layout(plot_bgcolor="#FFDADA")

    index = data.with_row_index()['index']

    for idx, column in enumerate(data.columns):
        trace = go.Scatter(
            x=index / sampling_rate,
            y=data[column],
            mode='lines',
            name='',
            line=dict(color='#333333'),
            showlegend=False,
        )
        col = (idx // 6) + 1
        row = (idx % 6) + 1
        fig.add_trace(trace, row=row, col=col)
        fig.add_annotation(
            text=f'{header[idx]} (mV)',
            showarrow=False,
            font=dict(size=10),
            row=row, col=col,
            y=1.15,
        )

    for i in range(2, 13):
        if i % 2 == 0:
            fig.layout[f'xaxis{i}'].matches = 'x'
        fig.layout[f'yaxis{i}'].matches = f'y{i-1}'

    fig.update_layout(
        hovermode = "x unified",
        legend_traceorder="normal",
        margin={
            'b': 0,
            't': 0,
            'pad': 0,
            'r': 0,
            'l': 0
        }
    )

    return fig.to_json()


def plot_RlignMedianBeats(data: DataFrame) -> str:
    """Generates a plot based on the computed median beats by Rlign.

    Args:
        data (DataFrame): The input median beats ECG.

    Returns:
        Any: The plotly-json of the median beats.
    """
    data = data.transpose()
    data.columns = header
    fig = px.line(
        data_frame=data.to_pandas(),
    )
    fig.update_layout(
        plot_bgcolor="#FFDADA"
    )
    return fig.to_json()


def plot_qrs(data: DataFrame) -> str:
    """Generates a plot based on the computed QRS-complexes.

    Args:
        data (DataFrame): The input QRS-complexes of an ECG.

    Returns:
        Any: The plotly-json of the QRS-complexes.
    """
    fig = px.line(
        x=data['Time'],
        y=data['Signal'],
        color=data['Label']
    )
    fig.update_layout(plot_bgcolor="#FFDADA")
    return fig.to_json()

def plot_events(data: DataFrame) -> str:
    """Generates a plot based on the computed QRS-complexes.

    Args:
        data (DataFrame): The input QRS-complexes of an ECG.

    Returns:
        Any: The plotly-json of the QRS-complexes.
    """

    fig = px.scatter(
        data_frame=data[:, data.columns[1:]].to_pandas(),
        labels = {'x': '', 'y': ''}
    )
    index = data.with_row_index()['index']
    fig.add_trace(go.Scatter(
        x=index,
        y=data['ECG_Clean'],
        mode='lines',
        name='ECG_Clean'
    ))
    fig.update_layout(plot_bgcolor="#FFDADA")
    return fig.to_json()

def plot_label_histogram(
    labels: DataFrame,
    names: List[str]
) -> str:
    
    if len(labels) == 0:
        return jsonify({'data': []})
    
    # aggregate and show loaded labels
    labels_count = labels['label'].value_counts()
    fig = px.bar(
        x=labels_count['label'],
        y=labels_count['count'],
        labels = {'x': 'label', 'y': 'count'}
    )

    if len(names) > 0:
        labels = labels.cast({"file_name": str})
        # check overlapping loaded labels and loaded ecgs
        df = labels.join(
            other=DataFrame({'file_name': names}),
            on='file_name',
            how='inner',
        ).drop(['file_name', 'ICD-10'])

        # aggregate and show loaded ecgs
        ecgs_count = df['label'].value_counts()
        fig.add_trace(go.Bar(
            x=ecgs_count['label'],
            y=ecgs_count['count'],
            name='Loaded ECGs',
            opacity=0.75
        ))
        fig.update_layout(barmode = "overlay")

    fig.update_layout(
        autosize = True,
        yaxis_title_text='Count'
    )
    return fig.to_json()


def plot_prediction_histogram(
    predictions: DataFrame,
    prediction_model: str,
    labels: DataFrame = None
) -> str:
    
    if len(predictions) == 0:
        return jsonify({'data': []})
    
    if labels.is_empty():
        # aggregate and show loaded labels
        pred_count = predictions.melt()['value'].value_counts()
        fig = px.bar(
            pred_count,
            x='value',
            y='count',
        )

        fig.update_layout(
            autosize = True,
            title=prediction_model,
            yaxis_title_text='Count',
            xaxis_title_text='Prediction'
        )
    else:
        predictions = predictions.melt().rename({'variable': 'file_name', 'value': 'prediction'})
        merged_df = predictions.join(labels, on='file_name')
    
        actual = merged_df['label']
        predicted = merged_df['prediction']

        labels = sorted(list(set(actual) | set(predicted)))  # Sort and unify labels for consistency
        cm = confusion_matrix(actual, predicted, labels=labels, normalize='true')
        f1 = f1_score(actual, predicted, average="macro")
        f1_weighted = f1_score(actual, predicted, average="weighted")
        fig = px.imshow(
            cm,
            labels=dict(x='Predicted Label', y='Actual Label', color='%'),
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale='Blues'
        )

        fig.update_layout(
            autosize = True,
            title=f"{prediction_model} - macro F1: {f1:.4} - weighted F1: {f1_weighted:.4}",
        )

    return fig.to_json()


def plot_prediction_roc(
    predictions: DataFrame,
    prediction_model: str,
    labels: DataFrame = None,
    class_names: List[str] = None
) -> str:
    try:
        pdf = predictions.to_pandas()
    except Exception:
        pdf = pd.DataFrame(predictions)

    # Detect filename column if present; prefer 'ECG'/'file_name'/'index'
    id_col = None
    for candidate in ("ECG", "file_name", "index"):
        if candidate in pdf.columns:
            id_col = candidate
            break
    if id_col is not None:
        pdf = pdf.set_index(id_col)
        
    class_cols = [c for c in class_names if c in pdf.columns] or class_cols
    # coerce class columns to numeric probabilities
    for c in class_cols:
        pdf[c] = pd.to_numeric(pdf[c].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0)

    # handle labels -> convert polars -> pandas if necessary
    label_map = {}
    if labels is not None and len(labels) > 0:
        try:
            lpdf = labels.to_pandas()
        except Exception:
            lpdf = pd.DataFrame(labels)

        # detect file id and label column names
        label_id = None
        for candidate in ("file_name", "ECG", "index"):
            if candidate in lpdf.columns:
                label_id = candidate
                break
        label_col = "label" if "label" in lpdf.columns else next((c for c in lpdf.columns if c != label_id), None)
        if label_id is not None:
            lpdf = lpdf.set_index(label_id)
        if label_col is not None:
            label_map = {str(k): v for k, v in lpdf[label_col].to_dict().items()}

    # If no labels or no overlap -> return fig signalling no labels
    if not label_map:
        return {}

    # Align predictions and labels (keep only common filenames)
    pred_files = list(pdf.index.astype(str)) if pdf.index is not None else [str(i) for i in range(len(pdf))]
    common = sorted(list(set(pred_files) & set(label_map.keys())))
    if len(common) == 0:
        return {}

    pdf = pdf.loc[common]
    file_names = common

    # ensure class ordering - use union of class cols and label classes
    label_classes = sorted(list(set(label_map.values())))
    if not class_cols:
        class_cols = label_classes
    else:
        # include any label-only classes
        class_cols = sorted(list(dict.fromkeys(list(class_cols) + label_classes)))

    probs = pdf[class_cols].to_numpy(dtype=float)

    # build binary ground-truth matrix (n_samples x n_classes)
    n = len(file_names)
    k = len(class_cols)
    y_trues = np.zeros((n, k), dtype=int)
    for i, fname in enumerate(file_names):
        true = label_map.get(fname, None)
        if true is None:
            true = label_map.get(fname.split("/")[-1], None)
        if true is None:
            continue
        try:
            idx = class_cols.index(true)
            y_trues[i, idx] = 1
        except ValueError:
            continue

    # compute per-class ROC, AUC, Fmax and suggestions
    all_roc = {}
    thresholds = {}
    suggestions = {}
    sens_target = 0.90
    spec_target = 0.90

    for idx, cls in enumerate(class_cols):
        y_true = y_trues[:, idx]
        y_score = probs[:, idx].astype(float)

        # skip classes without both pos and neg examples
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            all_roc[cls] = {"fpr": [], "tpr": [], "thresholds": [], "auc": 0.0}
            thresholds[cls] = {"threshold": 0.5, "fmax": 0.0}
            suggestions[cls] = {
                "sensitivity_target": sens_target,
                "suggested_threshold_for_sensitivity": None,
                "specificity_target": spec_target,
                "suggested_threshold_for_specificity": None
            }
            continue

        # ROC & AUC
        try:
            fpr, tpr, roc_th = roc_curve(y_true, y_score)
            roc_auc = float(auc(fpr, tpr))
            all_roc[cls] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_th.tolist(), "auc": roc_auc}
        except Exception:
            all_roc[cls] = {"fpr": [], "tpr": [], "thresholds": [], "auc": 0.0}

        # Fmax (scan thresholds from precision_recall_curve)
        try:
            pres, recs, pr_thresh = precision_recall_curve(y_true, y_score)
            best_thr = 0.5
            best_f1 = 0.0
            # candidate thresholds from pr_thresh (len = len(pres)-1)
            if len(pr_thresh) > 0:
                for thr in pr_thresh:
                    preds_bin = (y_score >= thr).astype(int)
                    f1 = f1_score(y_true, preds_bin, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = float(f1)
                        best_thr = float(thr)
            else:
                preds_bin = (y_score >= 0.5).astype(int)
                best_f1 = float(f1_score(y_true, preds_bin, zero_division=0))
                best_thr = 0.5
            thresholds[cls] = {"threshold": best_thr, "fmax": best_f1}
        except Exception:
            thresholds[cls] = {"threshold": 0.5, "fmax": 0.0}

        # suggestions: threshold for sensitivity / specificity targets
        roc = all_roc.get(cls, {"fpr": [], "tpr": [], "thresholds": []})
        fpr_arr = np.array(roc["fpr"]) if len(roc["fpr"]) > 0 else np.array([])
        tpr_arr = np.array(roc["tpr"]) if len(roc["tpr"]) > 0 else np.array([])
        thr_arr = np.array(roc["thresholds"]) if len(roc["thresholds"]) > 0 else np.array([])

        def pick_threshold(target_array, target_val):
            if target_array.size == 0 or thr_arr.size == 0:
                return None
            idxs = np.where(target_array >= target_val)[0]
            if idxs.size > 0:
                # choose index with highest specificity among candidates (if possible)
                spec = 1 - fpr_arr if fpr_arr.size > 0 else None
                if spec is not None and spec.size > 0:
                    pick = idxs[np.argmax(spec[idxs])]
                else:
                    pick = idxs[0]
                return float(thr_arr[pick])
            # fallback to nearest
            pick = int(np.argmin(np.abs(target_array - target_val)))
            return float(thr_arr[pick])

        thr_sens = pick_threshold(tpr_arr, sens_target)
        thr_spec = pick_threshold(1 - fpr_arr if fpr_arr.size > 0 else np.array([]), spec_target)

        suggestions[cls] = {
            "sensitivity_target": sens_target,
            "suggested_threshold_for_sensitivity": thr_sens,
            "specificity_target": spec_target,
            "suggested_threshold_for_specificity": thr_spec
        }

    # Build Plotly figure: one ROC line per class + marker for Fmax
    fig = go.Figure()
    for cls in class_cols:
        roc = all_roc.get(cls, {"fpr": [], "tpr": [], "thresholds": [], "auc": 0.0})
        if len(roc["fpr"]) > 0:
            fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode='lines', name=f"{cls} (AUC={roc['auc']:.2f})"))
            thr = thresholds.get(cls, {}).get("threshold", None)
            if thr is not None and len(roc.get("thresholds", [])) > 0:
                # find best matching index (nearest)
                try:
                    idx_pick = min(range(len(roc["thresholds"])), key=lambda i: abs(roc["thresholds"][i] - thr))
                    fig.add_trace(go.Scatter(
                        x=[roc["fpr"][idx_pick]],
                        y=[roc["tpr"][idx_pick]],
                        mode='markers',
                        marker=dict(size=8, symbol='diamond', color='red'),
                        name=f"{cls} Fmax={thresholds[cls]['fmax']:.2f} @ {thr:.2f}",
                        showlegend=True
                    ))
                except Exception:
                    pass
        else:
            # empty trace to keep legend consistent
            fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name=f"{cls} (no ROC)"))

    fig.update_layout(
        title=f"{prediction_model} - class-wise ROC",
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Sensitivity)",
        #legend=dict(
        #    orientation="h",
        #    y=0.0
        #),
        plot_bgcolor="#FFDADA",
        autosize=True,
        meta={
            "has_labels": True,
            "roc": all_roc,
            "thresholds": thresholds,
            "suggestions": suggestions
        }
    )

    return fig.to_json()