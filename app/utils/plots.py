from typing import List

from polars import (DataFrame)
import plotly.express as px
import plotly.graph_objects as go
from flask import jsonify
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, f1_score


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
        data_frame=data,
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
        data_frame=data[:, data.columns[1:]],
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

        # Add text annotations
        #for i in range(cm.shape[0]):
        #    for j in range(cm.shape[1]):
        #        fig.add_annotation(
        #            text=np.round(cm[i, j], 2),
        #            x=j,
        #            y=i,
        #            showarrow=False,
        #            font=dict(color="black", size=15),
        #        )

        fig.update_layout(
            autosize = True,
            title=f"{prediction_model} - macro F1: {f1:.4} - weighted F1: {f1_weighted:.4}",
        )

    return fig.to_json()