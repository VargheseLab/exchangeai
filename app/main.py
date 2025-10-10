from gevent import monkey
import yaml
monkey.patch_all()

from server_init import app, ecg_data, celery, redis_client, webdav_client
from server_init import run_prediction_task, upload_data_task, finetune_wrapper_task
from server_init import FINETUNE_DISABLED, DISCLAIMER, LOCKDOWN_DEMO

from hashlib import md5
import json
import logging
import os

from flask import (Response, jsonify, make_response, redirect, render_template, request, send_file, session, stream_with_context, url_for)
import gevent
from markupsafe import Markup

from utils.plots import *
from utils.decorators import limit_content_length, with_referrer, with_session_key


@app.before_request
def before_request_func():
    # This has to set a value to store the session id
    session_id = md5(request.remote_addr.encode()).hexdigest()
    if "session_id" not in session.keys():
        session["session_id"] = session_id

# for sklearn SVM
def identity_function(x): return x

# register tasks
ecg_data.register_prediction_task(run_prediction_task)
ecg_data.register_upload_data_task(upload_data_task)

@with_session_key()
def index_renderer(content, session_key=None):
    if LOCKDOWN_DEMO and ecg_data.label_statistics(session_key=session_key) != []:
        ecg_data.upload_demo_data(session_key=session_key)
    return render_template('index.html', content=content, FINETUNE_DISABLED=FINETUNE_DISABLED, DISCLAIMER=DISCLAIMER, LOCKDOWN_DEMO=LOCKDOWN_DEMO)

@app.route("/")
def index():
    return redirect(url_for("landing"))

@app.route("/landing")
def landing():
    content = Markup(render_template('landing.html', DISCLAIMER=DISCLAIMER))
    return index_renderer(content)

@app.route("/analyse")
def analyse():
    content = Markup(render_template('analyse.html'))
    return index_renderer(content)

@app.route("/predict")
def predict():
    content = Markup(render_template('prediction.html'))
    return index_renderer(content)

@app.route("/explainable")
def explainable():
    content = Markup(render_template('explainable.html'))
    return index_renderer(content)

@app.route("/project")
def project():
    content = Markup(render_template('project.html'))
    return index_renderer(content)


#### Finetune
@app.route("/finetune")
def finetune():
    sess_task_id = session.get("task_id")

    if sess_task_id is not None:
       return redirect(url_for(f'finetune_with_task_id', task_id=sess_task_id))
    else:
        content = Markup(render_template('finetune.html'))
        return index_renderer(content)

@app.route("/finetune/<task_id>")
def finetune_with_task_id(task_id):
    sess_task_id = session.get("task_id")

    if sess_task_id is None or task_id != sess_task_id :
       return redirect(url_for('finetune'))

    content = Markup(
        render_template(
            'running_finetune.html',
            model_name=session["model_name"],
            show_logs=session["show_logs"]
        )
    )
    return index_renderer(content)


### GET functions ###
@app.route('/loaded_data_statistics', methods=["GET"])
@with_session_key()
def loaded_data_statistics(session_key=None):
    return ecg_data.statistics(session_key=session_key)

@app.route('/loaded_label_statistics', methods=["GET"])
def loaded_label_statistics(session_key=None):
    return plot_label_histogram(ecg_data.get_all_labels(session_key=session_key), ecg_data.names(session_key=session_key))

@app.route('/names_ecgs_loaded', methods=["GET"])
@with_session_key()
def names_ecgs_loaded(session_key=None):
    return jsonify(ecg_data.names(session_key=session_key))

@app.route('/availableModels', methods=["GET"])
@with_referrer()
def available_models(referrer=None):
    match referrer:
        case 'finetune':
            models = ecg_data.available_training_models()
        case 'explainable':
            models = ecg_data.available_xai_models()
        case _:
            models = ecg_data.available_prediction_models()
    return jsonify(models if models is not None else [])


@app.route('/save_webdav_settings', methods=["POST"])
def save_webdav_settings():
    data = request.json
    yaml_data = yaml.dump(data)
    try:
        with open('exchange_models/configuration.yml', 'w') as file:
            file.write(yaml_data)
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/list_files', methods=["GET"])
def list_webdav_files():
    training_models = os.listdir("./training_models/")
    training_models.extend(os.listdir("./models/"))

    prediction_models = os.listdir("./models/")

    if os.path.exists(f"./exchange_models/{webdav_client.get_exchange_name()}"):
        exchange_models = os.listdir(f"./exchange_models/{webdav_client.get_exchange_name()}")
    else: 
        exchange_models = []

    all_models = {
        "training_models": training_models,
        "prediction_models": prediction_models,
        "exchange_models": exchange_models
    }
    checked_models = json.loads(ecg_data.selected_model_list.get("selected_model_list"))
    return jsonify(all_models, checked_models)

        

### These are backend function for fetch ###
@app.route('/get_ecg', methods=["POST"])
@with_session_key()
def get_ecg(session_key=None):
    data = json.loads(request.data.decode('utf-8'))
    ecg = data.get('ecg')
    require_transform = data.get('transform')

    if data:
        ecg = ecg_data.get_by_name(ecg, require_transform, session_key=session_key)
        if ecg is not None: 
            match require_transform:
                case 'qrs': 
                    return plot_qrs(ecg)
                case 'events': 
                    return plot_events(ecg)
                case 'Rlign-MedianBeats': 
                    return plot_RlignMedianBeats(ecg)
                case _: 
                    return plot(ecg, ecg_data.target_sampling_rate)
                    
    return jsonify({})

@app.route('/prediction', methods=["POST"])
@app.route('/prediction/explainable', methods=["POST"])
@with_session_key()
def prediction_process(session_key=None):
    data = json.loads(request.data.decode('utf-8'))
    ecg = data.get('ecg')
    prediction_model = data.get('prediction')

    if not ecg or not prediction_model:
        return jsonify({})

    match request.path:
        case '/prediction/explainable':
            data = ecg_data.get_by_name(ecg, require_transform='Rlign-MedianBeats', session_key=session_key)
            xai = True
        case _:
            data = ecg_data.get_by_name(ecg, session_key=session_key)
            xai = False
    if data is not None:
        prediction = run_prediction_task.delay(
            ecg, get_class=False,
            prediction_model=prediction_model,
            xai=xai,
            session_key=session_key,
            exchange_name=webdav_client.get_exchange_name()
        )
        return jsonify(prediction.get()[0])
    return jsonify({})
    

@app.route('/clear_loaded_data', methods=["POST", "GET"])
@with_referrer()
@with_session_key()
def clear_loaded_data(referrer=None, session_key=None):
    ecg_data.clear_session_storage(session_key=session_key)
    return redirect(url_for(referrer))
    

### POST functions ###
@app.route('/prediction_statistics', methods=["POST"])
@with_referrer()
@with_session_key()
def prediction_statistics(referrer=None, session_key=None):
    data = json.loads(request.data.decode('utf-8'))
    prediction_model = data.get("prediction")
    as_plot = bool(data.get("as_plot"))
    view = data.get("view", "distribution")

    

    if as_plot:
        if view == 'roc':
            predictions = ecg_data.prediction_statistics(
                prediction_model,
                get_class=False,
                xai=(referrer == 'explainable'),
                session_key=session_key,
                exchange_name=webdav_client.get_exchange_name()
            )
            labels = ecg_data.get_all_labels(session_key=session_key)
            class_names = labels["label"].unique().to_list()

            predictions = predictions.transpose(include_header=True, header_name="ECG", column_names=None)
            predictions = DataFrame(
                [
                    {"ECG": file_name, **class_probs}
                    for file_name, class_probs in predictions.iter_rows()
                ]
            )

            return plot_prediction_roc(predictions, prediction_model, labels, class_names=class_names)
        else:
            predictions = ecg_data.prediction_statistics(
                prediction_model,
                get_class=True,
                xai=(referrer == 'explainable'),
                session_key=session_key,
                exchange_name=webdav_client.get_exchange_name()
            )
            return plot_prediction_histogram(predictions, prediction_model, ecg_data.get_all_labels(session_key=session_key))
    else:
        predictions = ecg_data.prediction_statistics(prediction_model, get_class=False, xai=(referrer == 'explainable'), session_key=session_key, exchange_name=webdav_client.get_exchange_name())
        predictions = predictions.transpose(include_header=True, header_name="ECG", column_names=None)
        predictions = DataFrame(
            [
                {"ECG": file_name, **class_probs}
                for file_name, class_probs in predictions.iter_rows()
            ]
        )
        
        predictions = jsonify({"export": predictions.write_csv()})
        response = make_response(predictions)
        response.headers['Content-Disposition'] = f'attachment;filename={prediction_model}.csv'
        return response



@app.route('/upload', methods=["POST"])
@limit_content_length()
@with_session_key()
def upload(session_key=None):
    try:
        files = request.files.getlist('ecgFileInput')
        sampling_rate = request.form.get('sampling_rate', None)
        if sampling_rate == 'custom':
            sampling_rate = request.form.get('custom_sampling_rate', 500)
        sampling_rate = int(sampling_rate)
        adc_gain = int(request.form.get('adc_gain', 1000))
        lead_layout = request.form.get('lead_layout', 'leads_default')

        upload_ft = []
        if files:
            def process_file(file, session_key=None):
                _, filetype = os.path.splitext(file.filename)
                return upload_data_task.delay((file.filename, filetype, file.read()), sampling_rate=sampling_rate, adc_gain=adc_gain, lead_layout=lead_layout, session_key=session_key)
            
            upload_ft = [process_file(file, session_key=session_key) for file in files]     
            for ft in upload_ft:
                ft.get()
    except Exception as e:
        logging.warning(e)

    return "uploaded chunck"

@app.route('/upload_labels', methods=["POST"])
@with_referrer()
def upload_labels(referrer=None):
    try:
        labels = request.files.getlist('labelsFileInput')
        if labels:
            ecg_data.upload_labels(labels)
    except Exception as e:
        logging.info(e)

    return redirect(url_for(referrer))

@app.route('/save_selected_model_list', methods=["POST"])
@with_referrer()
def save_selected_model_list(referrer=None):
    data = json.loads(request.data.decode('utf-8')).get("files")
    ecg_data.save_selected_model_list(data)
    return redirect(url_for(referrer))


@app.route('/upload_models', methods=["POST"])
@with_referrer()
def upload_models(referrer=None):

    try:
        models = request.files.getlist('predictionModelsInput')
        if models:
            ecg_data.onnx_runtimes.upload_models(models)
    except Exception as e:
        logging.info(e)

    return redirect(referrer)

@app.route('/export', methods=["POST"])
@with_session_key()
def export(session_key=None):

    ecg_type, export_type = list(request.form.values())
    export_data = None
    filename = "export"

    try:
        export_data = {key: data for (key, data) in ecg_data.get_all(ecg_type=ecg_type, session_key=session_key)}

        match export_type:
            case 'json':
                export_data = jsonify({key: data.write_json() for (key, data) in export_data.items()})
            case 'csv':
                export_data = jsonify({key: data.write_csv() for (key, data) in export_data.items()})
            case _:
                raise Exception(f'No such export format supported {export_type}')
        
        response = make_response(export_data)
        response.headers['Content-Disposition'] = f'attachment;filename={filename}.{export_type}'
        return response
        

    except Exception as e:
        logging.exception(e)

    return {}


@app.route('/task_status', methods=['GET'])
def task_status():
    task_id = None
    while True:
        task_id = session.get('task_id')
        if task_id is None:
            gevent.sleep(1)
        else: break

    return Response(
        stream_with_context(task_progress(task_id)),
        content_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        }
    )


@app.route('/task_status_stop')
def stop_stream():
    # Set a session variable to stop streaming
    session['stop_stream'] = True


def task_progress(task_id):
    pubsub = redis_client.pubsub()
    pubsub.subscribe(f"{task_id}:logs", f"{task_id}:progress")
    while True:
        try:
            if session.get('stop_stream'):
                del session['stop_stream']
                break

            task_result = celery.AsyncResult(task_id)
            response = None

            if task_result.state == 'PENDING':
                response = {
                    'state': task_result.state,
                    'status': 'Pending...'
                }

            elif task_result.state == 'RUNNING':
                logs = []
                progress = []

                while True:
                    message = pubsub.get_message()
                    
                    if message is None:
                        break
                    if message['type'] == 'subscribe':
                        continue
                    if message is not None:
                        channel = message['channel'].decode('utf-8').split(':')[-1]
                        msg = message['data'].decode('utf-8')
                        match channel:
                            case 'logs':
                                logs.append(msg)
                            case 'progress':
                                progress.append(msg)

                if not logs and not progress:
                    response = None
                else:
                    response = {
                        'logs': logs,
                        'progress': progress
                    }
                
            elif task_result.state == 'SUCCESS':
                model_name = session.get('model_name')
                response = {
                    'status': 'SUCCESS',
                    'result': url_for('get_finetuned_model',  task_id=task_id),
                    'model_name': model_name
                }

                yield f"data: {json.dumps(response)}\n\n"
                return redirect(url_for('finetune'))

            else:
                response = {
                    'state': task_result.state,
                    'status': str(task_result.info),  # this is the exception raised
                }
                yield f"data: {json.dumps(response)}\n\n"
                return

            
            if response is None:
                yield f"ping\n\n"
                gevent.sleep(5)
            else:
                yield f"data: {json.dumps(response)}\n\n"
                gevent.sleep(1)
            
            
        except (Exception, AttributeError, OSError) as e:
            logging.info(f"Error: {e}")
            gevent.sleep(5)
            yield f"ping\n\n"
       



#### FINETUNING
@app.route('/get_finetuned_model/<task_id>', methods=["GET"])
def get_finetuned_model(task_id):
    if FINETUNE_DISABLED:
        return jsonify({}) 
    
    cR = celery.AsyncResult(task_id)
    zip_filename = cR.get()
    if os.path.isfile(zip_filename):
        celery.control.revoke(task_id, terminate=True)
        celery.control.purge()
        cR.forget()
        session.pop('task_id')
        session.pop('model_name')
        return send_file(
            zip_filename,
            as_attachment=True,
            download_name=zip_filename,
            mimetype='application/zip'
        )
    return jsonify({})
 
@app.route('/run_finetune', methods=["POST", "GET"])
@with_session_key()
def run_finetune(session_key=None):
    if FINETUNE_DISABLED:
        return jsonify({})

    data = json.loads(request.data.decode('utf-8'))
    base_model = data.get('base_model')
    model_name = data.get('model_name')
    train_method = data.get('train_method')
    optimizer = data.get('optimizer')
    lr = data.get('lr')
    lr_gamma = data.get('lr_gamma')
    batchsize = data.get('batchsize')
    epochs = data.get('epochs')
    show_logs = data.get('show_logs')

    if base_model and model_name and train_method and session_key is not None:
         # set session variables
        session["model_name"] = model_name
        session["show_logs"] = show_logs

        task = finetune_wrapper_task.apply_async(
            args=[
                session_key,
                base_model,
                model_name,
                train_method,
                optimizer,
                epochs,
                lr,
                lr_gamma,
                batchsize,
                show_logs,
                ecg_data.target_sampling_rate,
                webdav_client.get_exchange_name()
            ]
        )
        session['task_id'] = task.id

    return redirect(url_for('finetune_with_task_id', task_id=task.id))
    

@app.route('/abort_finetune', methods=["POST", "GET"])
@with_session_key()
def abort_finetune(session_key=None):
    if FINETUNE_DISABLED:
        return jsonify({})
    
    task_id = session.get('task_id', None)
    if task_id is not None:
        del session['task_id']
    if task_id is not None:
        celery.control.revoke(task_id, terminate=True)
    return redirect(url_for('finetune'))




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('onnxruntime').setLevel(logging.ERROR)
    logging.getLogger('onnx2torch').setLevel(logging.NOTSET)
