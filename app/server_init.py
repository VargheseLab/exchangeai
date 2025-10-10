import logging
import os

from flask import Flask
from flask_session import Session
from celery import Celery
from redis import Redis
import redis
import resource

from utils.model_exchange import ModelExchange
from utils.data import ECGData

# load environment
FINETUNE_DISABLED = os.environ.get('FINETUNE_DISABLED', "False").lower() in ('true', '1', 't')
SAMPLING_RATE = int(os.environ.get('SAMPLING_RATE', 100))
DISCLAIMER = os.environ.get('DISCLAIMER', "False").lower() in ('true', '1', 't')
LOCKDOWN_DEMO = os.environ.get('LOCKDOWN_DEMO', "False").lower() in ('true', '1', 't')

soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
limit = min(10000, hard)
try:
    resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))
except ValueError:
    logging.warning("Tried to set resource limit but failed, may cause issues with opening many files!")

def create_app():
    app = Flask(__name__)
    app.secret_key = os.environ.get('FLASK_SESSION_KEY', "DEBUG_KEY")
    if app.secret_key == "DEBUG_KEY":
        pass
        #logging.warning("Not secure!")

    
    app.config['SESSION_TYPE'] = 'redis'
    app.config['SESSION_REDIS'] = Redis.from_url('redis://127.0.0.1:6379')
    app.config.update(
        CELERY_BROKER_URL='redis://localhost:6379/0',
        RESULT_BACKEND='redis://localhost:6379/0'
    )

    # Session options
    app.config['SESSION_COOKIE_SAMESITE'] = None
    app.config['SESSION_COOKIE_SECURE'] = True  # Set to True if using HTTPS
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_DOMAIN'] = None #'127.0.0.1'
    app.config['SESSION_COOKIE_NAME'] = 'session'
    #app.config['SESSION_USER_SIGNER'] = True

    #configuration for upload
    app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024
    app.config['MAX_FORM_PARTS'] = 10000



    app.config.from_object(__name__)
    try:
        Session(app)
    except ConnectionRefusedError as e:
        logging.error("Redis not running?")
        raise Exception(e)

    return app

def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL'],
        broker_connection_retry_on_startup=True
    )
    celery.conf.update(app.config)
    TaskBase = celery.Task

    class ContextTask(TaskBase):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)

    celery.Task = ContextTask

    __all__ = ("celery_app",)

    return celery


app = create_app()
celery = make_celery(app)
webdav_client = ModelExchange()

redis_client = None
finetune_wrapper_task = None
#some startup
ecg_data = ECGData(
    target_sampling_rate = SAMPLING_RATE
)

@celery.task
def run_prediction_task(names, get_class=False, xai=False, prediction_model=None, session_key=None, exchange_name=None, **kwargs):
    if xai:
        data = ecg_data.get_by_name(names, ecg_type='Rlign-MedianBeats', xai=xai, session_key=session_key)
    else:
        data = ecg_data.get_by_name(names, ecg_type='Raw', session_key=session_key)

    if data is None: return {}
    
    predictions = ecg_data.onnx_runtimes.run_prediction(data, xai=xai, get_class=get_class, prediction_model=prediction_model, exchange_name=exchange_name,**kwargs)
    
    if predictions is None: return {}
    if isinstance(names, list):
        return dict(zip(names, predictions))
    else:
        return predictions

@celery.task
def upload_data_task(files, sampling_rate=500, adc_gain=1000, lead_layout='leads_default', session_key=None):
    ecg_data.upload(files, sampling_rate=sampling_rate, adc_gain=adc_gain, lead_layout=lead_layout, session_key=session_key)
    



if not FINETUNE_DISABLED:
    try:  
        # import Celery Task
        from utils_finetuning.finetuning_wrapper import FinetuneWrapper

        celery.register_task(FinetuneWrapper())
        finetune_wrapper_task = celery.tasks['finetune_wrapper_task']

        redis_client = redis.Redis(host='localhost', port=6379, db=0)

    except (ModuleNotFoundError) as e:
        logging.warning(f"Additional modules not found required for finetuning: {e}. \n -> Disabling finetuning.")
        FINETUNE_DISABLED = True


