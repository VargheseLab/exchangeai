from concurrent.futures import ThreadPoolExecutor
import logging
import os
from utils.decorators import time_cache
from webdav4.client import Client
import yaml


class ModelExchange():
    def __init__(
        self,
        yaml_file: str = "./exchange_models/configuration.yml"
    ):
        self.yaml_file = yaml_file
        with open(yaml_file, 'r') as stream:
            try:
                exchange_settings = yaml.safe_load(stream)
                self.exchange_name = exchange_settings.get("exchange_name", "default")
                self.remote_adress = exchange_settings.get("remote_adress", None)
                self.auth = tuple(exchange_settings.get("auth", {}).values())
            except yaml.YAMLError:
                logging.warning("Model Exchange Configuration could not be loaded!")

        self.initiate_client()
        self.update_model_exchange()

    def get_exchange_name(self):
        return self.exchange_name

    def initiate_client(self):
        if self.remote_adress is not None:
            self.webdav_client = Client(
                self.remote_adress,
                auth=self.auth
            )

    def update_remote_adress(
        self,
        exchange_name: str,
        remote_adress: str,
        auth: tuple[str, str]
    ):
        self.exchange_name = exchange_name
        self.remote_adress = remote_adress
        self.auth = auth
        #Update config file

        config = f"""
            # This should not contain sensitive data!
            # ReadOnly!
            exchange_name: {exchange_name}
            remote_adress: {remote_adress}
            auth:
                username: {auth[0]}
                password: {auth[1]}
        """
        config = yaml.safe_load(config)
        with open(self.yaml_file, 'w') as file:
            yaml.dump(config, file)

        self.update_model_exchange()

    def download_file(self, remote_path: str):
        try:
            local_path = os.path.join(f"exchange_models/{self.exchange_name}", os.path.basename(remote_path))
            if remote_path.endswith(".onnx") and not os.path.isfile(local_path):
                self.webdav_client.download_file(remote_path, local_path)
        except Exception as e:
            logging.info(f'Error downloading {remote_path}: {e}')

    #@time_cache(60*60*24)
    def update_model_exchange(self):
        logging.info(f"Updating external Model ExChanGe...")
        if self.webdav_client is not None:
            try:
                folders = self.webdav_client.ls("/", detail=False)
                files = [self.webdav_client.ls(f"{folder}/", detail=False) for folder in folders]
                flat_file_list = [file_path for sublist in files for file_path in sublist]
                os.makedirs(os.path.join(f"exchange_models/{self.exchange_name}"), exist_ok=True)

                with ThreadPoolExecutor(max_workers=4) as executor:
                    executor.map(self.download_file, flat_file_list)
                
                logging.info(f"Model ExChanGe successfully updated!")
            except Exception as e:
                logging.warning(f"Cant update models: {e}")
        
        else:
            logging.warning("Cant update models.")
