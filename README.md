<h1>
    <span style="color: #c54442;">E</span>x<span style="color: #c54442;">C</span>han<span style="color: #c54442;">G</span>eAI
</h1>
ExChanGeAI is an open-source, user-friendly ECG analysis framework. Features include visualization, data transformation, prediction, and model fine-tuning with user datasets. No prior machine learning expertise needed. Deployable via Docker, its ensures data privacy on local hardware. Packed with pretrained models for quick finetuning.

Visit [https://exchange-ai.uni-muenster.de](https://exchange-ai.uni-muenster.de), for an installation free demo version.

## Quickstart
### Installation
Install the end-to-end platform: [Installation](Installation.md).

### Getting Started
See a full introduction [here](Getting_Started.md).

Example data is available in this repository under [example_data.zip](example_data.zip).
You can upload data and labels on the side modal, and select or upload models on the model exchange. The side modal is accesible via top right menu, and the model exchange under `Model ExChanGe`.
The `Model ExChanGe` contains local (Prediction, Training) and downloaded, external models are available under <span style="color: #c54442;">E</span>x<span style="color: #c54442;">C</span>han<span style="color: #c54442;">G</span>e. The models are marked with an "*" for easier differentiation. 

### How to:
- Load the dataset and labels in the modal
- Open the `Finetune` tab
- Select a fitting base model
- Choose a finetuning method
- Set a meaningful name
- accept the terms
- Click `Finetune`

### Custom models (advanced):
- We are compatible with pytorch `.pt` and ONNX `.onnx` models.
- Any classification layer with `head` in its name, will be trained with the option: `finetuning (head)`.
- For ONNX, we require the batch size to be dynamic during export.
- Pytorch models, if custom, require their definition to be added in the `model_definitions` folder. (Please inquire a pull request if you want it to be added in regular releases)
- The models should contain the attributes as metadata
    - target_keys
    - standardizer None

If none are given it will be using atribrary keys and no standardizer. Pytorch model should contain them as attributes. It can be added with the following code to ONNX models:

    meta = model.metadata_props.add()
    meta.key = "target_keys"
    meta.value = str(match_keys)

    meta = model.metadata_props.add()
    meta.key = "standardizer"
    meta.value = str("minMax")

## Citation

    @misc{bickmann2026exchangeai,
        title={End-to-End Platform for Electrocardiogram Analysis and Model Fine-Tuning: Development and Validation Study}, 
        author={Lucas Bickmann and Lucas Plagwitz and Antonius Büscher and Lars Eckardt and Julian Varghese},
        year={2026},
        journal={J Med Internet Res},
        url={https://www.jmir.org/2026/1/e81116}, 
        doi={10.2196/81116},
        pages={e81116},
        volume={28}
    }