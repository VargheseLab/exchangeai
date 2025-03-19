<h1>
    <span style="color: #c54442;">E</span>x<span style="color: #c54442;">C</span>han<span style="color: #c54442;">G</span>eAI
</h1>
ExChanGeAI is an open-source, user-friendly ECG analysis framework. Features include visualization, data transformation, prediction, and model fine-tuning with user datasets. No prior machine learning expertise needed. Deployable via Docker, its ensures data privacy on local hardware. Packed with pretrained models for quick finetuning.

## Quickstart
### Installation
Install the end-to-end platform: [Installation](Installation.md).

### Getting Started
See a full introduction [here](Getting_Started.md).

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

    @misc{bickmann2025exchangeai,
        title={ExChanGeAI: An End-to-End Platform and Efficient Foundation Model for Electrocardiogram Analysis and Fine-tuning}, 
        author={Lucas Bickmann and Lucas Plagwitz and Antonius Büscher and Lars Eckardt and Julian Varghese},
        year={2025},
        eprint={2503.13570},
        archivePrefix={arXiv},
        primaryClass={cs.LG},
        url={https://arxiv.org/abs/2503.13570}, 
    }