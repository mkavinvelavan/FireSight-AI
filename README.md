# FireSight AI — Fire and Smoke Image Classification Using Vision Transformers

FireSight AI is an educational Streamlit application for exploring image-level fire and smoke classification with a fine-tuned Vision Transformer (ViT-B/16). It provides image upload and prediction, saved experiment visualizations, an optional local dataset gallery, and descriptions of the training workflow and limitations.

## Objectives

- Demonstrate transfer learning for multi-label fire and smoke image classification.
- Provide an interactive interface for inspecting model scores on uploaded images.
- Present the experiment outputs and data workflow recorded in the project notebook.

## Model and technologies

- **Architecture:** ViT-B/16, fine-tuned with two independent outputs for multi-label classification.
- **Pretrained checkpoint:** `google/vit-base-patch16-224-in21k` (recorded in the training notebook).
- **Inference:** PyTorch and Hugging Face Transformers.
- **Interface:** Streamlit.
- **Image handling:** Pillow.
- **Class mapping:** class `0` = Fire; class `1` = Smoke.

The application applies sigmoid to the two output logits and compares each score independently with a threshold of 0.50 by default. An image may therefore be predicted as Fire, Smoke, both, or neither.

## Project structure

```text
FireSight-AI/
├── app.py
├── requirements.txt
├── README.md
├── MODEL_SETUP.md
├── .gitignore
├── assets/
│   ├── augmentation_example.png
│   ├── confusion_matrices.png
│   ├── test_prediction_examples.png
│   ├── training_loss.png
│   └── validation_macro_f1.png
├── best_model/       # Local model export; ignored by Git
└── dataset/          # Optional local gallery data; ignored by Git
```

The visualizations in `assets/` are small project artifacts and are included. The model weights and dataset remain local because they are not appropriate to store in a standard GitHub repository.

The original Colab training notebook remains in the workspace. It contains about 4 MB of saved cell output and is excluded from Git because it is not needed to deploy this application.

## Dataset and attribution

The dataset used in this project is the **Indoor Fire Smoke Dataset**, obtained from [https://zenodo.org/records/15826133](https://zenodo.org/records/15826133). Credit the dataset to **Arya Krisna Putra, “Indoor Fire Smoke Dataset,” Zenodo, record 15826133**, under the **CC BY 4.0** license specified in the dataset record. The record provides the downloadable `Indoor Fire Smoke.zip` archive (about 200.5 MB). The dataset is not committed to this GitHub project; `.gitignore` excludes the local `dataset/` directory.

To use the local gallery, download the archive from the Zenodo record, extract it, and place the split folders in `dataset/` in this form:

```text
dataset/
├── train/images/  train/labels/
├── valid/images/  valid/labels/
├── test/images/   test/labels/
└── data.yaml
```

The class mapping is preserved: class `0` = Fire and class `1` = Smoke. The project uses the train, validation, and test splits and converts YOLO annotations into image-level labels; an image can contain Fire, Smoke, or both. Set `DATASET_DIR` if the extracted dataset is stored somewhere other than `dataset/`; relative paths are resolved from the project directory. The dataset is optional at runtime: if it is absent, the app displays a message that the local gallery is unavailable, while the other sections remain usable.

## Evaluation results recorded in the notebook

The app displays the experiment results saved with this project:

- **Test subset accuracy:** 0.9667 (96.67%).
- **Test macro F1:** 0.9839 (98.39%).
- **Threshold:** 0.50.
- Per-class precision, recall, F1, positive support, and confusion matrix counts are displayed in the Model Performance section. The report values are rounded as shown in the notebook output.

These are results from the recorded test evaluation, not a guarantee of performance on new data. The saved figures in `assets/` are optional at runtime and display fallback messages if missing.

## Install and run locally

From the project directory, create and activate a virtual environment, then install the listed dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

On macOS or Linux, activate with `source .venv/bin/activate`. Open the local URL printed by Streamlit.

### Configure the model

For local inference, put the exported files (`config.json`, `model.safetensors`, `preprocessor_config.json`, and `class_labels.json`) under `best_model/`. The app loads that local directory by default. See [MODEL_SETUP.md](MODEL_SETUP.md) for hosted model setup.

The trained `model.safetensors` file is about 343 MB, exceeding GitHub's normal 100 MB file limit. Keep it out of ordinary Git commits. For a public deployment, place the model files in a Hugging Face Hub model repository and configure `HF_MODEL_ID` as described in [MODEL_SETUP.md](MODEL_SETUP.md). A public model repository does not require a token. If you use a private repository, configure its access token as a Streamlit secret and add the corresponding authentication setup before deploying; never commit tokens.

Optional environment settings:

| Variable | Purpose | Default |
|---|---|---|
| `HF_MODEL_ID` | Hugging Face model repository ID | Empty; use local `best_model/` |
| `LOCAL_MODEL_DIR` | Local model folder | `best_model` |
| `DATASET_DIR` | Local dataset folder for the gallery | `dataset` |
| `PREDICTION_THRESHOLD` | Independent threshold for each class | `0.5` |

Relative paths are resolved from the project directory. Streamlit Community Cloud reads `HF_MODEL_ID` from app Secrets as well as from the environment.

## Application sections

The app retains its eight navigation sections:

1. **Overview Dashboard** — project summary and experiment snapshot.
2. **Live Image Analysis** — upload one image, run inference, and download JSON or CSV results.
3. **Sample Image Gallery** — display saved prediction examples from the notebook.
4. **Model Performance** — show recorded metrics, confusion matrices, and performance plots.
5. **Project & Workflow** — explain the training and inference pipeline.
6. **Dataset & Augmentation** — browse a local dataset and display the saved augmentation example.
7. **Use Cases & Impact** — describe research applications and future work.
8. **About & Limitations** — summarize technologies, intended use, and limitations.

## Limitations and intended use

This is an educational and research prototype. It classifies an entire uploaded image and does not provide object locations, monitor live video, or confirm an emergency. Results may be affected by dataset coverage, lighting, occlusion, image quality, and differences between training and deployment scenes. Scores are not calibrated estimates of real-world risk. A negative prediction does not establish that a scene is safe. Do not use this application in place of certified alarms, sensors, emergency procedures, or professional assessment.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository and push this project, keeping `.venv/`, `dataset/`, and `best_model/` excluded. The `assets/` folder and application files should be committed.
2. Make the trained model files available in a Hugging Face Hub repository. The model repository must contain the four files listed in `MODEL_SETUP.md`; keep it public for the documented no-token setup.
3. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app from the GitHub repository, choose the `main` branch, and set the app file to `app.py`.
4. In the app's **Settings → Secrets**, add:

   ```toml
   HF_MODEL_ID = "your-hugging-face-name/your-model-repository"
   ```

5. Deploy. The dataset gallery will report that local data is missing unless a dataset is separately provided; image upload and inference use the configured hosted model.

The app has not been deployed as part of preparing this repository.
