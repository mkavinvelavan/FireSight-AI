# Model setup

The application requires the trained model for image prediction. The project contains a local export in `best_model/`, including a `model.safetensors` file of about 343 MB. That file exceeds GitHub's normal 100 MB limit and `best_model/` is excluded by `.gitignore`.

## Run with the local export

Keep these exported files under `best_model/` beside `app.py`:

- `config.json`
- `model.safetensors`
- `preprocessor_config.json`
- `class_labels.json`

The `class_labels.json` included with this project maps the outputs to `Fire` then `Smoke`, with a threshold of `0.5`. The model configuration also identifies class 0 as Fire and class 1 as Smoke. When `HF_MODEL_ID` is not configured, the app loads `best_model/`.

The original notebook records the export location as `MyDrive/ViT_Fire_Smoke_Project/best_model/`. Copy the exported files from that location if it is available. No retraining is needed to run inference.

## Prepare a hosted model for Streamlit Community Cloud

1. Create a model repository on Hugging Face Hub and upload the four files above. Git LFS is suitable for transferring the large weight file. Do not add credentials to the repository.
2. For the simplest deployment, make the model repository public. The app can load a public repository without a Hugging Face token.
3. In Streamlit Community Cloud, open the app's **Settings → Secrets** and configure the repository ID:

   ```toml
   HF_MODEL_ID = "your-hugging-face-name/your-model-repository"
   ```

The app checks the `HF_MODEL_ID` environment variable first and then Streamlit Secrets. When neither is configured, it looks for the local `best_model/` directory. If neither model source is available, image prediction reports that model setup is required; the other app sections remain available.

The dataset and model are separate. The dataset is only needed for the local Dataset & Augmentation gallery and is not needed for model inference. See the README for dataset attribution and the local folder layout.
