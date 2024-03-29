# KinyaEmformerRNNT

This codebase implements a Emformer model training and inference scripts for lightweight Kinyarwanda streaming speech recognition.
The model can be deployed on a smartphone for real-time speech recognition using the phone's micrcophone.

## Getting started
TODO:
We will be releasing the pre-trained model for Kinyarwanda soon for inference on Android.

### Inference
Generating TorchScript model to be deployed on Android
````
cd Inference/
python generate_ts.py
python save_model_for_mobile.py

````

### Training
Training script using Mozilla Common Voice Kinyarwanda dataset:
````
cd Training/emformer_rnnt/
python train_cv.py

````

