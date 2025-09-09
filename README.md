# Requirements
diffusers>=0.21.0
transformers>=4.25.0
accelerate>=0.20.0
torch>=1.13.0
torchvision>=0.14.0
timm>=0.6.0
modelscope>=1.9.0
opencv-python>=4.5.0
scikit-image>=0.19.0

# Models
## Face Adapter Checkpoint
Download from: https://huggingface.co/Ledger039/adapter_dfa/tree/main
File: adapter_dfa.ckpt
## TransFace Model
File: ms1mv2_model_TransFace_S.pt
## Stable Diffusion
Model ID: runwayml/stable-diffusion-v1-5
## ModelScope Models
Face Detection: damo/cv_resnet50_face-detection_retinaface
Human Parsing: damo/cv_resnet101_image-multiple-human-parsing