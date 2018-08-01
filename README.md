# 2D Ensemble U-Nets with autoencoder pretraining scheme for 3D segmentation

This code implements 2D Ensemble U-Nets with autoencoder pretraining scheme for 3D segmentation. 
First, three 2D U-Nets are trained from scratch, respectively in the XY, YZ and XZ direction. For final 3D segmentation, the networks should be applied on the complete volumes and aggregated by e.g. weighted averaging. 
Secondly, three 2D U-Net autoencoders (i.e. no skip connections) are trained from scratch. The encoders are used as an initialization for the 2D U-Net segmentation networks. The XY network is trained with a randomly initialized decoder, the YZ network uses the trained XY decoder and the XZ network uses the YZ decoder to finetune from. 

## Requirements
- Tested with Python 3.5
- Python libraries: 
    - torch 0.4
    - datetime (for logging)
    - tensorboardX (for TensorBoard usage)
    - tifffile (for data loading)
    - [imgaug](https://github.com/aleju/imgaug) (data augmentation) 
- EPFL data should be in the [data](data) folder for testing the demo script. This data is located at /g/kreshuk/data/epfl. 

## Instructions
The algorithm can be tested by running [train_models.py](train/train_models.py). I tried to document most of the code, but if you have any questions, don't hesitate to ask me! 
