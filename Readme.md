# Person Re-Identification with a Locally Aware Transformer


This code is inspired from:


    1) PCB - https://github.com/layumi/Person_reID_baseline_pytorch
    2) Vit - https://github.com/lucidrains/vit-pytorch/tree/main/examples
    3) Pre-trained models: https://github.com/rwightman/pytorch-image-models
    
## Release 7/5/21
Demonstrates the working and performance of the LA-Transformer using two jupyter notebooks.

    1) LA-Transformer Training: Demonstrates the training process. We have included cell outputs in the juyter notebook. In the
    last cell, training results are shown. One can also refer to model/{name}/summary.csv if the cell outputs are not clear. To 
    run the jupyter notebook, install the requirements, download dataset using the link provided and extract it in data folder.

    2) LA-Transformer Testing: Demonstrates the testing process. You can download the weights using the link below or train 
    LA-transformer using the Training notebook. To use pre-trained weights, download them using the gdrive link below, extract
    them into model/{name} folder and run the Testing notebook. Performance metrics can be found in the last cell of the notebook.

## Requirements:

- Torch==1.8.1 & torchvision==0.8.2: [Link](https://pytorch.org/)
- timm==0.3.2: [Link](https://github.com/rwightman/pytorch-image-models)
- faiss==1.6.3: [Link](https://github.com/facebookresearch/faiss)
- tqdm==4.54.0 
- numpy==1.19.5

## Read-Only Versions:
LA-Transformer Training.html and LA-Transformer Testing.html are the read-only versions containing outputs to quickly verfiy the working of LA-Transformer.

## Google Drive:

Pretrained weights and dataset can be found on [this](https://drive.google.com/drive/folders/1CRkfn9iLEItaYur1WGf2abvpd2vT7nRB?usp=sharing) google drive. To remain anonymous we created a temporary gmail account to host weights and datasets. It will be changed to official account later.
