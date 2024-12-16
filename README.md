# DLF: Disentangled-Language-Focused Multimodal Sentiment Analysis, AAAI 2025.

## Main Contributions

Our main contributions can be summarized as follows:

- **Proposed Framework:** In this study, we propose a Disentangled-Language-Focused (DLF) multimodal representation learning framework to promote MSA tasks. The framework follows a structured pipeline: feature extraction, disentanglement, enhancement, fusion, and prediction.
- **Language-Focused Attractor (LFA):** We develop the LFA to fully harness the potential of the dominant language modality within the modality-specific space. The LFA exploits the language-guided multimodal cross-attention mechanisms to achieve a targeted feature enhancement ($X$$\rightarrow$Language).
- **Hierarchical Predictions:** We devise hierarchical predictions to leverage the pre-fused and post-fused features, improving the total MSA accuracy. 


## The Framework.
![](./imgs/Framework.png)
The framework of DLF. Please refer to [Paper Link](arxiv) for details.


## Usage

### Prerequisites
- Python 3.9.13
- PyTorch 1.13.0
- CUDA 11.7

### Installation
- Create conda environment. Please make sure you have installed conda before.
```
conda create -n DLF python==3.9.13
```
```
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
```
```
pip instal requirements.txt
```

### Datasets
Data files (containing processed MOSI, MOSEI datasets) can be downloaded from [here](https://drive.google.com/drive/folders/1BBadVSptOe4h8TWchkhWZRLJw8YG_aEi?usp=sharing). 
You can first build and then put the downloaded datasets into `./dataset` directory and revise the path in `./config/config.json`. For example, if the processed the MOSEI dataset is located in `./dataset/MOSEI/aligned_50.pkl`. Please make sure "dataset_root_dir": "./dataset" and "featurePath": "MOSI/aligned_50.pkl".
Please note that the meta information and the raw data are not available due to privacy of Youtube content creators. For more details, please follow the [official website](https://github.com/A2Zadeh/CMU-MultimodalSDK) of these datasets.

### Run the Codes
- Training

You can first set the traning dataset name in `./train.py` as "mosei" or "mosi", and then run:
```
python3 train.py
```
By default, the trained model will be saved in `./pt` directory. You can change this in `train.py`.

- Testing

You can first set the testing dataset name in `./test.py` as "mosei" or "mosi", and then test the trained model:
```
python3 test.py
```
We also provide pretrained models for testing. ([Google drive](https://drive.google.com/drive/folders/1GgCfC1ITAnRRw6RScGc7c2YUg5Ccbdba?usp=sharing))


### Citation
If you find the code and our idea helpful in your resarch or work, please cite the following paper.

```
@article{wang2025dlf,
  title={DLF: Disentangled-Language-Focused Multimodal Sentiment Analysis},
  author={Wang, Pan and Zhou, Qiang and Wu, Yawen and Chen, Tianlong and Hu, Jingtong},
  journal={arXiv preprint arXiv:2412},
  year={2024}
}
```




