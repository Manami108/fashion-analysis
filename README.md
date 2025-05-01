# Fashion Image Analysis and Classification
## Introduction
Due to the rapid development of e-commerce, particularly in the fashion industry, manual analysis of fashion images, clothing classification, and fashion style prediction have become difficult over time. To address these challenges, this project integrates classification and segmentation models using deep learning. <br>

This system includes two pipelines:
- Pipeline 1: Uses segmentation models to classify clothing types.
- Pipeline 2: Uses classification models to predict fashion styles.
  
The models were trained and evaluated using DeepFashion2 and FashionStyle14 datasets. Various deep learning architectures such as ResNet, DenseNet, and a RandomForest classifier were employed to evaluate performance through 5-fold cross-validation.

## Features
- Garment Classification: Indentifies clothing parts using segmentation.
- Fashion Style Classification: Categorizes fashion styles.
- Models: Implements ResNet, DenseNet, and Random Forest for classification to compare the performance.
- Multi-label Classification: Achieves high accuracy for predicting 13 clothing labels and 14 fashion styles.
- Cross-Validation: Uses 5-fold cross-validation for model selection.
- Final Prediction Pipeline: Integrates the best-performing models for real-world predictions.

## Datasets
The following datasets were used:
- DeepFashion2: Contains 801K clothing items with segmentation masks and 13 clothing categories.
- FashionStyle14: Contains 13K images with 14 Japanese fashion styles.

## Results
After training and evaluation, the following models were selected as the best performers:
- Pipeline 1: DenseNet-121 achieved the highest accuracy of 82.10%.
- Pipeline 2: ResNet-50 achieved 73.16% accuracy.
- Final system: Integrated the best models for real-world prediction.

For a detailed breakdown of accuracy, loss graphs, and confusion matrices, refer to the full report [(PDF included in this repository)](https://github.com/Manami108/Fashion_Analysis/blob/main/fashionAnalysis.pdf).

## Usage
To use the system, follow these steps:
1. Clone the Repository <br>
```bash
   git clone https://github.com/Manami108/Fashion_Analysis.git <br>
   cd Fashion_Analysis
```


2. Download the Datasets <br>
   Download the datasets from <br>
   DeepFashion2:  https://drive.google.com/drive/folders/125F48fsMBz2EF0Cpqk6aaHet5VH399Ok,
   FashionStyle14: https://esslab.jp/~ess/ja/data/fashionstyle14/
   
3. Run Inference on Images <br>
   Run the code to predict clothing type and fashion style <br>
```bash
   python fashionAnalysis.py
```

4. (Train the Model) <br>
   To train the models from scratch <br>
```bash
   Python ___.py
```

## Citation 
If you use this work, please cite: <br>
```bash
@software{yano2025fashion,
  author    = {Manami Yano},
  title     = {Fashion Image Analysis - Integrating Classification and Segmentation Models},
  year      = {2025},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.1234},
  url       = {https://github.com/Manami108/Fashion_Analysis},
  publisher = {GitHub},
}
```

## Dataset References
- DeepFashion2: <br>
  @article{DeepFashion2,
  author = {Yuying Ge and Ruimao Zhang and Lingyun Wu and Xiaogang Wang and Xiaoou Tang and Ping Luo},
  title={A Versatile Benchmark for Detection, Pose Estimation, Segmentation and Re-Identification of Clothing Images},
  journal={CVPR},
  year={2019}
  }
- FashionStyle14: <br>
  @InProceedings{TakagiICCVW2017,
  author    = {Moeko Takagi and Edgar Simo-Serra and Satoshi Iizuka and Hiroshi Ishikawa},
  title     = {{What Makes a Style: Experimental Analysis of Fashion Prediction}},
  booktitle = "Proceedings of the International Conference on Computer Vision Workshops (ICCVW)",
  year      = 2017,
  }
