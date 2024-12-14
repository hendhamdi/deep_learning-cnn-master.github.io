# Image Classification with a Convolutional Neural Network (CNN)

## Description
This project uses PyTorch to create, train, and evaluate a convolutional neural network (CNN) for image classification. The dataset is split into training (80%) and testing (20%) sets, and metrics such as loss and accuracy are tracked to analyze the model's performance.

## Features
 **Data Splitting** : Splitting the dataset into training (80%) and testing (20%) sets using  `split.py` script.
- **CNN Model** : Construction of a network with convolutional layers, normalization, activation (ReLU), pooling, and a fully connected layer for classification.
- **Advanced Optimization** : Implementation of the SGD (Stochastic Gradient Descent) algorithm with hyperparameter tuning such as learning rate and momentum.
- **Performance Analysis** : Tracking metrics across epochs, including loss and accuracy.
- **Visualization** : Generation of a graph illustrating loss and accuracy over epochs, saved as a PDF.

## Dataset
The dataset contains the following classes:
- Annual Crop
- Forest
- River
- Sea Lake
- Highway
- Industrial
- Pasture
- Permanent Crop
- Residential
- Herbaceous Vegetation

## Dependencies
Ensure the following libraries are installed:
- Python 3.x
- PyTorch
- Matplotlib
- Scikit-learn
- MySQL (if additional storage is needed)

## Usage
### Data Splitting
The `split.py` script in the  `other` directory splits the dataset into training and testing sets:
    ```python
    from sklearn.model_selection import train_test_split
    
An example of usage is included in the script.

### Model Training
. Training Phase : Adjusting weights through backpropagation.

.Testing Phase : Evaluating the model's ability to generalize.

## Results
.Final Accuracy : 85% (training), 81% (testing).

.Observed Trends : Progressive decrease in loss, consistent increase in accuracy.

## Visualization
A graph illustrating loss and accuracy across epochs is generated and saved as a PDF.

![image](https://github.com/user-attachments/assets/fc3eea4d-0a4c-419a-98e6-85de635e511e)
![image](https://github.com/user-attachments/assets/202f89dd-5432-4479-bf0b-e5b6c99158a6)

