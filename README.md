# Disease Prediction Using Facial Diagnosis

Leveraging deep learning for disease prediction through facial analysis, this project explores non-invasive diagnostic techniques by identifying health-related patterns in facial features.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation and Usage](#installation-and-usage)
- [Results](#results)
- [Future Work](#future-work)
- [Contributors](#contributors)

## Project Overview
This project focuses on the prediction of diseases by analyzing facial features. Using advanced deep learning algorithms, the system identifies potential health issues based on facial patterns, allowing for a non-invasive diagnostic tool. Such an approach has potential applications in telemedicine, preventive health monitoring, and accessible healthcare diagnostics.

## Dataset
The dataset consists of labeled images with various facial features that are indicative of different health conditions. The dataset includes:
- **Training Data**: Images with known health labels for training the model.
- **Testing Data**: Images for evaluating model performance.

**Note**: As per privacy concerns, I have used images sourced from Google.

## Model Architecture
We employ a deep learning model based on convolutional neural networks (CNN) to process facial images and predict potential diseases. Key aspects of the model include:
- **Feature Extraction**: Using CNN layers to capture intricate patterns in facial features.
- **Classification**: Predicting the likelihood of specific diseases based on extracted features.
- **Performance Metrics**: Accuracy, precision, recall, and F1-score to evaluate model effectiveness.

## Installation and Usage

### Prerequisites
- Python 3.x
- Required libraries listed in `requirements.txt`

### Steps to Run the Project
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Naidu-2002/Disease-Prediction.git
   cd Disease-Prediction

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt

3. **Run the Model**:
   ```bash
   python main.py

4. **View Results**:
   Output images with prediction results will be diaplayed.

## Results
The model successfully predicted the following diseases based on facial features:
- **Beta Thalassemia**
- **Hyperthyroidism**
- **Down Syndrome**
- **Leprosy**

These predictions were evaluated based on metrics such as accuracy, precision, and recall, with initial results indicating promising accuracy in identifying disease-specific facial patterns.

## Future Work
- **Improvement of Model Accuracy**: Explore other model architectures, including attention mechanisms, to increase prediction accuracy.
- **Real-time Prediction**: Develop a real-time application for continuous monitoring.
- **Expand Dataset**: Gather a more diverse and comprehensive dataset to improve model robustness.
- **Mobile Application Integration**: Create a mobile app to allow users to access the diagnostic tool from their smartphones.
- **Cross-Disease Prediction**: Expand the model’s capabilities to predict multiple diseases simultaneously based on facial features.
- **Explainability and Interpretability**: Integrate model interpretability techniques to help users understand which facial features are associated with different predictions.

## Contributors
- **Naidu-2002** - Project lead and developer
