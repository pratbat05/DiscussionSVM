# Sentiment Analysis of Developer Discussions

This project analyzes the sentiment of developer discussions from GitHub issue comments using basic machine learning techniques. It implements both Logistic Regression and Support Vector Machine (SVM) classifiers to categorize comments as positive, negative, or neutral.

## Features
- Fetches GitHub issue comments using the GitHub API
- Preprocesses text data for machine learning
- Implements two ML models: Logistic Regression and SVM
- Compares model performance metrics
- Provides sentiment prediction for new comments

## Setup
1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create a `.env` file and add your GitHub token:
```
GITHUB_TOKEN=your_token_here
```

## Usage
1. Run data collection:
```bash
python src/data_collector.py
```
2. Train and evaluate models:
```bash
python src/train_models.py
```
3. Make predictions:
```bash
python src/predict.py
```

## Project Structure
- `src/data_collector.py`: Scripts for fetching GitHub comments
- `src/preprocessor.py`: Data preprocessing utilities
- `src/train_models.py`: Model training and evaluation
- `src/predict.py`: Making predictions on new data
- `data/`: Directory for storing collected data
- `models/`: Directory for saving trained models 