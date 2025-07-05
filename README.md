# Sentiment Analysis of Developer Discussions

This project analyzes the sentiment of developer discussions from GitHub issue comments using machine learning techniques. It implements multiple classifiers to categorize comments as positive, negative, or neutral, achieving high accuracy on real-world data.

## Tech Stack
- **Python 3.12**
- **pandas, numpy** (data handling)
- **scikit-learn** (ML models, preprocessing, evaluation)
- **requests, python-dotenv** (GitHub API, environment management)
- **matplotlib, seaborn** (visualization)
- **joblib** (model persistence)

## Methodology
1. **Data Collection:**  
   - Fetched 15,000+ real GitHub issue comments from 53 major open-source repositories using the GitHub API.
   - Comments were automatically labeled for sentiment (positive, negative, neutral) using keyword heuristics.

2. **Preprocessing:**  
   - Cleaned and normalized text (lowercasing, removing URLs, code, mentions, etc.).
   - Balanced the dataset for fair model training.

3. **Model Training:**  
   - Trained and compared four models: Logistic Regression, SVM, Random Forest, and Naive Bayes.
   - Used TF-IDF vectorization with n-grams and stopword removal.
   - Performed cross-validation and hyperparameter tuning.

4. **Evaluation & Visualization:**  
   - Evaluated models using accuracy, precision, recall, F1-score, and confusion matrix.
   - Visualized class distribution and confusion matrix (see `plots/`).

## Results
- **Best Model:** SVM (Support Vector Machine)
- **Test Accuracy:** ~85% on real, balanced GitHub data
- **Class-wise F1-scores:**  
  - Negative: 0.88  
  - Neutral: 0.83  
  - Positive: 0.93
- **Visualizations:**  
  - Class distribution and confusion matrix plots saved in `plots/`

## Features
- Fetches GitHub issue comments using the GitHub API
- Preprocesses text data for machine learning
- Implements multiple ML models with performance comparison
- Provides sentiment prediction for new comments
- Generates comprehensive visualizations and reports

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
1. **Collect real GitHub data:**
```bash
python src/enhanced_data_collector.py
```

2. **Train and evaluate models:**
```bash
python src/enhanced_trainer.py
```

3. **Generate visualizations:**
```bash
python src/visualize_results.py
```

4. **Make predictions:**
```bash
python src/predict.py
```

## Project Structure
- `src/enhanced_data_collector.py`: Enhanced GitHub data collection
- `src/enhanced_trainer.py`: Advanced model training and evaluation
- `src/visualize_results.py`: Results visualization and analysis
- `src/synthetic_dataset_generator.py`: Synthetic data generation
- `src/predict.py`: Making predictions on new data
- `data/`: Directory for storing collected data
- `models/`: Directory for saving trained models
- `plots/`: Generated visualizations and charts

## Data Sources
The project collects data from 53 major open-source repositories including:
- TensorFlow, PyTorch, React, Vue.js, Angular
- Node.js, Python, TypeScript, Rust, Go
- Kubernetes, Docker, Terraform, Ansible
- And many more popular developer tools and frameworks
# redeploy
