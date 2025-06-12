# Sentiment Analysis of Developer Discussions

This project analyzes the sentiment of developer discussions from GitHub issue comments using basic machine learning techniques. It implements both Logistic Regression and Support Vector Machine (SVM) classifiers to categorize comments as positive, negative, or neutral.

- Fetches GitHub issue comments using the GitHub API
- Preprocesses text data for machine learning
- Implements two ML models: Logistic Regression and SVM
- Compares model performance metrics
- Provides sentiment prediction for new comments

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create a `.env` file and add your GitHub token:
```
GITHUB_TOKEN=your_token_here
```
