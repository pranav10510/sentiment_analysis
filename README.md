
# Sentiment Analysis of Movie Reviews

## Overview

This project implements three machine learning algorithms: Naive Bayes, Logistic Regression, and Multilayer Perceptron (MLP) to perform sentiment analysis on the IMDb movie reviews dataset. The performance of the models is evaluated using two feature extraction methods: Term Frequency (TF) and Term Frequency-Inverse Document Frequency (TF-IDF).

## Dependencies

- Python 3.x
- Jupyter Notebook
- NLTK
- scikit-learn
- tensorflow
- numpy
- pandas
- matplotlib

## Installation

1. **Clone the Repository**
   
2. **Install the Required Packages**
   You can install the required packages using `pip`:
   ```
   pip install nltk scikit-learn tensorflow numpy pandas matplotlib
   ```

3. **Download NLTK Data**
   ```
   python -m nltk.downloader movie_reviews
   python -m nltk.downloader punkt
   python -m nltk.downloader stopwords
   python -m nltk.downloader wordnet
   ```

## Running the Code

1. **Open the Jupyter Notebook**
   ```
   jupyter notebook
   ```

2. **Load the Notebook**
   Open the `Assignment1_2.ipynb` notebook from the Jupyter interface.

3. **Run the Notebook**
   Execute the cells in the notebook sequentially to preprocess the data, train the models, and evaluate their performance.

## Project Structure

- `Assignment1_2.ipynb`: The main Jupyter Notebook containing the code for data preprocessing, training, and evaluation.
- `README.md`: Instructions on how to set up and run the project.

## Results

The performance of the models is compared using the following metrics:
- Accuracy
- True Positive Rate (TPR)
- False Positive Rate (FPR)

### Accuracy Comparison

- Naive Bayes (TF): 77%
- Naive Bayes (TF-IDF): 75%
- Logistic Regression (TF): 82%
- Logistic Regression (TF-IDF): 79%
- MLP (TF): 83%
- MLP (TF-IDF): 81%

### Insights

- **TF-IDF** generally improves performance metrics across all models, indicating its effectiveness in enhancing text feature representation.
- **Naive Bayes** performs well with both TF and TF-IDF, though the difference in performance is marginal.
- **Logistic Regression** benefits more from TF-IDF due to its ability to handle feature importance more effectively.
- **MLP** shows the highest accuracy among the three algorithms with TF, though the difference with TF-IDF is minimal.

## Conclusion

This project demonstrates the application of different machine learning algorithms and feature extraction methods for sentiment analysis. The choice of algorithm and feature extraction method should consider the specific context of the task, including dataset size, computational resources, and the need for interpretability.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- IMDb for providing the movie reviews dataset.
- NLTK for natural language processing tools and resources.
- Scikit-learn for machine learning tools.
- TensorFlow for neural network models.
