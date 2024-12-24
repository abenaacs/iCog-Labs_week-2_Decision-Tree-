# Sentiment Analysis and Decision Tree Classifier

This project performs sentiment analysis on text data (headlines) using **TextBlob** and classifies the sentiment into three categories: **negative**, **neutral**, and **positive** using a **Decision Tree Classifier**. The decision tree is trained using text features extracted from the dataset and sentiment scores.

The steps are all executed in a **Jupyter Notebook** environment.

## Features

- **Text Preprocessing**: Cleans and standardizes text by converting it to lowercase and removing empty values.
- **Sentiment Analysis**: Calculates sentiment polarity using **TextBlob** and classifies text into three sentiment categories.
- **Decision Tree Classifier**: Trains a model to classify the text data based on sentiment.
- **Model Visualization**: Visualizes the trained decision tree.

## Prerequisites

Ensure you have Python 3.7+ installed on your machine. You will also need to set up a virtual environment and install necessary libraries to run the project.

### 1. Clone the repository

Clone this repository to your local machine:

```bash
git clone https://github.com/abenaacs/iCog-Labs_week-2_Decision-Tree-.git
cd iCog-Labs_week-2_Decision-Tree
```

### 2. Set Up a Virtual Environment

Create a virtual environment to isolate the project dependencies:

```bash
python -m venv env
```

Activate the virtual environment:

- On Windows:

  ```bash
  .\env\Scripts\activate
  ```

- On macOS/Linux:
  ```bash
  source env/bin/activate
  ```

### 3. Install Dependencies

Install the necessary Python packages using **pip**:

```bash
pip install -r requirements.txt
```

If the `requirements.txt` file is not available, you can install the required packages manually using the following commands:

```bash
pip install pandas numpy scikit-learn matplotlib textblob requests jupyter
```

### 4. Obtain the Dataset

This script expects a CSV file containing a **headline** column. To use your own dataset, ensure it has the correct format.

You can also use a dataset from a public source (e.g., Google Drive). Update the `file_path` variable in the notebook to point to your dataset file. Ensure the file is publicly accessible and use the direct download link as described below:

```python
file_path = "https://drive.google.com/uc?export=download&id=<file_id>"
```

### 5. Running the Jupyter Notebook

Once you have set up the environment and obtained the dataset, follow these steps:

1. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

2. Open the `decision_tree.ipynb` file from the Jupyter interface.

3. Run the notebook cell by cell by clicking "Run" or pressing **Shift + Enter**.

The notebook will guide you through the following steps:

- **Loading the Dataset**: Download and load the CSV file (e.g., from Google Drive).
- **Preprocessing Data**: Clean and standardize the text data (convert to lowercase, remove empty entries).
- **Sentiment Analysis**: Use **TextBlob** to analyze the sentiment of each headline and classify them as **negative**, **neutral**, or **positive**.
- **Feature Extraction**: Use **TF-IDF** vectorization to convert the text data into numerical features. Sentiment scores are also included as additional features.
- **Train and Evaluate**: A **Decision Tree Classifier** is trained on the processed features, and its accuracy and performance are evaluated.
- **Visualize the Decision Tree**: Visualize the trained decision tree using **matplotlib**.

### 6. Example Output

The output will include the following:

- **Accuracy** of the trained model on the test dataset.
- **Classification Report** with metrics like precision, recall, and F1-score for each sentiment class.
- **Confusion Matrix** to visualize the model's predictions versus true labels.
- **Decision Tree Visualization** showing how the classifier splits the data at each node.

### 7. Troubleshooting

- **Google Drive Download Issues**: Ensure the dataset is shared with "Anyone with the link can view" permissions. Update the Google Drive link to use the direct download format.
- **Missing Dependencies**: If any required libraries are missing, run `pip install <library_name>` to install them manually.
- **File Format Issues**: Ensure your CSV file is properly formatted (i.e., with a column named `headline`).

### 8. License

This project is licensed under the MIT License.

---

## Example Directory Structure

```
sentiment-analysis/
│
├── decision_tree.ipynb      # Jupyter Notebook for sentiment analysis and decision tree
├── requirements.txt              # List of required Python libraries
├── README.md                     # This README file
└── data/
    └── dataset.csv               # Dataset file containing headlines (CSV format)
```

---

## Conclusion

This project demonstrates how to analyze text data, extract sentiment, and build a machine learning model to classify text using **Decision Tree Classifier**. The decision tree provides a clear way of understanding how the model makes predictions based on text features and sentiment analysis.

Feel free to modify the notebook to work with other datasets or enhance the model by experimenting with different classifiers or preprocessing techniques.

---
