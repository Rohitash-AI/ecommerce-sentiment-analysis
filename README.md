This project is an end-to-end Natural Language Processing (NLP) pipeline that automatically classifies e-commerce customer reviews as Positive, Negative, or Neutral. Built with Python, it processes raw, noisy text data to help businesses instantly analyze customer satisfaction at scale without manual reading.

**Key Workflow:**

* **Data Preprocessing:** Cleans synthetic real-world datasets, handles missing values, and fixes outliers using **Pandas**.
* **NLP Processing:** Normalizes text using regex, stop-word removal, and lemmatization via **NLTK**, then converts the text into numerical vectors using **TF-IDF**.
* **Machine Learning:** Trains a Logistic Regression classification model using **Scikit-Learn** to predict the underlying sentiment of unseen reviews.
* **Evaluation & Visualization:** Assesses model performance using accuracy, precision, and recall metrics, and visualizes the results through confusion matrices and product rating charts using **Matplotlib** and **Seaborn**.
