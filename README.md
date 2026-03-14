# Fake-news-detection

# Approach

This project follows a complete Natural Language Processing workflow for building a fake news classification system using transformer-based models.

**Data Ingestion:**
The dataset was obtained from Kaggle using the Fake and Real News dataset. It consists of two files containing news articles labeled as fake or real. These datasets were merged into a single dataframe to create a unified dataset for training and evaluation.

**Preprocessing:**
The title and main article text were combined into a single input sequence using a `[SEP]` token. This allows the model to capture contextual information from both the headline and the article body.

**Tokenization:**
The textual data was tokenized using the `BertTokenizer` from the HuggingFace Transformers library. Tokenization converts raw text into numerical tokens that the model can process. Padding and truncation were applied with a maximum sequence length of 128 tokens to maintain consistent input sizes and optimize memory usage.

**Data Splitting:**
The dataset was divided into training and validation sets using an 80/20 split. This allowed the model to learn patterns from the training data and then be evaluated on unseen validation data.

**Model Training:**
A pretrained transformer model was fine-tuned for sequence classification using the HuggingFace Trainer API. The training process involved updating the pretrained weights on the fake news dataset over multiple epochs.

**Deployment:**
A simple web interface was created using Gradio. This interface allows users to enter a news headline or article and receive an instant prediction indicating whether the news is real or fake.

# Model Used

**Primary Model:**
BERT (Bidirectional Encoder Representations from Transformers)

**Architecture:**
`bert-base-uncased`

**Improved Model:**
DistilBERT (a lighter and faster variant of BERT used for comparison and performance improvement)

**Frameworks and Libraries Used:**

* HuggingFace Transformers
* PyTorch
* scikit-learn
* pandas
* Gradio

# Metrics

The model was evaluated using standard classification metrics.

**Evaluation Metrics Used:**

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

These metrics provide a comprehensive understanding of how well the model distinguishes between fake and real news articles.

# Improvements & Error Analysis

During evaluation, several misclassified samples were analyzed to understand where the model struggled. Most errors occurred in cases where the news articles were extremely short or lacked sufficient contextual information.

To improve model performance, DistilBERT was implemented as an alternative architecture. DistilBERT reduces the number of parameters while maintaining strong performance, resulting in faster training and inference.

This comparison demonstrated how different transformer architectures can affect efficiency and predictive performance.

# Key Learnings

This project provided valuable hands-on experience with modern Natural Language Processing techniques and transformer-based models.

Key learnings include:

* Understanding the complete NLP pipeline from data preprocessing to model deployment.
* Implementing custom PyTorch Dataset classes for efficient data loading.
* Fine-tuning pretrained transformer models using the HuggingFace Trainer API.
* Evaluating model performance using classification metrics and confusion matrices.
* Performing error analysis to identify weaknesses in the model.
* Building a simple deployment interface using Gradio to enable real-time predictions.


