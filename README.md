# Sentiment Analysis for Farsi Hotel Reviews using BiLSTM

A multi-output sentiment analysis system for Persian/Farsi hotel reviews using Bidirectional Long Short-Term Memory (BiLSTM) networks with Word2Vec embeddings.

## Overview

This project implements a deep learning-based sentiment analysis model that classifies Farsi hotel reviews across five service quality dimensions based on the SERVQUAL framework:

- **Tangibles**: Physical facilities, equipment, and appearance
- **Reliability**: Ability to perform the promised service dependably
- **Empathy**: Caring, individualized attention to customers
- **Assurance**: Knowledge and courtesy of employees and their ability to inspire trust
- **Responsiveness**: Willingness to help customers and provide prompt service

Each dimension is classified into sentiment categories (negative, neutral, positive, very positive).

## Dataset

- **Source**: Hotel reviews in Farsi/Persian language
- **Size**: 2,558 reviews
- **Format**: Excel file (`dataa.xlsx`)
- **Features**:
  - `reviews`: Raw review text in Farsi
  - `Tangibles`: Sentiment score (-1 to 2)
  - `Reliability`: Sentiment score (-1 to 2)
  - `Empathy`: Sentiment score (-1 to 2)
  - `Assurance`: Sentiment score (-1 to 2)
  - `Responsiveness`: Sentiment score (-1 to 2)

## Model Architecture

### Neural Network Design

The model uses a multi-output architecture with shared layers and dimension-specific output heads:

1. **Input Layer**: Padded sequences (max length: 200 tokens)
2. **Embedding Layer**:
   - Pre-trained Word2Vec embeddings (300 dimensions)
   - Vocabulary size: ~8,292 words
   - Non-trainable weights
3. **LSTM Layer**: 128 units with return sequences
4. **Bidirectional LSTM Layer**: 64 units (128 total with bidirectional)
5. **Output Heads**: Separate dense layers for each dimension with:
   - Dropout layers for regularization
   - Dimension-specific architectures optimized for each task
   - Softmax activation for multi-class classification

### Model Performance

Test set results (5% split):

| Dimension | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| Tangibles | 72.66% | 69.74% | 72.66% | 70.66% |
| Reliability | 78.91% | 62.26% | 78.91% | 69.60% |
| Empathy | 93.75% | 87.89% | 93.75% | 90.73% |
| Assurance | 82.03% | 80.16% | 82.03% | 80.21% |
| Responsiveness | 84.38% | 71.19% | 84.38% | 77.22% |

## Requirements

- Python 3.x
- TensorFlow/Keras
- pandas
- numpy
- hazm (Persian NLP library)
- gensim (Word2Vec)
- scikit-learn
- matplotlib
- seaborn

## Installation

```bash
# Install required packages
pip install tensorflow pandas numpy hazm==0.7.0 nltk==3.3 gensim scikit-learn matplotlib seaborn

# Note: The project uses specific versions for compatibility
pip install numpy==1.23
```

## Usage

### Running the Notebook

The main analysis is contained in `sentiment_bilstm(1).ipynb`. The workflow includes:

1. **Data Loading and EDA**
   - Load dataset from Excel file
   - Explore data structure and statistics
   - Handle missing values and duplicates

2. **Text Preprocessing**
   - Character distribution analysis
   - Text normalization using Hazm
   - Lemmatization
   - Number and character mapping
   - Filtering unwanted characters

3. **Text Cleaning Steps**
   - Convert Arabic-Indic digits to standard digits
   - Normalize equivalent Persian letters
   - Remove stopwords
   - Apply lemmatization
   - Filter non-Persian characters

4. **Feature Engineering**
   - Tokenization
   - Sequence padding/truncation
   - Word2Vec embedding training

5. **Model Training**
   - Train/test split (95/5)
   - Early stopping with patience of 5 epochs
   - Batch size: 512
   - Validation split: 10%
   - Optimizer: Adam
   - Loss: Sparse categorical cross-entropy

6. **Evaluation**
   - Performance metrics calculation
   - Visualization of training history
   - Confusion matrices and detailed metrics

## Text Preprocessing Pipeline

### Character Mapping

The preprocessing pipeline includes:

1. **Digit Normalization**:
   - Extended Arabic-Indic digits (۰-۹) → Western digits (0-9)
   - Arabic-Indic digits (٠-٩) → Western digits (0-9)

2. **Letter Normalization**:
   - Various forms of 'heh' → standard Persian 'ه'
   - Various forms of 'yeh' → standard Persian 'ی'
   - 'noon ghunna' → standard 'ن'

3. **Allowed Characters**:
   - Persian alphabet: ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی
   - Additional characters: ۀءآأؤإئة
   - English letters, numbers, and emojis (configurable)

### Stopword Removal

Uses the Hazm library's Persian stopword list to filter common words that don't contribute to sentiment.

## Project Structure

```
sentiment_bilstm_farsi/
│
├── sentiment_bilstm(1).ipynb          # Main Jupyter notebook with full pipeline
├── dataa.xlsx                          # Dataset (hotel reviews)
├── word2vec.model                      # Pre-trained Word2Vec model
├── export_alphabet_distribution.csv    # Character frequency analysis
├── export_other_alphabet.csv          # Non-standard characters found
├── report.pdf                         # Project report (if available)
└── download*.png                      # Visualization plots (training curves)
```

## Key Features

1. **Multi-Output Classification**: Simultaneously predicts sentiment across five service dimensions
2. **Persian NLP**: Specialized preprocessing for Farsi text using Hazm library
3. **Transfer Learning**: Uses pre-trained Word2Vec embeddings
4. **Regularization**: Dropout layers to prevent overfitting
5. **Early Stopping**: Automatic training termination based on validation loss
6. **Comprehensive Evaluation**: Detailed metrics including accuracy, precision, recall, and F1-score

## Model Training Details

### Hyperparameters

- **Embedding Dimension**: 300
- **Max Sequence Length**: 200
- **LSTM Units**: 128 (first layer), 64 (bidirectional layer = 128 total)
- **Batch Size**: 512
- **Training Epochs**: Up to 100 (with early stopping)
- **Early Stopping Patience**: 5 epochs
- **Validation Split**: 10%
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Cross-entropy

### Dropout Strategy

Different dropout rates are applied based on dimension complexity:
- **Tangibles**: 10% dropout
- **Empathy**: 25% dropout
- **Reliability**: 25% dropout
- **Responsiveness**: Multi-layer with varying dropout (25%, 6.25%, 12.5%)
- **Assurance**: 6.25% and 12.5% dropout

## Visualizations

The project includes several visualizations:

1. **Text Length Distribution**: Histogram showing review length distribution
2. **Character Frequency**: Analysis of Persian character usage
3. **Training Curves**: Loss and accuracy plots for each dimension
4. **Model Performance**: Per-dimension accuracy comparison

## Data Preprocessing Statistics

- **Original Dataset**: 2,558 reviews
- **Duplicates Removed**: 5 records
- **Missing Values**: 1 (filled with mode)
- **Final Dataset**: 2,557 reviews
- **Unique Characters**: 166 (before cleaning)
- **Vocabulary Size**: 8,292 unique words
- **Average Review Length**: Varies (10-1,987 characters)

## Challenges and Solutions

1. **Persian Text Normalization**: Used Hazm library for proper handling of Persian-specific characters
2. **Multiple Output Labels**: Implemented multi-output architecture with shared feature extraction
3. **Imbalanced Classes**: Applied weighted loss functions and dropout regularization
4. **Overfitting**: Used early stopping, dropout, and validation monitoring

## Future Improvements

- Implement attention mechanisms for better context understanding
- Experiment with transformer-based models (BERT for Persian)
- Increase dataset size for better generalization
- Add cross-validation for more robust evaluation
- Implement model ensembling
- Deploy as a REST API for real-time predictions

## License

This project is available for educational and research purposes.

## Acknowledgments

- Hazm library for Persian NLP tools
- Word2Vec implementation by Gensim
- SERVQUAL framework for service quality dimensions

## Citation

If you use this code or dataset in your research, please cite appropriately.

---

**Note**: This project was developed for sentiment analysis research on Persian hotel reviews using deep learning techniques.
