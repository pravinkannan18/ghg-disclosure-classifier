# GHG Disclosure Classifier 🌱

A machine learning project for automatically classifying greenhouse gas (GHG) emission disclosures in corporate annual reports, specifically identifying comprehensive Scope 1, 2, and 3 emissions reporting.

## 🎯 Project Overview

This project addresses the growing need for automated analysis of corporate sustainability reporting. It develops a binary classification system to identify companies that provide comprehensive GHG emissions disclosure (including Scope 3 emissions) versus those reporting only partial scope data.

### Key Features

- **Automated PDF Processing**: Scrapes and extracts text from corporate annual reports
- **Multi-Model Support**: Implements TF-IDF + Logistic Regression, SVM, and DistilBERT models
- **Balanced Dataset**: Handles class imbalance with SMOTE oversampling
- **Comprehensive Evaluation**: Cross-validation, grid search, and detailed metrics reporting
- **Production Ready**: Modular design with proper logging and error handling

## 📊 Dataset

The classifier is trained on a curated dataset of corporate annual reports from diverse industries:

- **Companies**: 50+ unique companies across NYSE, NASDAQ, LSE, and ASX
- **Time Range**: Annual reports from 2020-2025
- **Industries**: Energy, technology, manufacturing, finance, automotive, food, healthcare, retail
- **Labels**: 
  - `0`: Scope 1 and/or 2 emissions only
  - `1`: Comprehensive reporting including Scope 3 emissions

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/pravinkannan18/ghg-disclosure-classifier.git
   cd ghg-disclosure-classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt

   ```

### Usage

#### 1. Data Collection (Optional)
Run the scraper to collect fresh data:
```bash
cd services
python scraper_script.py
```

#### 2. Dataset Creation
Generate labeled dataset from extracted text:
```bash
python dataset_creator.py
```

#### 3. Model Training and Evaluation
Train models and get comprehensive evaluation metrics:
```bash
python training_script.py --validation validation.csv
```

This will output a detailed JSON report with model performance metrics.

## 🏗️ Project Structure

```
ghg-disclosure-classifier/
├── data/
│   ├── dataset.csv          # Labeled training dataset
│   ├── urls.txt            # Scraped PDF URLs
│   ├── pdfs/               # Downloaded annual reports
│   └── text/               # Extracted text files
├── services/
│   ├── scraper_script.py   # PDF scraping and downloading
│   ├── dataset_creator.py  # Automatic labeling system
│   ├── training_script.py  # Model training and evaluation
│   └── validation.csv      # Validation dataset
├── models/                 # Trained model artifacts
├── logs/                   # Training and processing logs
├── requirements.txt        # Python dependencies
└── README.md
```

## 🔬 Methodology

### Data Processing Pipeline

1. **Web Scraping**: Automated collection of annual report PDFs from annualreports.com
2. **Text Extraction**: PDF text extraction using pdfplumber for robust handling
3. **Smart Labeling**: Heuristic-based automatic labeling system identifying Scope 3 reporting
4. **Data Balancing**: SMOTE oversampling for handling class imbalance

### Machine Learning Models

#### 1. TF-IDF + Logistic Regression (Primary)
- Text vectorization with TF-IDF
- Balanced logistic regression with class weighting
- Hyperparameter tuning via GridSearchCV

#### 2. Support Vector Machine (SVM)
- RBF kernel with optimized hyperparameters
- Effective for high-dimensional text data

#### 3. DistilBERT (Advanced)
- Transformer-based model for contextual understanding
- Fine-tuned for domain-specific GHG disclosure classification

### Evaluation Metrics

- **Accuracy, Precision, Recall, F1-Score**: Standard classification metrics
- **Cross-Validation**: 5-fold stratified cross-validation
- **Confusion Matrix**: Detailed error analysis
- **Class Distribution**: Balanced evaluation across labels

## 📈 Performance

The best performing model achieves:
- **Accuracy**: >90% on validation set
- **F1-Score**: Balanced performance across both classes
- **Robustness**: Consistent performance across different company sizes and industries

## 🛠️ Technical Implementation

### Key Dependencies

- **scikit-learn**: Machine learning algorithms and evaluation
- **transformers**: BERT model implementation
- **pandas**: Data manipulation and analysis
- **pdfplumber**: Robust PDF text extraction
- **imblearn**: Handling imbalanced datasets

### Logging and Monitoring

- Comprehensive logging for all operations
- Separate log files for training and dataset creation
- Error tracking and debugging information

## 🔍 Labeling Logic

The automatic labeling system uses sophisticated heuristics:

- **Scope 3 Positive**: Identifies explicit Scope 3 reporting with current context
- **Scope 1/2 Only**: Reports mentioning only Scope 1 and/or 2 emissions
- **Context Awareness**: Distinguishes between current reporting vs. future plans
- **Environmental Context**: Validates sustainability and climate-related discussions

## 📝 Assumptions and Limitations

### Assumptions
- PDF links follow consistent annualreports.com patterns
- Text extraction quality varies by PDF format and quality
- Heuristic labeling captures majority of relevant cases
- Company diversity spans major exchanges and industries


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request



## 📧 Contact

**Pravin K** - [Email](pravinkannan18@gmail.com)

Project Link: [https://github.com/pravinkannan18/ghg-disclosure-classifier](https://github.com/pravinkannan18/ghg-disclosure-classifier)

---

*Building sustainable AI solutions for corporate transparency and environmental accountability* 🌍
