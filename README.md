# Classifier-Server

A Flask-based REST API server for document classification using machine learning models. This server provides two types of classifiers: a General Classifier and a Confidential Classifier, both built using scikit-learn and deployed as RESTful web services.

## Features

- **General Classification**: Classifies documents into general categories (e.g., sports, politics, technology)
- **Confidential Classification**: Specialized classifier for confidential document categorization
- **REST API**: Easy-to-use HTTP endpoints for text classification
- **Cross-Origin Resource Sharing (CORS)**: Enabled for cross-domain requests
- **Pre-trained Models**: Uses pickled machine learning models for fast inference

## Technology Stack

- **Flask**: Web framework for the REST API
- **Flask-CORS**: Cross-origin resource sharing support
- **scikit-learn**: Machine learning library (TF-IDF vectorization and classification models)
- **Python 3**: Programming language

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AyeshW/Classifer-Server.git
cd Classifer-Server
```

2. Install the required dependencies:
```bash
pip install flask flask-cors scikit-learn
```

3. Ensure the following pickle files are present in the root directory:
   - `gen_clf.pickle` - General classifier model
   - `gen_tfidf.pickle` - General TF-IDF vectorizer
   - `gen_id_map.pickle` - General category ID mapping
   - `conf_clf.pickle` - Confidential classifier model
   - `conf_tfidf.pickle` - Confidential TF-IDF vectorizer
   - `conf_id_map.pickle` - Confidential category ID mapping

## Usage

### Starting the Server

Run the Flask application:
```bash
python app.py
```

The server will start on the default Flask port (5000). You can access the welcome page at:
```
http://localhost:5000/
```

### API Endpoints

#### 1. General Classification

**Endpoint**: `/gen_category`  
**Method**: `POST`  
**Content-Type**: `application/json`

**Request Body**:
```json
[
  {
    "path": "document1.txt",
    "text": "Sri Lanka cricket team won the 1996 world championship"
  },
  {
    "path": "document2.txt",
    "text": "Your text content here"
  }
]
```

**Response**:
```json
[
  {
    "path": "document1.txt",
    "category": "sport"
  },
  {
    "path": "document2.txt",
    "category": "politics"
  }
]
```

#### 2. Confidential Classification

**Endpoint**: `/conf_category`  
**Method**: `POST`  
**Content-Type**: `application/json`

**Request Body**:
```json
[
  {
    "path": "confidential_doc1.txt",
    "text": "Your confidential text content here"
  }
]
```

**Response**:
```json
[
  {
    "path": "confidential_doc1.txt",
    "category": "classified_category"
  }
]
```

### Example Usage with cURL

```bash
curl -X POST http://localhost:5000/gen_category \
  -H "Content-Type: application/json" \
  -d '[{"path": "test.txt", "text": "Sri Lanka cricket team won the 1996 world championship"}]'
```

## Project Structure

```
Classifer-Server/
├── app.py                                      # Flask application with API endpoints
├── classifier.py                               # Classifier classes (General and Confidential)
├── Classifer_General_Classifier_notebook.ipynb # Jupyter notebook for model training
├── gen_clf.pickle                              # General classifier model (pickled)
├── gen_tfidf.pickle                            # General TF-IDF vectorizer (pickled)
├── gen_id_map.pickle                           # General category ID mapping (pickled)
├── conf_clf.pickle                             # Confidential classifier model (pickled)
├── conf_tfidf.pickle                           # Confidential TF-IDF vectorizer (pickled)
├── conf_id_map.pickle                          # Confidential category ID mapping (pickled)
├── Tests/                                      # Test directory
│   ├── __init__.py
│   └── test_generalClassifier.py              # Unit tests for general classifier
└── READ ME.txt                                 # Original readme notes
```

## Architecture

The application follows an object-oriented design with a base `Classifier` class and specialized subclasses:

- **Classifier (Base Class)**: Defines the common classification logic
- **GeneralClassifier**: Implements general document classification
- **ConfidentialClassifier**: Implements confidential document classification

Each classifier loads its respective pre-trained model, TF-IDF vectorizer, and category mapping from pickle files.

## Testing

Run the unit tests using Python's unittest framework:

```bash
python -m unittest Tests.test_generalClassifier
```

Example test case:
```python
from classifier import GeneralClassifier

clf = GeneralClassifier()
category = clf.classify('Sri Lanka cricket team won the 1996 world championship')
# Expected output: "sport"
```

## Model Retraining

To retrain the models with a new dataset:

1. Use the Jupyter notebook `Classifer_General_Classifier_notebook.ipynb` to train your model
2. Export the trained model, TF-IDF vectorizer, and ID mapping as pickle files
3. Replace the existing pickle files with your newly trained ones:
   - `gen_clf.pickle`
   - `gen_tfidf.pickle`
   - `gen_id_map.pickle`
   - (Or the corresponding `conf_*` files for confidential classifier)
4. Restart the Flask server

## API Response Format

All classification endpoints return JSON arrays with objects containing:
- `path`: The original document path/identifier
- `category`: The predicted category label

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source. Please check with the repository owner for specific licensing terms.

## Notes

- The server supports batch classification (multiple documents in a single request)
- CORS is enabled, allowing requests from any origin
- The classification models use TF-IDF (Term Frequency-Inverse Document Frequency) for text feature extraction
