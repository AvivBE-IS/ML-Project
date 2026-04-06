import pandas as pd
import os
import re
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================================
# Configuration & Constants
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the script location
TEST_FOLDER_PATH = os.path.join(SCRIPT_DIR, 'test')
OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'classification.csv')
TRAIN_FEATS_FILE = os.path.join(SCRIPT_DIR, 'features_data.csv')
SENSED_DATA_FILE = os.path.join(SCRIPT_DIR, 'sensed_data.csv')

# Base URL for The Guardian (required for reconstructing the full link)
BASE_URL = "https://www.theguardian.com/"

# ==========================================
# Text Cleaning Function
# ==========================================
def clean_text(text):
    """
    Applies the same preprocessing used during training.
    """
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(text.split())
    return text

# ==========================================
# Model Loading / Retraining Logic
# ==========================================
def train_or_load_model():
    """
    Attempts to load artifacts from .pkl files.
    If files are missing, it retrains utilizing the best hyperparameters.
    """
    # 1. Load or Re-fit Vectorizer
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        print("Status: Loaded vectorizer from pickle file.")
    except FileNotFoundError:
        print("Status: Vectorizer pickle not found. Retraining vectorizer...")
        
        if os.path.exists(SENSED_DATA_FILE) and os.path.exists(TRAIN_FEATS_FILE):
            feats_df = pd.read_csv(TRAIN_FEATS_FILE)
            vocab = feats_df.drop('label', axis=1).columns.tolist()
            
            df = pd.read_csv(SENSED_DATA_FILE)
            df['text'] = df['title'].apply(clean_text) + " " + df['body'].apply(clean_text)
            
            vectorizer = TfidfVectorizer(vocabulary=vocab, stop_words='english')
            vectorizer.fit(df['text'])
        else:
            raise FileNotFoundError(f"Critical error: CSV files missing in {SCRIPT_DIR}.")

    # 2. Load or Retrain Model
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Status: Loaded model from pickle file.")
    except FileNotFoundError:
        print("Status: Model pickle not found. Retraining model...")
        
        if os.path.exists(TRAIN_FEATS_FILE):
            data = pd.read_csv(TRAIN_FEATS_FILE)
            X_train = data.drop('label', axis=1)
            y_train = data['label']
            
            # Using your BEST hyperparameters found in the notebook
            model = MLPClassifier(
                hidden_layer_sizes=(50,),
                activation='tanh',
                learning_rate_init=0.01,
                batch_size=64,
                max_iter=100,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            print("Status: Model retraining complete.")
        else:
            raise FileNotFoundError(f"Critical error: '{TRAIN_FEATS_FILE}' missing.")
            
    return model, vectorizer

# ==========================================
# Main Execution
# ==========================================
def main():
    model, vectorizer = train_or_load_model()
    
    results = []
    
    if not os.path.exists(TEST_FOLDER_PATH):
        print(f"Warning: Folder '{TEST_FOLDER_PATH}' does not exist.")
        return

    print(f"Scanning folder: {TEST_FOLDER_PATH} for .txt files...")
    
    files_found = False
    for filename in os.listdir(TEST_FOLDER_PATH):
        if filename.endswith(".txt"):
            files_found = True
            file_path = os.path.join(TEST_FOLDER_PATH, filename)
            
            try:
                # ---------------------------------------------------------
                # NEW LOGIC: Reconstruct URL from Filename
                # ---------------------------------------------------------
                # 1. Remove extension
                clean_name = filename.replace(".txt", "")
                # 2. Replace underscores with slashes to restore path structure
                url_suffix = clean_name.replace("_", "/")
                # 3. Create full URL
                full_url = f"{BASE_URL}{url_suffix}"

                # ---------------------------------------------------------
                # Read content for classification
                # ---------------------------------------------------------
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Preprocess
                cleaned_content = clean_text(content)
                features_sparse = vectorizer.transform([cleaned_content])
                features_df = pd.DataFrame(features_sparse.toarray(), columns=vectorizer.get_feature_names_out())
                
                # Predict
                prediction = model.predict(features_df)[0]
                
                # Append result using the RECONSTRUCTED URL
                results.append({'URL': full_url, 'Category': prediction})
                
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    # Step 3: Save Results to CSV
    if files_found:
        output_df = pd.DataFrame(results)
        output_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Success: Classification results saved to {OUTPUT_FILE}")
    else:
        print("Note: No .txt files found in Test folder.")

if __name__ == "__main__":
    main()