"""
STEP 2b: Train Custom Sentiment Classifier (YOUR CONTRIBUTION!)
================================================================
This is THE CORE of your research contribution.

Pipeline Comparison:
1. Sparse: TF-IDF + Multinomial NB
2. Dense:  TF-IDF + SVD + SVM
3. Topic:  TF-IDF + LDA + XGBoost
4. Embedding: Word2Vec + XGBoost

老师要求：
"You must train your own sentiment model, not just use pre-trained FinBERT.
Your contribution is in this pipeline and hyperparameter tuning."
"""

import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import joblib
import warnings

warnings.filterwarnings('ignore')

print("="*70)
print("🧠 STEP 2b: TRAINING CUSTOM SENTIMENT CLASSIFIER")
print("="*70 + "\n")

TICKERS = ['XLK', 'XLF']

def clean_text(text):
    """Clean text for NLP"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def load_labeled_data(ticker):
    """Load the labeled data from Step 2a"""
    path = f"data/sentiment_labels_{ticker}.csv"
    if not os.path.exists(path):
        print(f"   ❌ Labels not found: {path}")
        print("      Run step2a_create_sentiment_labels.py first!")
        return None, None
    
    df = pd.read_csv(path)
    
    # Find text column
    text_col = next((c for c in df.columns if c not in ['Sentiment_Label']), None)
    
    if text_col is None:
        print(f"   ❌ No text column found")
        return None, None
    
    X = df[text_col].apply(clean_text)
    y = df['Sentiment_Label']
    
    return X, y

def train_pipeline_sparse(X_train, X_test, y_train, y_test):
    """
    Pipeline A: Sparse Matrix with Multinomial NB
    老师推荐：适合sparse matrix (TF-IDF)
    """
    print("   1️⃣  Training: TF-IDF + Multinomial NB (Sparse)")
    
    param_grid = {
        'tfidf__max_features': [1000, 3000, 5000],
        'tfidf__ngram_range': [(1,1), (1,2)],
        'clf__alpha': [0.1, 1.0, 10.0]
    }
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', MultinomialNB())
    ])
    
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"      ✅ Best Params: {best_params}")
    print(f"      ✅ Test Accuracy: {acc:.2%}")
    
    return best_model, acc, 'Sparse_NB'

def train_pipeline_dense(X_train, X_test, y_train, y_test):
    """
    Pipeline B: Dense Matrix with SVD + SVM
    Topic Modeling + Classification
    """
    print("   2️⃣  Training: TF-IDF + SVD + SVM (Dense)")
    
    param_grid = {
        'tfidf__max_features': [3000, 5000],
        'svd__n_components': [50, 100],
        'clf__C': [0.1, 1.0, 10.0],
        'clf__kernel': ['rbf', 'linear']
    }
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('svd', TruncatedSVD(random_state=42)),
        ('clf', SVC(probability=True, random_state=42))
    ])
    
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"      ✅ Best Params: {best_params}")
    print(f"      ✅ Test Accuracy: {acc:.2%}")
    
    return best_model, acc, 'Dense_SVM'

def train_pipeline_lda(X_train, X_test, y_train, y_test):
    """
    Pipeline C: LDA Topic Modeling + XGBoost
    Alternative to SVD
    """
    print("   3️⃣  Training: TF-IDF + LDA + XGBoost")
    
    param_grid = {
        'tfidf__max_features': [3000],
        'lda__n_components': [30, 50],
        'clf__max_depth': [3, 5],
        'clf__learning_rate': [0.1, 0.3]
    }
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('lda', LatentDirichletAllocation(random_state=42, max_iter=10)),
        ('clf', xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, n_jobs=-1))
    ])
    
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"      ✅ Best Params: {best_params}")
    print(f"      ✅ Test Accuracy: {acc:.2%}")
    
    return best_model, acc, 'LDA_XGB'

# ---------------- 新增 Word2Vec 组件 ----------------

class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    """自定义转换器，将 Word2Vec 嵌入到 sklearn Pipeline 中"""
    def __init__(self, vector_size=100, window=5, min_count=2):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.w2v_model = None
        
    def fit(self, X, y=None):
        sentences = [str(text).split() for text in X]
        self.w2v_model = Word2Vec(sentences, vector_size=self.vector_size, 
                                  window=self.window, min_count=self.min_count, workers=4)
        return self
        
    def transform(self, X):
        sentences = [str(text).split() for text in X]
        X_embedded = np.array([
            np.mean([self.w2v_model.wv[w] for w in words if w in self.w2v_model.wv]
                    or [np.zeros(self.vector_size)], axis=0)
            for words in sentences
        ])
        return X_embedded

def train_pipeline_word2vec(X_train, X_test, y_train, y_test):
    """
    Pipeline D: Word Embedding (Word2Vec) + XGBoost
    """
    print("   4️⃣  Training: Word2Vec + XGBoost (Embedding)")
    
    param_grid = {
        'w2v__vector_size': [50, 100],
        'clf__max_depth': [3, 5],
        'clf__learning_rate': [0.1, 0.2]
    }
    
    pipeline = Pipeline([
        ('w2v', Word2VecVectorizer()),
        ('clf', xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, n_jobs=-1))
    ])
    
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"      ✅ Best Params: {best_params}")
    print(f"      ✅ Test Accuracy: {acc:.2%}")
    
    return best_model, acc, 'Word2Vec_XGB'

# ----------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, ticker, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title(f'Confusion Matrix: {ticker} ({model_name})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if not os.path.exists('results'): os.makedirs('results')
    plt.savefig(f'results/confusion_matrix_{ticker}_{model_name}.png')
    plt.close()

def train_all_pipelines(ticker):
    print(f"\n{'='*60}")
    print(f"🎯 TRAINING SENTIMENT CLASSIFIERS: {ticker}")
    print(f"{'='*60}")
    
    # Load data
    X, y = load_labeled_data(ticker)
    
    if X is None:
        return None
    
    print(f"   📊 Data: {len(X)} samples")
    print(f"   📋 Classes: {y.nunique()} (Negative/Neutral/Positive)")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   📊 Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train all pipelines
    results = []
    
    try:
        model1, acc1, name1 = train_pipeline_sparse(X_train, X_test, y_train, y_test)
        results.append({'Model': name1, 'Accuracy': acc1, 'Pipeline': model1})
    except Exception as e:
        print(f"      ❌ Sparse pipeline failed: {e}")
    
    try:
        model2, acc2, name2 = train_pipeline_dense(X_train, X_test, y_train, y_test)
        results.append({'Model': name2, 'Accuracy': acc2, 'Pipeline': model2})
    except Exception as e:
        print(f"      ❌ Dense pipeline failed: {e}")
    
    try:
        model3, acc3, name3 = train_pipeline_lda(X_train, X_test, y_train, y_test)
        results.append({'Model': name3, 'Accuracy': acc3, 'Pipeline': model3})
    except Exception as e:
        print(f"      ❌ LDA pipeline failed: {e}")
        
    try:
        model4, acc4, name4 = train_pipeline_word2vec(X_train, X_test, y_train, y_test)
        results.append({'Model': name4, 'Accuracy': acc4, 'Pipeline': model4})
    except Exception as e:
        print(f"      ❌ Word2Vec pipeline failed: {e}")
    
    # Select best model
    if not results:
        print("   ❌ All pipelines failed!")
        return None
    
    best = max(results, key=lambda x: x['Accuracy'])
    
    print(f"\n   🏆 WINNER: {best['Model']} ({best['Accuracy']:.2%})")
    
    # Save best model
    model_path = f"models/sentiment_classifier_{ticker}.pkl"
    if not os.path.exists('models'): os.makedirs('models')
    
    joblib.dump(best['Pipeline'], model_path)
    print(f"   💾 Saved model: {model_path}")
    
    # Plot confusion matrix
    y_pred = best['Pipeline'].predict(X_test)
    plot_confusion_matrix(y_test, y_pred, ticker, best['Model'])
    
    # Classification report
    print(f"\n   📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))
    
    return {'Ticker': ticker, 'Best_Model': best['Model'], 'Accuracy': best['Accuracy']}

def main():
    print("\n⭐ YOUR RESEARCH CONTRIBUTION:")
    print("   This is where you TRAIN YOUR OWN sentiment classifier!")
    print("   Not using pre-trained FinBERT - building from scratch.")
    print("="*70 + "\n")
    
    summary = []
    
    for ticker in TICKERS:
        try:
            result = train_all_pipelines(ticker)
            if result:
                summary.append(result)
        except Exception as e:
            print(f"\n❌ Error training {ticker}: {e}")
            import traceback
            traceback.print_exc()
    
    if summary:
        print("\n" + "="*70)
        print("📊 TRAINING SUMMARY")
        print("="*70)
        df_summary = pd.DataFrame(summary)
        print(df_summary.to_string(index=False))
        
        df_summary.to_csv('results/sentiment_model_comparison.csv', index=False)
        print("\n✅ Saved: results/sentiment_model_comparison.csv")
    
    print("\n" + "="*70)
    print("✅ SENTIMENT CLASSIFIER TRAINING COMPLETE")
    print("="*70)
    print("\n📌 Next Steps:")
    print("   1. Run step2c_generate_sentiment_features.py")
    print("   2. Use these features in XGBoost/LSTM models")
    print("\n💡 For Your Report:")
    print("   'We trained our own sentiment classifier using [best_model],")
    print("    achieving XX% accuracy on test data.'")

if __name__ == "__main__":
    main()