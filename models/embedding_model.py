import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# 전역 임베딩 모델 로드
embedding_model = SentenceTransformer("BAAI/bge-m3")

def load_data(preprocessed_csv, embeddings_file):
    """CSV 데이터와 임베딩 행렬 로드"""
    card_data = pd.read_csv(preprocessed_csv)
    embeddings_matrix = np.load(embeddings_file)
    return card_data, embeddings_matrix
