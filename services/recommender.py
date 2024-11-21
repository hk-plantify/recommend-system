import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from models.embedding_model import embedding_model

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from models.embedding_model import embedding_model

card_data = pd.read_csv('.tmp/embedding_card_category_data.csv')
card_category = pd.read_csv('.tmp/preprocessed_card_data.csv')
embeddings_matrix = np.load('.tmp/embeddings_matrix.npy')

categories = card_category['benefit_category'].unique()
category_embeddings = embedding_model.encode(categories)
category_to_embedding = dict(zip(categories, category_embeddings))

def recommend_cards_by_category(category: str, top_n: int = 5):
    """
    Args:
        category (str): 추천받고자 하는 카테고리
        top_n (int): 추천할 카드의 수

    Returns:
        tuple: top_card (추천 카드 1개), other_recommendations (나머지 추천 카드)
    """
    category_embedding = category_to_embedding.get(category)
    if category_embedding is None:
        raise ValueError(f"Category '{category}' not found in category embeddings.")
    
    similarities = cosine_similarity([category_embedding], embeddings_matrix)[0]
    
    top_indices = similarities.argsort()[-top_n:][::-1]

    top_card = card_data.iloc[top_indices[0]]
    other_recommendations = card_data.iloc[top_indices[1:top_n]]
    
    return top_card, other_recommendations

def get_available_categories():
    """
    사용 가능한 모든 카테고리를 반환하는 함수.
    
    Returns:
        list: 고유 카테고리 목록
    """
    return list(categories)
