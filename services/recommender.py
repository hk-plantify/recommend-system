import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from models.embedding_model import embedding_model
from services.formatter import extract_and_format_benefits_with_llm_batch

card_data = pd.read_csv('.tmp/embedding_card_data.csv')
embeddings_matrix = np.load('.tmp/description_matrix.npy')
category_embeddings_path = '.tmp/category_embeddings.npy'

if not os.path.exists(category_embeddings_path):
    categories = card_data['category'].unique()
    category_embeddings = embedding_model.encode(categories, batch_size=32)
    np.save(category_embeddings_path, category_embeddings)
else:
    categories = card_data['category'].unique()
    category_embeddings = np.load(category_embeddings_path)

category_to_embedding = dict(zip(categories, category_embeddings))

def recommend_cards_by_category(category: str, top_n: int = 5):
    if category not in category_to_embedding:
        available_categories = ', '.join(get_available_categories())
        raise ValueError(f"Category '{category}' not found. Available categories: {available_categories}")
    
    category_embedding = category_to_embedding[category]
    similarities = cosine_similarity([category_embedding], embeddings_matrix)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]

    top_card = card_data.iloc[top_indices[0]].to_dict()  # Series -> dict 변환
    other_recommendations = card_data.iloc[top_indices[1:top_n]].to_dict(orient="records")  # DataFrame -> list of dicts
    
    formatted_result = extract_and_format_benefits_with_llm_batch([top_card] + other_recommendations, category)

    return {
        "top_card": formatted_result[0],  # 가장 유사한 카드
        "other_cards": formatted_result[1:]  # 추가 추천 카드
    }
    # return top_card, other_recommendations

def get_available_categories():
    """
    Returns:
        list: 고유 카테고리 목록
    """
    return list(categories)
