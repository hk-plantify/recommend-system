import os
import boto3
import numpy as np
import pandas as pd
from io import BytesIO, StringIO
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.metrics.pairwise import cosine_similarity
from services.formatter import extract_and_format_benefits_with_llm_batch

S3_BUCKET = "hk-project-6-bucket"

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name="ap-northeast-1"
)

def load_env_from_s3(bucket_name, key, local_env_path=".env"):
    """
    S3에서 .env 파일을 다운로드하고 환경 변수로 설정
    """
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    env_content = response['Body'].read().decode('utf-8')  # Bytes 데이터를 문자열로 변환

    with open(local_env_path, "w") as env_file:
        env_file.write(env_content)

    load_dotenv(local_env_path)

# .env 파일 로드
load_env_from_s3(S3_BUCKET, "env-files/.env")

# S3 파일 다운로드 함수
def download_from_s3(bucket_name, key):
    """
    S3에서 데이터를 다운로드하여 BytesIO 객체로 반환
    """
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    return BytesIO(response['Body'].read())

# S3에서 데이터 로드
card_data_key = "embedding_card_data.csv"
description_matrix_key = "description_matrix.npy"
category_embeddings_key = "category_embeddings.npy"

# CSV 데이터 로드
card_data_bytes = download_from_s3(S3_BUCKET, card_data_key)
card_data = pd.read_csv(StringIO(card_data_bytes.read().decode('utf-8')))

# Numpy 데이터 로드
embeddings_matrix_bytes = download_from_s3(S3_BUCKET, description_matrix_key)
embeddings_matrix_bytes.seek(0)
embeddings_matrix = np.load(embeddings_matrix_bytes)

# 카테고리 임베딩 로드
category_embeddings_bytes = download_from_s3(S3_BUCKET, category_embeddings_key)
category_embeddings_bytes.seek(0)
category_embeddings = np.load(category_embeddings_bytes)

# 카테고리와 임베딩 매핑
categories = card_data['category'].unique()
category_to_embedding = dict(zip(categories, category_embeddings))

# 카드 추천 함수
def recommend_cards_by_category(category: str, top_n: int = 5):
    if category not in category_to_embedding:
        available_categories = ', '.join(get_available_categories())
        raise ValueError(f"Category '{category}' not found. Available categories: {available_categories}")
    
    category_embedding = category_to_embedding[category]
    similarities = cosine_similarity([category_embedding], embeddings_matrix)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]

    # 가장 유사한 카드와 추가 추천 카드 선택
    top_card = card_data.iloc[top_indices[0]].to_dict()  # Series -> dict 변환
    other_recommendations = card_data.iloc[top_indices[1:top_n]].to_dict(orient="records")  # DataFrame -> list of dicts
    
    # 혜택을 포맷팅
    formatted_result = extract_and_format_benefits_with_llm_batch([top_card] + other_recommendations, category)

    return {
        "top_card": formatted_result[0],  # 가장 유사한 카드
        "other_cards": formatted_result[1:]  # 추가 추천 카드
    }

# 사용 가능한 카테고리 반환
def get_available_categories():
    """
    Returns:
        list: 고유 카테고리 목록
    """
    return list(categories)