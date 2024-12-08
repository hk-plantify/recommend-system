import os
import boto3
import numpy as np
import pandas as pd
from io import BytesIO, StringIO
from dotenv import dotenv_values
from sqlalchemy import create_engine
from sklearn.metrics.pairwise import cosine_similarity
from models.embedding_model import embedding_model
from services.formatter import extract_and_format_benefits_with_llm_batch

# AWS S3 설정
S3_BUCKET = "hk-project-6-bucket"
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name="ap-northeast-1"
)

def load_env_from_s3(bucket_name, key):
    """
    S3에서 .env 파일을 다운로드하고 환경 변수로 설정
    """
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    env_content = response['Body'].read().decode('utf-8')  # Bytes 데이터를 문자열로 변환

    # .env 파일 내용 파싱
    env_vars = dotenv_values(stream=StringIO(env_content))  # StringIO로 래핑하여 dotenv_values로 파싱
    for key, value in env_vars.items():
        os.environ[key] = value  # 환경 변수로 설정

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

# 카테고리 임베딩 확인 및 로드
try:
    category_embeddings_bytes = download_from_s3(S3_BUCKET, category_embeddings_key)
    category_embeddings_bytes.seek(0)
    category_embeddings = np.load(category_embeddings_bytes)
except s3_client.exceptions.NoSuchKey:
    # S3에서 파일이 없으면 새로 생성
    categories = card_data['category'].unique()
    category_embeddings = embedding_model.encode(categories, batch_size=32)

    # 생성된 데이터를 S3에 업로드
    buffer = BytesIO()
    np.save(buffer, category_embeddings)
    buffer.seek(0)
    s3_client.put_object(Bucket=S3_BUCKET, Key=category_embeddings_key, Body=buffer.getvalue())

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