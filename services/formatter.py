import os
import boto3
from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.chat_models.openai import ChatOpenAI  # 업데이트된 import
import json

# AWS S3 설정
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
    # S3에서 .env 파일 다운로드
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    env_content = response['Body'].read().decode('utf-8')  # Bytes 데이터를 문자열로 변환

    # .env 파일을 로컬에 저장
    with open(local_env_path, "w") as env_file:
        env_file.write(env_content)

    # .env 파일 로드하여 환경 변수로 설정
    load_dotenv(local_env_path)

# .env 파일 로드
load_env_from_s3(S3_BUCKET, "env-files/.env")

# OPENAI API 키 로드
openai_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    max_tokens=512
)

def extract_and_format_benefits_with_llm_batch(cards_info, category):
    # 시스템 메시지
    system_prompt = """
    당신은 카드 정보를 분석하는 전문가입니다.
    제공된 카드 정보에서 해당 카테고리와 관련된 핵심 혜택을 추출하세요.
    """

    # 카드 정보를 간결히 표현
    cards_prompt = "\n".join([
        f"카드 {i + 1}: 혜택: {card['title']} | 카테고리: {category}"
        for i, card in enumerate(cards_info)
    ])

    # 프롬프트 작성
    prompt = f"""
    카테고리: {category}
    추가적인 텍스트, 포맷(예: 마크다운), 설명은 포함하지 마십시오. 순수한 JSON 데이터만 반환하세요.
    카드 정보:
    {cards_prompt}

    응답 형식:
    [
        {{
            "discount_target": "string",
            "discount_type": "string",
            "benefit_point": "int"
        }},
        ...
    ]

    **요구 사항**:
    1. 각 카드에 대해 명시된 JSON 구조를 따르는 객체를 반환하세요.
    2. `discount_target`: 해당 카테고리와 관련된 할인 대상을 입력하세요.
    3. `discount_type`: `discount_target`과 관련된 할인 내역을 입력하세요.
    4. `benefit_point`: 소비 금액 10,000원을 기준으로 계산된 혜택 금액을 입력하세요.
    """

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt)])

    try:
        benefits = json.loads(response.content)

        formatted_results = []
        for card, benefit in zip(cards_info, benefits):
            formatted_results.append({
                "card_name": card["name"],
                "card_image": card["image"],
                **benefit
            })

        return formatted_results
    except json.JSONDecodeError:
        return [{"error": "Failed to parse JSON response", "raw_response": response.content}]
