import json
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.chat_models.openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)


def extract_and_format_benefits_with_llm_batch(cards_info, category):
    """
    여러 카드 정보를 한 번에 처리하는 LLM 호출 함수.
    """
    system_prompt = """
    당신은 카드 정보를 엄격한 JSON 형식으로 작성하는 전문가입니다. 
    모든 응답은 제공된 JSON 구조를 엄격히 준수해야 하며, 추가 텍스트나 포맷을 포함하지 마십시오.
    """

    # 각 카드 정보를 프롬프트에 추가
    cards_prompt = "\n".join([
        f"""
        카드 정보 {i + 1}:
        - Card Name: {card['name']}
        - Benefit Description: {card['combined_benefits']}
        - Input Category: {category}
        - Assume a spending amount of 10,000 KRW for calculating remaining benefit.
        """
        for i, card in enumerate(cards_info)
    ])

    prompt = f"""
    아래 카드 정보들을 각각 엄격한 JSON 형식으로 변환하세요:
    ### JSON Structure
    {{
        "card_name": "string",
        "discount_target": "string",
        "discount_type": "string",
        "remaining_benefit": "string"
    }}

    **요구 사항**:
    1. `discount_target`: 카드의 혜택 설명 중 해당 카테고리와 관련된 할인 대상(예: "대중교통", "택시")만 포함하세요.
    2. `discount_type`: `discount_target`과 관련된 할인 내역을 입력하세요(예: "10% 할인", "5% 캐시백").
    3. `remaining_benefit`: 소비 금액 10,000원 기준으로 계산된 할인 금액을 입력하세요. (예: 10% 할인 -> 1,000원, 5% 캐시백 -> 500원).
    4. 다른 카테고리의 정보는 무시하고, 제공된 `Input Category`에 따라 가장 관련 있는 혜택만 추출하세요.

    {cards_prompt}
    """

    # LLM 호출
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt)])
    
    # JSON 응답 파싱
    try:
        # LLM이 반환한 응답이 여러 JSON 객체일 경우 배열로 반환되도록 설계
        responses = json.loads(response.content)
        return responses
    except json.JSONDecodeError:
        return [{"error": "Failed to parse JSON response", "raw_response": response.content}]
