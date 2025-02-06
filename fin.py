import sys
import os
import json

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# 환경 변수 로드
load_dotenv()

# API 키 설정
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

## Pinecone 초기화
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "fin-index"
index = pc.Index(index_name)

# OpenAI API 초기화
client = OpenAI(api_key=OPENAI_API_KEY)

# SentenceTransformer 모델 (KoE5 사용)
model = SentenceTransformer("nlpai-lab/KoE5")


def analyze_financial(stock_name):

    # 질문 생성
    question = (
        f"{stock_name}의 최근 재무제표 데이터를 바탕으로 "
        f"부채율, 총자산 대비 기업 가치, 자산 대비 매출 비율을 분석하고, "
        f"이 기업의 재무 안정성과 성장 가능성을 평가해 주세요."
    )

    # 질문을 벡터로 변환
    query_vector = model.encode(question).tolist()

    # Pinecone에서 벡터 검색 (top_k=100)
    response = index.query(vector=query_vector, top_k=100, include_metadata=True)

    # 검색 결과에서 컨텍스트 생성
    chunks = [matches['metadata']['raw_data'] for matches in response['matches']]
    context = " ".join(chunks)

    # RAG 프롬프트 생성
    rag_prompt = (
        f"{context}\n\n"
        f"이 문서는 전자공시시스템 DART에서 제공된 데이터를 바탕으로 작성되었습니다. "
        f"모든 수치는 공식 데이터를 사용하며, {context}에 없는 정보의 경우 인터넷 검색을 활용하되 가정이나 추정, 예시를 사용하지 않도록 해주세요.\n\n"
        f"다음 질문에 답변해 주세요:\n\n"
        f"1. {stock_name}의 부채율, 총자산 대비 기업 가치, 자산 대비 매출 비율을 분석해 주세요.\n"
        f"2. 해당 지표를 기반으로 이 기업의 재무 안정성과 성장 가능성을 평가해 주세요.\n"
        f"3. 투자자 관점에서 유용한 결론과 권장 사항(예: 매수, 보유, 매도)을 제시해 주세요.\n\n"
        f"결과는 간결하지만 명확하게 작성하며, 모든 데이터는 신뢰할 수 있는 출처를 기반으로 작성해 주세요."
    )

    # OpenAI GPT-4 호출
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",
             "content": "당신은 개인 투자자들이 특정 종목에 대한 재무 데이터를 이해하고 투자 결정을 내릴 수 있도록 돕는 금융 분석 전문가입니다."},
            {"role": "user", "content": rag_prompt}
        ],
        temperature=0.1
    )

    # 결과 출력 및 반환
    result = response.choices[0].message.content
    print("\nGenerated Result:\n")
    print(result)

    json_output = json.dumps(
        {
            "result_fin": result,
            "reference": {
                'title': '전자공시시스템',
                'description': '(DART : Data Analysis, Retrieval and Transfer System)의 OpenDART 재무정보조회 데이터에 근거한 분석 결과입니다. ',
                'url': 'https://dart.fss.or.kr/'
            }
        },
        ensure_ascii=False, indent=4)

    print(json_output)


if __name__ == "__main__":
    stock_name = sys.argv[1]
    # stock_name = '삼성전자'
    analyze_financial(stock_name)