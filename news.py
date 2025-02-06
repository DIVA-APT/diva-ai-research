import sys
import os
import json

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# API KEY 설정
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# OpenAI API 초기화
client = OpenAI(api_key=OPENAI_API_KEY)

# Pinecone 초기화
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "news-index"
index = pc.Index(index_name)

# SentenceTransformer 모델 (KoE5 사용)
model = SentenceTransformer("nlpai-lab/KoE5")


def generate_questions(stock_name):
    """
    주식 관련 질문 생성.
    """
    # 기본 질문 생성
    base_question = f"{stock_name}의 최근 주식 시장 동향은 어떠한가?"

    # 추가 질문 생성 (LLM 활용)
    prompt = (
        f"다음 종목 '{stock_name}'에 대한 주식 시장 전망과 관련된 3가지 구체적인 질문을 생성해 주세요. "
        f"각 질문은 투자자들이 관심을 가질 만한 중요한 뉴스 이슈를 다루어야 하며, 한국어로 작성해 주세요."
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system",
             "content": "당신은 최신 주식 뉴스를 분석하여 투자자들에게 유용한 정보를 제공하는 금융 뉴스 전문가입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )

    # 생성된 질문 리스트 반환
    questions = [base_question] + response.choices[0].message.content.strip().split('\n')
    return questions


def search_articles(questions):
    """
    Pinecone에서 각 질문에 대한 관련 뉴스를 검색.
    """
    results = {}
    for question in questions:
        query_vector = model.encode(question).tolist()
        response = index.query(vector=query_vector, top_k=10, include_metadata=True)
        results[question] = response['matches']
    return results


def extract_all_chunks(articles_data):
    """
    검색된 뉴스 데이터에서 텍스트 추출.
    """
    all_chunks = []
    for question, documents in articles_data.items():
        for document in documents:
            if 'metadata' in document and 'chunk' in document['metadata']:
                all_chunks.append(document['metadata']['chunk'])
    return all_chunks


def analyze_news(stock_name):
    """
    뉴스 데이터를 기반으로 종합적인 분석 수행.
    """
    # 질문 생성 및 뉴스 검색
    questions = generate_questions(stock_name)
    articles = search_articles(questions)

    # 뉴스 텍스트 추출 및 컨텍스트 구성
    chunk_list = extract_all_chunks(articles)
    context = " ".join(chunk_list)

    # RAG 프롬프트 생성
    rag_prompt = (
        f"다음은 '{stock_name}'에 대한 최신 뉴스 데이터입니다:\n\n"
        f"{context}\n\n"
        f"위 데이터를 바탕으로 다음 질문에 답변해 주세요:\n"
        f"1. {stock_name}과 관련된 최근 주요 이슈는 무엇인가요?\n"
        f"2. 해당 이슈가 주식 시장과 투자자들에게 미치는 영향은 무엇인가요?\n"
        f"3. {stock_name}의 미래 전망과 투자 전략(예: 매수, 보유, 매도)에 대한 권장 사항을 제시해 주세요.\n\n"
        f"결과는 간결하지만 명확하게 작성하며, 모든 데이터는 신뢰할 수 있는 출처를 기반으로 작성해 주세요."
    )

    # OpenAI GPT-4 호출
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system",
             "content": "당신은 최신 뉴스를 종합적으로 분석하여 투자자들에게 유용한 정보를 제공하는 금융 전문가입니다."},
            {"role": "user", "content": rag_prompt}
        ],
        temperature=0.1
    )

    # 결과 출력 및 반환
    result = response.choices[0].message.content
    print("\nGenerated Result:\n")
    print(result)


if __name__ == "__main__":
    # 테스트용 종목 이름 설정
    stock_name = sys.argv[1]

    #stock_name = '삼성전자'

    # 뉴스 분석 실행
    analyze_news(stock_name)
