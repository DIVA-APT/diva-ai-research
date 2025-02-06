import sys
import os
import json
import re
import hashlib

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

## API KEY
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

## OpenAI API 호출
client = OpenAI(
    api_key=OPENAI_API_KEY,  # This is the default and can be omitted
)

## Pinecone 초기화
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "analyst-index"
index = pc.Index(index_name)

## doc2vec (KoE5 사용)
model = SentenceTransformer("nlpai-lab/KoE5")

## 산업분류 리스트
standard_industrial_classifications = [
    "IT", "통신", "반도체", "디스플레이",
    "전기전자", "철강금속", "섬유화학", "섬유의류",
    "조선", "해운", "건설", "유통", "항공운송", "자동차",
    "에너지", "제약", "바이오", "화학",
    "타이어", "소비재", "음식료", "여행", "유틸리티",
    "게임", "미디어", "인터넷포탈",  "지주회사",
    "금융", "보험", "은행", "증권",
    "휴대폰", "화장품", "서비스", "기타"
]


## 주요 사업 분야 추출(상위 3개 또는 1개 선택)
def get_industries(stock_name):
    prompt = f"""
        "{stock_name}"의 주요 사업 분야를 다음 리스트 내에서만 선택해 나열해 주세요:
        {', '.join(standard_industrial_classifications)}.
    
        가능한 분야를 1개 이상 정확히 골라서 반환해 주세요.
        예를 들어, {stock_name}이 반도체와 통신 산업과 관련이 있다면 "반도체, 통신"과 같은 형식으로 답변해 주세요.
        만약 {stock_name}이 하나의 산업과 관련이 있다면 하나의 산업만 선택해 답변해 주세요.
        """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.1
    )

    # 응답 내용 정리
    industry_list = response.choices[0].message.content.strip().split(",")
    industries = [industry.strip() for industry in industry_list if industry.strip()]  # 공백 제거

    # 상위 3개 또는 최소 1개 선택
    return industries[:3] if len(industries) > 1 else industries


# vectorDB 문서 검색
def filter_by_stock_and_industries(stock_name, industries):
    # 종목명(stock_name) 관련 문서 검색
    stock_question = f"{stock_name}와 관련된 핵심 트렌드, {stock_name}의 경쟁력 평가, {stock_name}과 관련된 긍정적/부정적 의견과 그 이유"
    stock_query_vector = model.encode(stock_question).tolist()
    stock_response = index.query(vector=stock_query_vector, top_k=10, include_metadata=True, namespace="example-namespace1")

    # 산업(industry) 관련 문서 검색
    industry_question = f"{industries} 산업의 주요 이슈가 {stock_name}에 미치는 영향, {industries} 산업의 변화가 {stock_name}의 주가 또는 시장에서의 위치에 미치는 영향"
    industry_query_vector = model.encode(industry_question).tolist()
    industry_response = index.query(vector=industry_query_vector, top_k=10, include_metadata=True, namespace="example-namespace1")

    # print("Results (stock):", stock_response)
    # print("Results (industries):", industry_response)

    return stock_response["matches"], industry_response["matches"]


## 동일 문서 처리
def remove_duplicate_documents(documents):
    unique_docs = {}
    for doc in documents:
        # 문서 고유 식별자 생성
        doc_id = (
        doc['metadata']['title'], doc['metadata']['firm'], doc['metadata']['date'], doc['metadata']['category'],
        doc['metadata']['source_url'])

        # unique_docs에 동일한 식별자의 문서가 없으면 추가
        if doc_id not in unique_docs:
            unique_docs[doc_id] = {
                "title": doc['metadata']['title'],
                "firm": doc['metadata']['firm'],
                "category": doc['metadata']['category'],
                "date": doc['metadata']['date'],
                "source_url": doc['metadata']['source_url']
            }

    return list(unique_docs.values())


## 전문가 의견 도출을 위한 프롬프트 생성
def create_prompt(stock_name, industries, docs_stock, docs_industry):
    # Extract and prepare text and metadata
    docs_stock_context = " ".join(doc['metadata']['chunk'] for doc in docs_stock)
    docs_industry_context = " ".join(doc['metadata']['chunk'] for doc in docs_industry)

    references_used = remove_duplicate_documents(docs_stock + docs_industry)

    # 프롬프트 생성
    prompt = f"""
            아래는 "{stock_name}"와 관련된 문서입니다:

            ### {stock_name} 관련 문서 ###
            {docs_stock_context}

            위 문서들을 분석하여 다음 질문에 답변해 주세요:
            1. "{stock_name}"와 관련된 핵심 트렌드는 무엇인가요?
            2. "{stock_name}"의 경쟁력은 어떻게 평가되나요?
            3. 관련된 긍정적/부정적 의견과 그 이유는 무엇인가요?

            ### {', '.join(industries)} 산업 관련 문서 ###
            {docs_industry_context}

            위 문서들을 분석하여 다음 질문에 답변해 주세요:
            1. "{', '.join(industries)}" 산업의 주요 이슈가 "{stock_name}"에 미치는 영향은 무엇인가요?
            2. 해당 산업의 변화가 "{stock_name}"의 주가 또는 시장에서의 위치에 어떤 영향을 줄 수 있나요?

            - **최종 요약**: (위 문서들을 분석하여 얻은 답변과 위에 모든 문서들로부터 알 수 있는 전문가 의견을 종합하여 한글 기준 약 2000자 분량으로 상세히 작성해 주세요.)
            """

    return prompt, references_used


def analyze_expert_opinions(stock_name):
    # 1. 주요 사업 분야 추출
    industries = get_industries(stock_name)

    # 2. vectorDB 문서 검색
    docs_stock, docs_industry = filter_by_stock_and_industries(stock_name, industries)

    # 3. LLM 프롬프트 생성
    prompt, references_used = create_prompt(stock_name, get_industries(stock_name), docs_stock, docs_industry)
    # print("\nGenerated Prompt:\n")
    # print(prompt)

    # 4. LLM 기반 분석 리포트 생성
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system",
                   "content": "당신은 개인 투자자들이 특정 종목에 투자하기 위해 적합한 자료(증권사 애널리스트가 작성한 투자 리포트)에 쉽게 접근하고"
                              "데이터 기반의 효율적인 투자 결정을 내릴 수 있도록 돕는 주식 투자 정보 제공자입니다."},
                  {"role": "user", "content": prompt}],
        temperature=0.1
    )

    result = response.choices[0].message.content
    # print("\nGenerated Result:\n")
    # print(result)

    reference_links = [
        {
            "title": f"[{doc['title']}]",
            "description": f" {doc['firm']}에서 {doc['date']}에 발행한 {doc['category']} 보고서",
            "url": doc['source_url']
        }
        for doc in references_used
    ]

    # 생성된 리스트를 JSON 형식의 문자열로 변환합니다.
    json_output = json.dumps(
        {
            "result_report": result,
            "reference_links": reference_links
        },
        ensure_ascii=False, indent=4)

    print(json_output)


if __name__ == "__main__":
    # stock_name = sys.argv[1]
    stock_name = '삼성전자'
    analyze_expert_opinions(stock_name)
