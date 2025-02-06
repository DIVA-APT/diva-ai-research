import sys
import os
import json

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
index_name = "example-index"
index = pc.Index(index_name)

## doc2vec (KoE5 사용)
model = SentenceTransformer("nlpai-lab/KoE5")

## 산업분류 리스트
standard_industrial_classifications = [
    "IT", "반도체", "타이어", "통신",
    "조선", "자동차", "철강", "화학",
    "에너지", "바이오", "건설", "유통",
    "금융", "서비스", "소비재"
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


## vectorDB 문서 검색
def filter_by_metadata_and_industries(stock_name, industries):
    # 메타데이터 필터링 조건
    filter_conditions_stock = {"stock": stock_name}
    filter_conditions_industries = {"stock": {"$in": industries}}

    # stock_name 기준 문서 검색
    results_stock = index.query(
        vector=[0] * 1024,  # 벡터는 필요없으므로 임의값 사용
        filter=filter_conditions_stock,
        top_k=100,
        include_metadata=True,
        namespace="example-namespace1"
    )

    # industries 기준 문서 검색
    results_industries = index.query(
        vector=[0] * 1024,
        filter=filter_conditions_industries,
        top_k=100,
        include_metadata=True,
        namespace="example-namespace1"
    )

    # print("Results (stock):", results_stock)
    # print("Results (industries):", results_industries)

    return results_stock["matches"], results_industries["matches"]


## 전문가 의견 도출을 위한 프롬프트 생성
def create_prompt(stock_name, industries, docs_stock, docs_industry):
    # 종목명(stock_name) 관련 문서 분석
    docs_stock_text = "\n".join(f"- {doc['metadata']['content']}" for doc in docs_stock)

    # 산업(industry) 관련 문서 분석
    docs_industry_text = "\n".join(f"- {doc['metadata']['content']}" for doc in docs_industry)

    # 프롬프트 생성
    prompt = f"""
            아래는 "{stock_name}"와 관련된 문서입니다:
        
            ### {stock_name} 관련 문서 ###
            {docs_stock_text}
        
            위 문서들을 분석하여 다음 질문에 답변해 주세요:
            1. "{stock_name}"와 관련된 핵심 트렌드는 무엇인가요?
            2. "{stock_name}"의 경쟁력은 어떻게 평가되나요?
            3. 관련된 긍정적/부정적 의견과 그 이유는 무엇인가요?
        
            ### {', '.join(industries)} 산업 관련 문서 ###
            {docs_industry_text}
        
            위 문서들을 분석하여 다음 질문에 답변해 주세요:
            1. "{', '.join(industries)}" 산업의 주요 이슈가 "{stock_name}"에 미치는 영향은 무엇인가요?
            2. 해당 산업의 변화가 "{stock_name}"의 주가 또는 시장에서의 위치에 어떤 영향을 줄 수 있나요?
        
            - **최종 요약**: (위 문서들을 분석하여 얻은 답변과 위에 모든 문서들로부터 알 수 있는 전문가 의견을 종합하여 한글 기준 약 2000자 분량으로 상세히 작성해 주세요.)
            """

    return prompt


def analyze_expert_opinions(stock_name):
    # 1. 주요 사업 분야 추출
    industries = get_industries(stock_name)

    # 2. vectorDB 문서 검색
    docs_stock, docs_industry = filter_by_metadata_and_industries(stock_name, industries)

    # 3. LLM 프롬프트 생성
    prompt = create_prompt(stock_name, get_industries(stock_name), docs_stock, docs_industry)
    # print("\nGenerated Prompt:\n")
    # print(prompt)

    ## 4. LLM 기반 분석 리포트 생성
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system",
                   "content": "당신은 개인 투자자들이 특정 종목에 투자하기 위해 적합한 자료에 쉽게 접근하고"
                              "데이터 기반의 효율적인 투자 결정을 내릴 수 있도록 돕는 주식 투자 정보 제공자입니다."},
                  {"role": "user", "content": prompt}],
        temperature=0.1
    )

    result = response.choices[0].message.content
    # print("\nGenerated Result:\n")
    # print(result)

    result_json = {
        "result_report": result
    }
    print(json.dumps(result_json, ensure_ascii=False))
    #return result_json


if __name__ == "__main__":
    stock_name = sys.argv[1]
    # stock_name = '카카오'
    analyze_expert_opinions(stock_name)
