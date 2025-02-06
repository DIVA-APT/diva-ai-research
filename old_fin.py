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
    api_key=OPENAI_API_KEY,
)

## Pinecone 초기화
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "fin-index"
index = pc.Index(index_name)

## doc2vec (KoE5 사용)
model = SentenceTransformer("nlpai-lab/KoE5")


def analyze_financial(stock_name):
    question = f"{stock_name}의 부채율, 총자산 대비 기업 가치, {stock_name}이 가지고 있는 전체 자산 대비 매출 등을 바탕으로 동일 산업의 타종목과 비교하여 이 기업의 가치가 어떤지 알 수 있는 지표"

    query_vector = model.encode(question).tolist()
    response = index.query(vector=query_vector, top_k=100, include_metadata=True)

    chunks = [matches['metadata']['raw_data'] for matches in response['matches']]
    context = " ".join(chunks)

    rag_prompt = (f"{context} 이 문서는 전자공시시스템인 DART에서 제공하는 오픈API서비스를 통해 얻은 '회사별 주요 재무지표'에 대한 것입니다. 문서 내의 'corp_code', 'corp_name' 등의 기본 개념을 바탕으로 {stock_name}에 대한 주요 재무재표 분석을 제공해 주세요. 추가로 부채율, 총자산 대비 기업 가치, {stock_name}의 전체 자산 대비 매출을 중심으로 실제 데이터를 기반으로 한 분석을 수행하며, 이를 동일 산업 내 다른 주요 기업들과 비교하여 표 형식으로 정리해 주세요. 모든 수치는 공식 데이터를 사용하며, {context}에 없는 정보의 경우 인터넷 검색을 활용하되 가정이나 추정, 예시를 사용하지 않도록 해주세요. 비교 대상 기업들의 이름과 각 기업의 비교 지표도 명확히 제공해 주세요. 결과는 곧바로 사용자에게 유용한 정보로 제공되어야 하며, 분석 과정이나 결과의 해석에 대한 내용은 제외해야 합니다.")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",
             "content": "당신은 개인 투자자들이 특정 종목에 투자하기 위해 필요한 재무제표 데이터에 쉽게 접근하고 데이터 기반의 효율적인 투자 결정을 내릴 수 있도록 돕는 주식 투자 정보 제공자입니다."},
            {"role": "user", "content": rag_prompt}
        ],
        temperature=0.1
    )

    result = response.choices[0].message.content
    print("\nGenerated Result:\n")
    print(result)

    result_json = {
        "result_fin": result
    }
    # print(json.dumps(result_json, ensure_ascii=False))
    # return result_json


if __name__ == "__main__":
    # stock_name = sys.argv[1]
    stock_name = '삼성전자'
    analyze_financial(stock_name)