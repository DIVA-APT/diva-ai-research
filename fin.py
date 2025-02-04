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

    rag_prompt = f"{context} 이 문서는 전자공시시스템인 DART에서 제공하는 오픈API서비스 중 '단일회사 주요 재무지표'에 대한 것입니다. 문서 내의 corp_code, corp_name 등 용어에 대한 기본 개념을 바탕으로 {stock_name}에 대한 다음 정보를 바탕으로 회사의 정체성과 시장 포지셔닝을 약 200자로 해석합니다."
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a stock news analysis expert synthesizing complex information."},
            {"role": "user", "content": rag_prompt}
        ],
        temperature=0.1
    )

    result = response.choices[0].message.content
    # print("\nGenerated Result:\n")
    # print(result)

    result_json = {
        "result_fin": result
    }
    print(json.dumps(result_json, ensure_ascii=False))
    # return result_json


if __name__ == "__main__":
    stock_name = sys.argv[1]
    # stock_name = '삼성전자'
    analyze_financial(stock_name)

