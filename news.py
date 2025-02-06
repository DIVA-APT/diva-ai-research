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
index_name = "news-index"
index = pc.Index(index_name)

def generate_questions(stock_name):
    # 기본 질문 생성
    base_question = f"{stock_name}의 주식은 어때?"

    # 관련 질문 생성
    prompt = f"Create three questions about the stock market prospects of {stock_name} outlook in Korean:"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",
             "content": "You are a stock news analysis expert who helps investors quickly understand relevant information through the latest news analysis."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )

    questions = [base_question] + [message for message in response.choices[0].message.content.split('\n')]
    # print(f"생성된 질문: {questions}")
    return questions


def search_articles(questions):
    model = SentenceTransformer("nlpai-lab/KoE5")

    results = {}
    for question in questions:
        query_vector = model.encode(question).tolist()
        response = index.query(vector=query_vector, top_k=5, include_metadata=True)
        results[question] = response['matches']
    return results


def extract_all_chunks(articles_data):
    all_chunks = []
    # 모든 키(=질문)에 대해서 반복
    for question, documents in articles_data.items():
        # 각 문서에서 'metadata' 안의 'chunk'를 추출
        for document in documents:
            if 'metadata' in document and 'chunk' in document['metadata']:
                all_chunks.append(document['metadata']['chunk'])
    return all_chunks


def analyze_news(stock_name):
    questions = generate_questions(stock_name)
    articles = search_articles(questions)

    chunk_list = extract_all_chunks(articles)
    context = " ".join([chunk for chunk in chunk_list])
    rag_prompt = f"Summarize the following information about the stock outlook in Korean: {context}"
    # print(f"rag_prompt: {rag_prompt}")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a stock news analysis expert synthesizing complex information."},
            {"role": "user", "content": rag_prompt}],
        temperature=0.1
    )

    result = response.choices[0].message.content
    # print("\nGenerated Result:\n")
    # print(result)

    result_json = {
        "result_news": result
    }
    print(json.dumps(result_json, ensure_ascii=False))

if __name__ == "__main__":
    stock_name = sys.argv[1]
    # stock_name = '현대모비스'

    analyze_news(stock_name)

