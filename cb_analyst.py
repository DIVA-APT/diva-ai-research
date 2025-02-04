import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

model = SentenceTransformer("nlpai-lab/KoE5")

pc = Pinecone(api_key=PINECONE_API_KEY)

indexes = {
    "expert": pc.Index("example-index"),
    "news": pc.Index("news-index"),
    "financials": pc.Index("fin-index")
}


def get_koe5_embedding(query):
    query_with_prefix = f"query: {query}"
    return model.encode(query_with_prefix).tolist()


def search_all_indexes(embedding, top_k=5):
    all_results = []

    # 재무제표 인덱스 검색
    fin_results = indexes["financials"].query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )
    for r in fin_results['matches']:
        raw_data = json.loads(r['metadata']['raw_data'])
        all_results.append((r['score'], f"재무제표: {raw_data.get('corp_name', '')} - {raw_data.get('stlm_dt', '')}"))

    # 뉴스 인덱스 검색
    news_results = indexes["news"].query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )
    for r in news_results['matches']:
        all_results.append(
            (r['score'], f"뉴스: {r['metadata'].get('title', '')} - {r['metadata'].get('chunk', '')[:200]}..."))

    # 전문가 분석 인덱스 검색
    expert_results = indexes["expert"].query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )
    for r in expert_results['matches']:
        all_results.append(
            (r['score'], f"전문가 분석: {r['metadata'].get('stock', '')} - {r['metadata'].get('content', '')}"))

    # 점수에 따라 정렬하고 상위 5개 선택
    return sorted(all_results, key=lambda x: x[0], reverse=True)[:5]


def truncate_text(texts, max_length=3000):
    combined_text = "\n\n".join(texts)
    if len(combined_text) > max_length:
        return combined_text[:max_length] + "..."
    return combined_text


def generate_bot_message(user_query):
    embedding = get_koe5_embedding(user_query)
    top_results = search_all_indexes(embedding)

    # 점수와 내용 분리
    texts = [result[1] for result in top_results]

    combined_texts = truncate_text(texts)

    return {"botMessage": combined_texts}


if __name__ == "__main__":
    user_query = input("Enter your query: ")
    response = generate_bot_message(user_query)

    print("\nGenerated Response:")
    print(response)
