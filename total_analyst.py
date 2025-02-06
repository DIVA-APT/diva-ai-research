import os
from dotenv import load_dotenv
from openai import OpenAI
import sys


# 환경 변수 로드
load_dotenv()

# API 키 설정
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# OpenAI API 초기화
client = OpenAI(api_key=OPENAI_API_KEY)


def generate_rag_prompt(stock_name, fin_result, analyst_result, news_result):
    """
    RAG 프롬프트 생성 함수
    """
    context = (
        f"**재무제표 데이터**:\n{fin_result}\n\n"
        f"**전문가 분석 데이터**:\n{analyst_result}\n\n"
        f"**뉴스 데이터**:\n{news_result}\n\n"
    )

    rag_prompt = (
        f"{context}"
        f"이 문서는 전자공시시스템 DART, 최신 뉴스 데이터, 그리고 전문가 분석 데이터를 바탕으로 작성되었습니다. "
        f"모든 수치는 공식 데이터를 사용하며, {context}에 없는 정보의 경우 가정이나 추정을 포함하지 않도록 해주세요.\n\n"
        f"다음 질문에 따라 종합적인 리포트를 작성해 주세요:\n\n"
        f"1. **재무제표 분석**:\n"
        f"   - {stock_name}의 주요 재무 지표(예: 매출, 영업이익, 순이익 등)와 전년 대비 변화.\n"
        f"   - 부채율, 총자산 대비 기업 가치, 자산 대비 매출 비율을 중심으로 재무 안정성과 성장 가능성을 평가해 주세요.\n\n"
        f"2. **뉴스 분석**:\n"
        f"   - {stock_name}과 관련된 최근 주요 이슈와 트렌드를 요약해 주세요.\n"
        f"   - 해당 이슈가 주식 시장과 투자자들에게 미치는 영향을 분석해 주세요.\n\n"
        f"3. **전문가 분석**:\n"
        f"   - {stock_name}의 주요 사업 분야를 기반으로 전문가들이 제시한 투자 전략과 의견을 요약해 주세요.\n"
        f"   - 긍정적/부정적 관점과 그 근거를 명확히 제시해 주세요.\n\n"
        f"4. **SWOT 분석**:\n"
        f"   - Strengths (강점): 기업의 경쟁 우위 요소와 강점.\n"
        f"   - Weaknesses (약점): 재무적/운영적 약점 또는 개선 필요 영역.\n"
        f"   - Opportunities (기회): 시장에서의 확장 가능성 또는 외부 기회 요인.\n"
        f"   - Threats (위협): 경쟁, 규제, 시장 환경 등 외부 위험 요소.\n\n"
        f"5. **시나리오 기반 예측**:\n"
        f"   - 긍정적 시나리오: 시장 상황이 호전될 경우 예상되는 성과와 기회.\n"
        f"   - 중립적 시나리오: 현재 상태가 유지될 경우 기업의 예상 성과.\n"
        f"   - 부정적 시나리오: 시장 상황이 악화될 경우 예상되는 리스크와 대응 방안.\n\n"
        f"6. **최종 평가 및 권장 사항**:\n"
        f"   - 재무제표, 뉴스, 전문가 의견을 통합적으로 해석하여 {stock_name}의 현재 상태와 미래 전망을 평가해 주세요.\n"
        f"   - 투자자에게 적합한 행동 권고(예: 매수, 보유, 매도)를 제시하고 그 이유를 명확히 설명해 주세요.\n\n"
        f"결과는 간결하지만 명확하게 작성하며, 모든 데이터는 신뢰할 수 있는 출처를 기반으로 작성해 주세요."
    )
    return rag_prompt


def generate_report(stock_name, fin_result, analyst_result, news_result):
    """
    종합 리포트를 생성하는 함수
    """
    # RAG 프롬프트 생성
    rag_prompt = generate_rag_prompt(stock_name, fin_result, analyst_result, news_result)

    # OpenAI GPT-4 호출
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system",
             "content": "당신은 최신 재무제표, 뉴스 데이터 및 전문가 의견을 바탕으로 투자자들에게 유용한 정보를 제공하는 금융 분석 전문가입니다."},
            {"role": "user", "content": rag_prompt}
        ],
        temperature=0.1
    )

    # 결과 반환
    result = response.choices[0].message.content
    return result


if __name__ == "__main__":
    # 테스트용 입력 데이터 설정
    stock_name = sys.argv[1]

    #stock_name = "삼성전자"

    # 데이터 연결 해야됌
    fin_result = """"""

    analyst_result = """"""

    news_result = """"""

    # 종합 리포트 생성
    report = generate_report(stock_name, fin_result, analyst_result, news_result)

    # 결과 출력
    print("\nGenerated Report:\n")
    print(report)
