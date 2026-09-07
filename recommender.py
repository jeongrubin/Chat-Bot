"""Retrieval and response helpers for the extracurricular-program chatbot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DISPLAY_FIELDS = (
    "설명",
    "신청대상",
    "신청기간",
    "기간",
    "장소",
    "혜택",
    "운영시간",
    "문의처",
)


def load_programs(path: str | Path) -> list[dict[str, Any]]:
    """Load, validate, and remove exact duplicate program records."""
    with Path(path).open(encoding="utf-8") as file:
        payload = json.load(file)

    programs = payload.get("프로그램_정보")
    if not isinstance(programs, list):
        raise ValueError("JSON에 '프로그램_정보' 목록이 필요합니다.")

    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in programs:
        if not isinstance(item, dict) or not str(item.get("제목", "")).strip():
            continue
        fingerprint = (
            str(item.get("제목", "")).strip(),
            str(item.get("신청기간", "")).strip(),
            str(item.get("기간", "")).strip(),
        )
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(item)
    return unique


def program_to_text(program: dict[str, Any]) -> str:
    """Convert one structured record into searchable Korean text."""
    parts = [str(program.get("제목", ""))]
    for field in DISPLAY_FIELDS:
        value = program.get(field)
        if isinstance(value, list):
            value = " ".join(map(str, value))
        if value:
            parts.append(str(value))
    return " ".join(parts)


class ProgramRetriever:
    """Local Korean text retriever that works without an external API."""

    def __init__(self, programs: list[dict[str, Any]]) -> None:
        if not programs:
            raise ValueError("검색할 프로그램 데이터가 없습니다.")
        self.programs = programs
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(
            program_to_text(program) for program in programs
        )

    def search(
        self, query: str, top_k: int = 3, min_score: float = 0.02
    ) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix)[0]
        ranked_indices = scores.argsort()[::-1]

        results: list[dict[str, Any]] = []
        for index in ranked_indices:
            score = float(scores[index])
            if score < min_score:
                continue
            result = dict(self.programs[index])
            result["검색점수"] = score
            results.append(result)
            if len(results) >= top_k:
                break
        return results


def format_program(program: dict[str, Any]) -> str:
    """Format one search result without inventing missing information."""
    lines = [f"**{program['제목']}**"]
    for field in DISPLAY_FIELDS:
        value = program.get(field)
        if not value:
            continue
        if isinstance(value, list):
            value = "; ".join(map(str, value))
        lines.append(f"- {field}: {value}")
    return "\n".join(lines)


def build_fallback_answer(results: list[dict[str, Any]]) -> str:
    """Return a grounded answer without calling an external API."""
    if not results:
        return (
            "등록된 데이터에서 관련 프로그램을 찾지 못했습니다. "
            "관심 분야나 대상 학년을 조금 더 구체적으로 입력해 주세요."
        )
    body = "\n\n".join(format_program(program) for program in results)
    return f"입력한 조건과 관련성이 높은 프로그램입니다.\n\n{body}"


def generate_gemini_answer(
    query: str,
    results: list[dict[str, Any]],
    *,
    api_key: str,
    model: str,
) -> str:
    """Generate a grounded summary from retrieved records using Gemini."""
    from google import genai

    context = json.dumps(results, ensure_ascii=False, indent=2)
    prompt = f"""당신은 대학 비교과 프로그램 안내 도우미입니다.
아래 검색 결과에 포함된 정보만 사용해 한국어로 답변하세요.
기간이 지난 프로그램일 수 있으므로 최신 정보라고 단정하지 마세요.
정보가 없으면 추측하지 말고 확인할 수 없다고 말하세요.

사용자 질문: {query}

검색 결과:
{context}
"""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    answer = (response.text or "").strip()
    if not answer:
        raise RuntimeError("Gemini API가 빈 응답을 반환했습니다.")
    return answer
