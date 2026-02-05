from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

from app.settings import get_settings
from app.rag_pipeline import RAGPipeline


def load_questions(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main() -> None:
    s = get_settings()
    rag = RAGPipeline(s)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    qpath = os.path.join(base_dir, "eval", "questions.jsonl")
    rpath = os.path.join(base_dir, "eval", "results.jsonl")

    questions = load_questions(qpath)
    if not questions:
        print("No questions found.")
        return

    retrieval_times = []
    generation_times = []
    refused = 0

    with open(rpath, "w", encoding="utf-8") as out:
        for q in questions:
            qid = q.get("id")
            question = q.get("question", "").strip()
            if not question:
                continue

            try:
                result = rag.query(question)
            except Exception as e:
                record = {"id": qid, "question": question, "error": str(e)}
                out.write(json.dumps(record) + "\n")
                continue

            ans = result["answer"]
            timing = result["timing"]
            sources = result["sources"]

            retrieval_times.append(float(timing["retrieval_seconds"]))
            generation_times.append(float(timing["generation_seconds"]))

            if ans.strip() == "I don't know based on the provided documents.":
                refused += 1

            record = {
                "id": qid,
                "question": question,
                "answer": ans,
                "timing": timing,
                "top_sources": [
                    {
                        "filename": ssrc.get("filename"),
                        "page_number": ssrc.get("page_number"),
                        "score": ssrc.get("score"),
                        "chunk_id": ssrc.get("chunk_id"),
                    }
                    for ssrc in sources
                ],
            }
            out.write(json.dumps(record) + "\n")

    avg_ret = sum(retrieval_times) / max(1, len(retrieval_times))
    avg_gen = sum(generation_times) / max(1, len(generation_times))
    print(f"questions: {len(questions)}")
    print(f"avg retrieval seconds: {avg_ret:.3f}")
    print(f"avg generation seconds: {avg_gen:.3f}")
    print(f"refusals (I don't know...): {refused}")
    print(f"saved results to: {rpath}")


if __name__ == "__main__":
    main()
