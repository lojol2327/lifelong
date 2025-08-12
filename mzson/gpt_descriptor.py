from __future__ import annotations

import base64
from typing import List, Literal

import numpy as np
import requests

from mzson.const import END_POINT, OPENAI_KEY


def _rgb_to_base64_png(image_rgb: np.ndarray) -> str:
    import io
    from PIL import Image

    img = image_rgb
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_text_descriptors(
    image_rgb: np.ndarray,
    stage: Literal["rapid", "goal"],
    top_n: int = 5,
    model: str = "gpt-4o-mini",  # lightweight multimodal
    use_cot: bool = False,
    self_consistency: int = 1,
) -> List[str]:
    """Call OpenAI Vision model to produce up to N textual descriptors from a frontier image.

    - stage == "rapid": exploration-friendly cues
    - stage == "goal": goal-oriented cues using goal_hint(text) if provided
    """
    img_b64 = _rgb_to_base64_png(image_rgb)
    base_prompt = (
        "You are a navigation assistant. From the given egocentric image, list concise textual hints (max 12 words each). "
        "Avoid redundancy; make them diverse."
    )
    if stage == "rapid":
        base_prompt += " Focus on exploration signals (doorway, corridor direction, open space, stairs, intersections, signage)."
    else:
        base_prompt += " Focus on how to reach or locate the target described in the task using only visual context (do not assume unknown details)."
    if use_cot:
        # CoT 지시: 내부적으로 다양하게 사고한 뒤, 출력은 bullet만
        base_prompt = (
            "Think step by step to enumerate visually grounded cues first (do not output your reasoning). "
            + base_prompt
        )
    base_prompt += f" Return {top_n} bullet items only."

    headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
    url = END_POINT.rstrip("/") + "/chat/completions"

    def _one_call() -> List[str]:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": base_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    ],
                }
            ],
            "temperature": 0.7 if use_cot else 0.2,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        txt = resp.json()["choices"][0]["message"]["content"]
        lines = [l.strip("- ●•* \t") for l in txt.strip().splitlines() if l.strip()]
        return lines

    try:
        all_lines: List[str] = []
        rounds = max(1, int(self_consistency))
        for _ in range(rounds):
            lines = _one_call()
            for l in lines:
                if l not in all_lines:
                    all_lines.append(l)
                if len(all_lines) >= top_n:
                    break
            if len(all_lines) >= top_n:
                break
        return all_lines[:top_n] if all_lines else []
    except Exception:
        # return empty to signal "no reliable hints"
        return []


def generate_text_descriptors_chain(
    image_rgb: np.ndarray,
    stage: Literal["rapid", "goal"],
    top_n: int = 5,
    model: str = "gpt-4o-mini",
    max_retries: int = 2,
) -> List[str]:
    """Chain-of-Descriptor: 한 번에 1개씩, 중복 금지 조건을 누적해 N개 생성.

    - 각 스텝에서 이전에 뽑힌 힌트 목록을 프롬프트에 넣어 중복 회피를 강제
    - 간단한 문자열 유사도 필터(소문자/공백정리 + n-gram 자카드)로 2차 방어
    """
    def _norm(s: str) -> str:
        return " ".join(s.lower().strip().split())

    def _jaccard(a: str, b: str) -> float:
        def grams(x: str) -> set:
            toks = x.split()
            return set(zip(toks, toks[1:])) if len(toks) > 1 else set(toks)
        A, B = grams(_norm(a)), grams(_norm(b))
        if not A or not B:
            return 1.0 if _norm(a) == _norm(b) else 0.0
        return len(A & B) / max(1, len(A | B))

    found: List[str] = []
    for _ in range(top_n):
        retry = 0
        while retry <= max_retries:
            prompt_items = "\n".join([f"- {h}" for h in found]) if found else "(none)"
            base = (
                "You are a navigation assistant. Propose ONE new concise textual hint (<= 12 words) from the image.\n"
                "Already found hints (do NOT repeat or paraphrase):\n" + prompt_items + "\n"
                "The new hint must be semantically distinct from all listed."
            )
            if stage == "rapid":
                base += " Focus on exploration signals (doorway, corridor direction, open space, stairs, intersections, signage)."
            else:
                base += " Focus on visually grounded cues to locate the described target; do not invent facts."

            headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
            url = END_POINT.rstrip("/") + "/chat/completions"
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": base},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_rgb_to_base64_png(image_rgb)}"}},
                        ],
                    }
                ],
                "temperature": 0.5,
            }
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=60)
                resp.raise_for_status()
                cand = resp.json()["choices"][0]["message"]["content"].strip("- \n\t")
                # 중복/유사 필터
                if any(_jaccard(cand, h) >= 0.6 for h in found):
                    retry += 1
                    continue
                found.append(cand)
                break
            except Exception:
                retry += 1
                if retry > max_retries:
                    # give up this slot without forcing a possibly incorrect hint
                    break
    return found[:top_n]


