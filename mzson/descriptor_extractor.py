from __future__ import annotations

import json
import logging
from typing import List, Optional, Union, Tuple, Any, Dict
from dataclasses import dataclass
from PIL import Image as PILImage
from PIL.Image import Image as PILImageType

import numpy as np

from .siglip_itm import SigLipITM

import base64
import io
from openai import OpenAI


@dataclass
class DescriptorExtractorConfig:
    """Descriptor 추출 설정"""
    use_chain_descriptors: bool = False
    gpt_model: str = "gpt-4o-mini"
    n_descriptors: int = 3


class DescriptorExtractor:
    """
    A class to handle the extraction of text descriptors from image observations
    using different strategies.
    """

    def __init__(self, itm: SigLipITM, cfg: DescriptorExtractorConfig):
        self.itm = itm
        self.cfg = cfg
        self.n_descriptors = cfg.n_descriptors
        self.client = OpenAI()


    def extract_target_objects_from_description(self, description: str) -> Tuple[List[str], List[str]]:
        """
        Uses GPT to extract a main target and context objects from a descriptive goal.
        Returns a tuple: (main_target_list, context_objects_list)
        """
        try:
            # Prepare the prompt for GPT
            prompt = (
                f"Analyze the following user instruction for navigation: '{description}'\n"
                "Identify the single main physical object the user ultimately wants to find (main_target_object) "
                "and any other physical objects mentioned that describe its location (context_objects).\n"
                "Return the result in JSON format with two keys: 'main_target_object' (a string) and 'context_objects' (a list of strings)."
            )
            
            response = self.client.chat.completions.create(
                model=self.cfg.gpt_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                response_format={"type": "json_object"},
            )
            response_str = response.choices[0].message.content
            
            if not response_str: return [], []

            response_json = json.loads(response_str)
            main_target = response_json.get("main_target_object")
            context_objects = response_json.get("context_objects", [])

            main_target_list = [main_target] if isinstance(main_target, str) and main_target else []
            
            if not isinstance(context_objects, list) or not all(isinstance(i, str) for i in context_objects):
                logging.warning(f"Could not parse a valid list for context_objects. Got: {context_objects}")
                context_objects = []

            logging.info(f"From '{description}', extracted Main: {main_target_list}, Context: {context_objects}")
            return main_target_list, context_objects

        except Exception as e:
            logging.error(f"Failed to extract objects from description due to an error: {e}", exc_info=True)
            return [], []

    def extract_keywords_from_image(self, goal_image: PILImageType) -> List[str]:
        """VLM을 사용하여 목표 이미지에서 핵심 키워드를 추출합니다."""
        logging.info("Extracting keywords from goal image...")
        try:
            # VLM에 전달하기 위해 이미지 인코딩
            buffered = io.BytesIO()
            goal_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            prompt = (
                "Analyze the following image and describe the main object "
                "with a few concise, comma-separated keywords. "
                "Focus on color, material, and object type. For example: "
                "'red leather armchair', 'round wooden coffee table', 'tall green ficus tree'."
            )

            response = self.client.chat.completions.create(
                model=self.cfg.gpt_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_str}"},
                            },
                        ],
                    }
                ],
                max_tokens=100,
                seed=getattr(self.cfg, 'seed', None),  # config에서 seed 가져오기
            )
            keywords_str = response.choices[0].message.content
            if not keywords_str:
                return []
            
            # 따옴표 제거 및 공백 정리
            keywords_str = keywords_str.replace("'", "").replace('"', '')
            keywords = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
            logging.info(f"Extracted keywords: {keywords}")
            return keywords

        except Exception as e:
            logging.error(f"Error extracting keywords from image: {e}", exc_info=True)
            return []


    def analyze_scene_for_goal(
        self,
        rgb: np.ndarray,
        goal: Any, # Can be string or PIL Image
        key_objects: List[str],
    ) -> Tuple[List[str], float, List[str]]:
        """
        Performs a unified VLM analysis of a scene to get descriptors, a likelihood score, and parsed objects.
        """
        try:
            messages: List[Dict[str, Any]]

            # Prepare scene image encoding first, as it's used in multiple places
            scene_buffered = io.BytesIO()
            PILImage.fromarray(rgb).save(scene_buffered, format="PNG")
            scene_img_str = base64.b64encode(scene_buffered.getvalue()).decode("utf-8")

            # Goal-dependent message preparation (JSON response)
            if isinstance(goal, str) and goal.strip():
                prompt = (
                    "You are an indoor navigation assistant robot.\n"
                    "Analyze the provided scene image with respect to the textual goal.\n"
                    f"Goal: '{goal}'.\n"
                    f"Return a compact JSON object with exactly these keys: \n"
                    f"- 'descriptors': an array of {self.n_descriptors} short, distinct visual descriptors (strings).\n"
                    f"- 'objects': an array of physical object names (strings) that APPEAR IN the descriptors (verbatim substrings);\n"
                    f"   do not invent new objects; include only nouns/noun phrases present in descriptors; deduplicate.\n"
                    f"- 'likelihood': a float in [0.0, 1.0] indicating how promising it is to proceed in this direction.\n"
                    f"No extra keys or text."
                )
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{scene_img_str}"},
                            },
                        ],
                    }
                ]
            elif isinstance(goal, PILImageType):
                goal_buffered = io.BytesIO()
                goal.save(goal_buffered, format="PNG")
                goal_img_str = base64.b64encode(goal_buffered.getvalue()).decode("utf-8")
                prompt = (
                    "You are an indoor navigation assistant robot.\n"
                    "The first image is the current scene; the second is the goal object.\n"
                    f"Return a compact JSON object with exactly these keys: \n"
                    f"- 'descriptors': an array of {self.n_descriptors} short, distinct visual descriptors (strings) of the scene relevant to the goal.\n"
                    f"- 'objects': an array of physical object names (strings) that APPEAR IN the descriptors (verbatim substrings);\n"
                    f"   do not invent new objects; include only nouns/noun phrases present in descriptors; deduplicate.\n"
                    f"- 'likelihood': a float in [0.0, 1.0] reflecting how promising it is to proceed toward the goal.\n"
                    f"No extra keys or text."
                )
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{scene_img_str}", "detail": "low"},
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{goal_img_str}", "detail": "low"},
                            },
                        ],
                    }
                ]
            else:
                prompt = (
                    "You are an indoor navigation assistant robot.\n"
                    f"Return a compact JSON object with: 'descriptors' (array of {self.n_descriptors} short visual descriptors), "
                    "'objects' (array of object names), and 'likelihood' (float 0.5)."
                )
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{scene_img_str}"},
                            },
                        ],
                    }
                ]

            response = self.client.chat.completions.create(
                model=self.cfg.gpt_model,
                messages=messages,
                max_tokens=300,
                response_format={"type": "json_object"},
                seed=getattr(self.cfg, 'seed', None),  # config에서 seed 가져오기
            )
            response_str = response.choices[0].message.content
            if not response_str:
                return [], 0.0, []

            # --- Parse JSON response ---
            try:
                parsed = json.loads(response_str)
            except Exception:
                logging.warning("Failed to parse JSON; falling back to empty objects.")
                parsed = {}

            descriptors = parsed.get("descriptors", [])
            if not isinstance(descriptors, list):
                descriptors = []
            descriptors = [d.strip() for d in descriptors if isinstance(d, str) and d.strip()]

            likelihood_score = parsed.get("likelihood", 0.0)
            try:
                likelihood_score = float(likelihood_score)
            except Exception:
                likelihood_score = 0.0
            likelihood_score = max(0.0, min(1.0, likelihood_score))

            objects_raw: List[str] = parsed.get("objects", [])
            if not isinstance(objects_raw, list):
                objects_raw = []
            objects_raw = [o.strip() for o in objects_raw if isinstance(o, str) and o.strip()]
            # Keep only objects that actually appear in descriptors (case-insensitive substring)
            objects: List[str] = []
            desc_lower = [d.lower() for d in descriptors]
            for o in objects_raw:
                ol = o.lower()
                if any(ol in d for d in desc_lower):
                    if o not in objects:
                        objects.append(o)
            
            return descriptors, likelihood_score, objects

        except Exception as e:
            logging.error(f"Failed during unified VLM analysis due to an error: {e}", exc_info=True)
            return [], 0.0, []



