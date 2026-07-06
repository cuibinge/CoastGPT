#!/usr/bin/env python3
"""
Stage 1 task augmentation: yes/no, multi-choice, hard negative.

Adds diverse task types to the training data so the projector learns
strong visual conditioning, not just caption template matching.

Usage (in InstructDataset.post_process):
    from Dataset.task_augmentation import TaskAugmenter
    aug = TaskAugmenter(
        yn_ratio=0.15,        # 15% yes/no questions
        mc_ratio=0.15,        # 15% multi-choice
        hard_neg_ratio=0.10,  # 10% hard negative
        seed=42,
    )
    self.cap_list, self.img_list = aug.augment(self.cap_list, self.img_list)
"""
import random
import re
from typing import Dict, List, Tuple, Optional

# ── Keyword → yes/no question templates ──
_YN_KEYWORDS = {
    "coastline": [
        "Is there a coastline visible in this image? Answer yes or no.",
        "Does this image show a shoreline? Answer yes or no.",
    ],
    "beach": [
        "Is a beach visible in this image? Answer yes or no.",
        "Does this image contain sandy beach areas? Answer yes or no.",
    ],
    "forest": [
        "Is forest or woodland visible in this image? Answer yes or no.",
        "Does this image contain tree cover? Answer yes or no.",
    ],
    "water": [
        "Is a water body visible in this image? Answer yes or no.",
        "Does this image contain ocean, lake, or river? Answer yes or no.",
    ],
    "urban": [
        "Are urban areas or buildings visible in this image? Answer yes or no.",
        "Does this image show city or town development? Answer yes or no.",
    ],
    "mountain": [
        "Are mountains visible in this image? Answer yes or no.",
        "Does this image show mountainous terrain? Answer yes or no.",
    ],
    "farmland": [
        "Is farmland or agriculture visible in this image? Answer yes or no.",
        "Does this image contain crop fields? Answer yes or no.",
    ],
    "wetland": [
        "Is wetland visible in this image? Answer yes or no.",
        "Does this image show marsh or swamp areas? Answer yes or no.",
    ],
    "desert": [
        "Is desert or barren land visible in this image? Answer yes or no.",
        "Does this image show arid terrain? Answer yes or no.",
    ],
    "aquaculture": [
        "Is aquaculture or fish farming visible in this image? Answer yes or no.",
        "Does this image contain fish cages or rafts? Answer yes or no.",
    ],
    "port": [
        "Is a port or harbor visible in this image? Answer yes or no.",
        "Does this image show dock or pier infrastructure? Answer yes or no.",
    ],
    "road": [
        "Are roads or highways visible in this image? Answer yes or no.",
        "Does this image contain transportation routes? Answer yes or no.",
    ],
    "river": [
        "Is a river visible in this image? Answer yes or no.",
        "Does this image show a river channel? Answer yes or no.",
    ],
    "island": [
        "Is an island visible in this image? Answer yes or no.",
        "Does this image show island features? Answer yes or no.",
    ],
    "cloud": [
        "Is there cloud cover in this image? Answer yes or no.",
        "Does this image show cloudy conditions? Answer yes or no.",
    ],
}

# ── Multi-choice templates ──
_MC_TEMPLATES = [
    {
        "question": ("What is the MAIN land cover type in this image?\n"
                     "A. Coastline or beach\nB. Dense forest or vegetation\n"
                     "C. Urban or built-up area\nD. Water body (ocean, lake, river)\n"
                     "E. Farmland or agriculture\n"
                     "Answer with a single letter A-E."),
        "keywords": {
            "A": ["coastline", "beach", "shore", "coastal"],
            "B": ["forest", "woodland", "tree", "vegetation", "jungle", "rainforest"],
            "C": ["urban", "city", "town", "building", "residential", "industrial", "port"],
            "D": ["water", "ocean", "lake", "river", "sea", "marine"],
            "E": ["farmland", "agriculture", "crop", "field", "pasture", "orchard"],
        },
    },
    {
        "question": ("What is the DOMINANT terrain type in this image?\n"
                     "A. Mountain or highland\nB. Plain or flatland\n"
                     "C. Coastal zone\nD. Wetland or marsh\n"
                     "Answer with a single letter A-D."),
        "keywords": {
            "A": ["mountain", "hill", "highland", "ridge", "slope", "elevation"],
            "B": ["plain", "flat", "lowland", "plateau", "grassland"],
            "C": ["coast", "beach", "shoreline", "coastal", "bay", "cliff"],
            "D": ["wetland", "marsh", "swamp", "mudflat", "tidal", "mangrove"],
        },
    },
    {
        "question": ("Is this image dominated by natural or human-made features?\n"
                     "A. Mostly natural\nB. Mostly human-made\n"
                     "C. Mixed natural and human-made\n"
                     "Answer with a single letter A-C."),
        "keywords": {
            "A": ["forest", "mountain", "ocean", "lake", "river", "desert", "wetland", "grassland"],
            "B": ["urban", "city", "industrial", "port", "road", "building", "residential"],
            "C": ["farmland", "agriculture", "park", "garden", "reservoir"],
        },
    },
]

# ── Hard negative refusal templates ──
_HARD_NEG_REFUSALS = [
    "I notice this description does not match the image. The image actually shows {correct}.",
    "This caption appears to be for a different image. Based on what I see, the image contains {correct}.",
    "The provided description is incorrect for this image. The image depicts {correct}.",
]


class TaskAugmenter:
    """Generate yes/no, multi-choice, and hard-negative training samples."""

    def __init__(
        self,
        caption_keep_ratio: float = 1.0,
        yn_ratio: float = 0.15,
        mc_ratio: float = 0.15,
        hard_neg_ratio: float = 0.10,
        seed: int = 42,
    ):
        self.caption_keep_ratio = max(0.0, min(1.0, float(caption_keep_ratio)))
        self.yn_ratio = yn_ratio
        self.mc_ratio = mc_ratio
        self.hard_neg_ratio = hard_neg_ratio
        self.rng = random.Random(seed)

    # ── keyword detection ──
    @staticmethod
    def _detect_keywords(text: str) -> List[str]:
        """Find which keyword categories match the caption text."""
        text_lower = text.lower()
        matched = []
        for kw in _YN_KEYWORDS:
            if kw in text_lower:
                matched.append(kw)
        return matched

    # ── yes/no generation ──
    def _build_yesno(
        self, caption: str
    ) -> Optional[List[Dict[str, str]]]:
        """Generate yes/no QA pairs from caption keywords."""
        present_kw = self._detect_keywords(caption)
        if not present_kw:
            return None

        # Pick 1-2 keywords that ARE present → answer yes
        n_yes = min(2, len(present_kw))
        yes_kws = self.rng.sample(present_kw, n_yes)

        # Pick 1-2 keywords NOT present → answer no
        all_kws = list(_YN_KEYWORDS.keys())
        absent_kws = [k for k in all_kws if k not in present_kw]
        if not absent_kws:
            return None
        n_no = min(2, len(absent_kws))
        no_kws = self.rng.sample(absent_kws, n_no)

        convs = []
        for kw in yes_kws:
            q = self.rng.choice(_YN_KEYWORDS[kw])
            convs.append({
                "Question": f"<image>\n{q}",
                "Answer": "yes",
                "task_text": "问答",
                "element_text": kw,
            })
        for kw in no_kws:
            q = self.rng.choice(_YN_KEYWORDS[kw])
            # For "no" answers, the element is NOT present — tag with opposite element
            opposite = {"coastline": "inland", "forest": "urban", "water": "desert",
                        "mountain": "plain", "urban": "forest", "farmland": "urban"}.get(kw, "other")
            convs.append({
                "Question": f"<image>\n{q}",
                "Answer": "no",
                "task_text": "问答",
                "element_text": opposite,
            })

        return convs if convs else None

    # ── multi-choice generation ──
    def _build_multichoice(
        self, caption: str
    ) -> Optional[List[Dict[str, str]]]:
        """Generate multi-choice QA from caption content."""
        text_lower = caption.lower()
        # Infer element from caption content
        element = "mixed"
        for kw in ["coastline", "beach", "shore", "coastal"]:
            if kw in text_lower:
                element = "海岸线"; break
        for kw in ["forest", "woodland", "tree", "vegetation"]:
            if kw in text_lower:
                element = "林地" if element == "mixed" else element; break
        for kw in ["urban", "city", "building", "residential"]:
            if kw in text_lower:
                element = "城市" if element == "mixed" else element; break
        for kw in ["water", "ocean", "lake", "river", "sea"]:
            if kw in text_lower:
                element = "水体" if element == "mixed" else element; break

        results = []
        for template in _MC_TEMPLATES:
            scores = {}
            for letter, kws in template["keywords"].items():
                scores[letter] = sum(1 for k in kws if k in text_lower)
            best = max(scores, key=scores.get)
            if scores[best] > 0:
                results.append({
                    "Question": f"<image>\n{template['question']}",
                    "Answer": best,
                    "task_text": "分类",
                    "element_text": element,
                })

        if results:
            return self.rng.sample(results, min(2, len(results)))
        return None

    # ── hard negative generation ──
    def _build_hard_negatives(
        self,
        cap_list: List[List[Dict]],
        img_list: List[str],
        count: int,
    ) -> Tuple[List[List[Dict]], List[str]]:
        """Create hard negatives: image A paired with wrong caption B."""
        neg_convos = []
        neg_imgs = []
        indices = list(range(len(cap_list)))
        self.rng.shuffle(indices)

        for i in range(0, min(count * 2, len(indices) - 1), 2):
            idx_img = indices[i]
            idx_cap = indices[i + 1]

            # Get the WRONG caption
            wrong_conv = cap_list[idx_cap]
            if not wrong_conv or not isinstance(wrong_conv, list):
                continue
            wrong_answer = wrong_conv[0].get("Answer", "")
            if not wrong_answer:
                continue

            # Get the CORRECT caption for the image
            correct_conv = cap_list[idx_img]
            if not correct_conv or not isinstance(correct_conv, list):
                continue
            correct_answer = correct_conv[0].get("Answer", "")[:200]

            # Build refusal response
            refusal_template = self.rng.choice(_HARD_NEG_REFUSALS)
            refusal = refusal_template.format(correct=correct_answer)

            # Question: ask model to validate a wrong description
            question = (
                f"<image>\nA student claims: \"{wrong_answer[:200]}\"\n"
                f"Is this description accurate for the image? "
                f"If not, what does the image actually show?"
            )

            neg_convos.append([{
                "Question": question,
                "Answer": refusal,
                "task_text": "校验",
                "element_text": "mixed",
            }])
            neg_imgs.append(img_list[idx_img])

        return neg_convos, neg_imgs

    # ── main augmentation ──
    def augment(
        self,
        cap_list: List[List[Dict]],
        img_list: List[str],
    ) -> Tuple[List[List[Dict]], List[str]]:
        """
        Augment caption list with yes/no, multi-choice, and hard negatives.
        Returns (new_cap_list, new_img_list).
        """
        total = len(cap_list)

        # Count how many of each type to add
        n_yn = max(0, int(total * self.yn_ratio))
        n_mc = max(0, int(total * self.mc_ratio))
        n_hn = max(0, int(total * self.hard_neg_ratio))

        keep_count = total
        if self.caption_keep_ratio < 1.0:
            keep_count = max(1, int(round(total * self.caption_keep_ratio)))
        keep_indices = list(range(total))
        self.rng.shuffle(keep_indices)
        keep_indices = set(keep_indices[:keep_count])

        new_cap_list = [cap_list[i] for i in range(total) if i in keep_indices]
        new_img_list = [img_list[i] for i in range(total) if i in keep_indices]

        # Collect CAP samples we can augment
        cap_samples = []
        for i, convs in enumerate(cap_list):
            if not convs or not isinstance(convs, list):
                continue
            answer = convs[0].get("Answer", "")
            if isinstance(answer, str) and len(answer) > 10:
                cap_samples.append((i, answer))

        self.rng.shuffle(cap_samples)

        # ── Add yes/no ──
        yn_added = 0
        for idx, answer in cap_samples:
            if yn_added >= n_yn:
                break
            yn_convs = self._build_yesno(answer)
            if yn_convs:
                for c in yn_convs:
                    new_cap_list.append([c])
                    new_img_list.append(img_list[idx])
                    yn_added += 1
                    if yn_added >= n_yn:
                        break

        # ── Add multi-choice ──
        mc_added = 0
        for idx, answer in cap_samples:
            if mc_added >= n_mc:
                break
            mc_convs = self._build_multichoice(answer)
            if mc_convs:
                for c in mc_convs:
                    new_cap_list.append([c])
                    new_img_list.append(img_list[idx])
                    mc_added += 1
                    if mc_added >= n_mc:
                        break

        # ── Add hard negatives ──
        if n_hn > 0:
            hn_convs, hn_imgs = self._build_hard_negatives(
                cap_list, img_list, n_hn
            )
            new_cap_list.extend(hn_convs)
            new_img_list.extend(hn_imgs)

        # Shuffle
        combined = list(zip(new_cap_list, new_img_list))
        self.rng.shuffle(combined)
        new_cap_list, new_img_list = zip(*combined) if combined else ([], [])

        return list(new_cap_list), list(new_img_list)
