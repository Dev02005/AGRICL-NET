"""Turn calibrated classifier output plus the lesion mask into one answer.

status:
  disease     confident and lesions were found
  unknown     lesions found but the classifier is not confident: "consult an expert"
  uncertain   classifier is confident but almost no lesion pixels were found
  no_disease  no lesions found and the classifier is not confident
The classifier has no "healthy" class (PlantSeg has no healthy images), so a
healthy leaf is recognised by the absence of lesions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_LESION_SHARE = 0.5  # percent of the leaf; below this the leaf counts as lesion-free

MESSAGES = {
    "disease": "Disease identified.",
    "unknown": "Unknown disease: the symptoms do not match any disease the system knows well. "
               "Please consult an expert. The photo has been saved to help the system learn.",
    "uncertain": "Uncertain: the classifier suggests a disease, but almost no diseased area was found. "
                 "Take a closer photo of the affected part.",
    "no_disease": "No diseased area found. If you can see symptoms, take a closer, sharper photo.",
}


@dataclass
class Decision:
    status: str
    disease: str | None  # None unless status == "disease"
    confidence: float
    top3: list[tuple[str, float]]
    message: str


def decide(probs: np.ndarray, class_names: list[str], threshold: float, lesion_share: float,
           min_lesion_share: float = MIN_LESION_SHARE) -> Decision:
    """probs: calibrated class probabilities for one image (sums to 1).
    lesion_share: severity percent (lesion pixels / leaf pixels)."""
    order = np.argsort(-probs)[:3]
    top3 = [(class_names[i], float(probs[i])) for i in order]
    conf = top3[0][1]
    has_lesions = lesion_share >= min_lesion_share
    confident = conf >= threshold
    if has_lesions and confident:
        status = "disease"
    elif has_lesions:
        status = "unknown"
    elif confident:
        status = "uncertain"
    else:
        status = "no_disease"
    return Decision(status, top3[0][0] if status == "disease" else None, conf, top3, MESSAGES[status])
