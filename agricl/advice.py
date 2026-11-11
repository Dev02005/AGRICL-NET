"""Treatment guidance by disease type and severity level.

Deliberately general and non-prescriptive: it names good practices, never a
chemical or a dose, and always refers the farmer to a local agricultural
extension officer for products registered for their crop and region.
"""

from __future__ import annotations

from dataclasses import dataclass

# Diseases whose type the keyword rules below would get wrong.
OVERRIDES = {
    "cabbage black rot": "bacterial",  # Xanthomonas; apple and grape black rot are fungal
    "cucumber angular leaf spot": "bacterial",
    "bean halo blight": "bacterial",
    "raspberry fire blight": "bacterial",
    "citrus canker": "bacterial",
    "citrus greening disease": "vector",  # bacterium spread by psyllid insects
    "banana panama disease": "wilt",
    "bell pepper blossom end rot": "physiological",
}

# First matching rule wins.
RULES = [
    ("vector", ("virus", "mosaic", "bunchy top", "leafroll", "leaf curl virus", "pox")),
    ("bacterial", ("bacterial", "canker", "black chaff")),
    ("wilt", ("wilt",)),
    ("fungal", ("rust", "mildew", "blight", "spot", "rot", "scab", "anthracnose", "mold", "smut",
                "scorch", "blotch", "blast", "streak", "curl", "pocket", "mummy", "tar spot")),
]

TYPE_NAMES = {
    "fungal": "fungal or fungus-like disease",
    "bacterial": "bacterial disease",
    "vector": "virus or insect-spread disease",
    "wilt": "vascular wilt disease",
    "physiological": "growing-condition problem (not an infection)",
    "general": "plant disease",
}

TYPE_ADVICE = {
    "fungal": [
        "Remove and destroy infected leaves; do not compost them.",
        "Keep leaves dry: water at the base of the plant, in the morning.",
        "Improve air flow by spacing and pruning plants.",
    ],
    "bacterial": [
        "Remove infected leaves and disinfect tools after use.",
        "Avoid working with plants while they are wet.",
        "Avoid overhead irrigation; bacteria spread in water splashes.",
    ],
    "vector": [
        "There is no cure once a plant is infected: remove infected plants to protect the rest.",
        "Control the insects that spread it (aphids, whiteflies, psyllids), e.g. with sticky traps and nets.",
        "Use certified disease-free seed or planting material next season.",
    ],
    "wilt": [
        "Remove wilted plants with their roots.",
        "Do not move soil or water from infected areas to healthy ones.",
        "Rotate to a different crop and choose resistant varieties next season.",
    ],
    "physiological": [
        "Water regularly and evenly; avoid cycles of drought and heavy watering.",
        "Mulch to keep soil moisture steady.",
        "Ask about a soil test for calcium and other nutrients.",
    ],
    "general": [
        "Remove affected leaves and keep the area around plants clean.",
        "Monitor nearby plants for the same symptoms.",
    ],
}

SEVERITY_ADVICE = {
    "Mild": "Early stage: act now with the steps above and check the plant again within a few days.",
    "Moderate": "Spreading: apply the steps above today and ask your local agricultural extension officer "
                "which treatments are registered for this crop.",
    "Severe": "Severe: remove heavily infected plants to protect the rest of the field and contact an "
              "agricultural expert as soon as possible.",
}

DISCLAIMER = ("This is decision support, not a diagnosis by an expert. Use only treatments registered for your "
              "crop and region, following your local agricultural extension officer's advice.")


@dataclass(frozen=True)
class Advice:
    disease_type: str
    type_name: str
    steps: tuple[str, ...]
    severity_note: str
    disclaimer: str = DISCLAIMER


def disease_type(disease: str) -> str:
    d = disease.lower()
    if d in OVERRIDES:
        return OVERRIDES[d]
    for kind, words in RULES:
        if any(w in d for w in words):
            return kind
    return "general"


def advice_for(disease: str | None, severity_level: str) -> Advice:
    """disease=None (unknown disease) gives general advice plus a referral."""
    kind = disease_type(disease) if disease else "general"
    steps = list(TYPE_ADVICE[kind])
    if disease is None:
        steps.insert(0, "The disease could not be identified with confidence: show this plant to an expert.")
    note = SEVERITY_ADVICE.get(severity_level, "")
    return Advice(kind, TYPE_NAMES[kind], tuple(steps), note)
