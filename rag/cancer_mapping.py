from __future__ import annotations

import re


CANONICAL_GROUPS = {
    "colorectal_cancer": {
        "colon cancer",
        "colorectal cancer",
        "rectal cancer",
        "colon and rectal cancer",
    },
    "endometrial_cancer": {
        "uterine cancer",
        "uterine corpus cancer",
        "endometrial cancer",
        "endometrial cancer (uterine cancer)",
    },
    "kidney_cancer": {
        "kidney cancer",
        "renal cancer",
        "renal cell cancer",
        "renal cell carcinoma",
        "kidney (renal cell) cancer",
    },
    "lung_cancer": {
        "lung cancer",
        "non-small cell lung cancer",
        "small cell lung cancer",
        "bronchial tumors",
    },
    "lymphoma": {
        "lymphoma",
        "non-hodgkin lymphoma",
        "hodgkin lymphoma",
        "primary cns lymphoma",
        "aids-related lymphoma",
        "burkitt lymphoma",
        "cutaneous t-cell lymphoma",
    },
    "leukemia": {
        "leukemia",
        "acute myeloid leukemia",
        "acute lymphoblastic leukemia",
        "chronic lymphocytic leukemia",
        "chronic myelogenous leukemia",
        "aml",
        "all",
        "cll",
        "cml",
    },
    "head_and_neck_cancer": {
        "head and neck cancer",
        "oral cancer",
        "oral cavity cancer",
        "mouth cancer",
        "tongue cancer",
        "tonsil cancer",
        "oropharyngeal cancer",
        "nasopharyngeal cancer",
        "laryngeal cancer",
        "hypopharyngeal cancer",
        "pharyngeal cancer",
        "nasal cavity cancer",
        "paranasal sinus cancer",
    },
    "melanoma": {
        "melanoma",
        "skin melanoma",
    },
    "eye_cancer": {
        "eye cancer",
        "intraocular melanoma",
        "melanoma, intraocular (eye)",
        "retinoblastoma",
    },
}


def normalize_label(text: str | None) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[®™]", "", text)
    return text


def canonical_cancer_family(cancer_type: str | None) -> str | None:
    x = normalize_label(cancer_type)
    if not x:
        return None

    for family, aliases in CANONICAL_GROUPS.items():
        if x in aliases:
            return family

    # Conservative fallback: normalized slug.
    slug = re.sub(r"[^a-z0-9]+", "_", x).strip("_")
    return slug or None
