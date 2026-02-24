from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Advice:
    urgency: str                 # "urgent" | "soon" | "monitor"
    contagious: bool
    base_see_doctor: bool        # baseline before confidence adjustment
    summary: str                 # short paragraph
    self_care: List[str]
    red_flags: List[str]


def _confidence_band(conf: float) -> str:
    if conf >= 0.75:
        return "high"
    if conf >= 0.55:
        return "medium"
    return "low"


# === PRECISE mapping: your class_names.json (DermNet) ===
CLASS_ADVICE: Dict[str, Advice] = {
    "Acne and Rosacea Photos": Advice(
        urgency="monitor",
        contagious=False,
        base_see_doctor=False,
        summary=(
            "This may be consistent with acne or rosacea. Start gentle skincare and avoid harsh scrubs. "
            "If redness/flushing dominates (rosacea), prescription treatment can help."
        ),
        self_care=[
            "Use a gentle cleanser; avoid scrubbing.",
            "Choose non-comedogenic products; avoid heavy oils.",
            "If redness/flushing: avoid triggers (heat, spicy foods, alcohol) and use daily sunscreen.",
            "Avoid picking to reduce scarring."
        ],
        red_flags=[
            "Painful deep cysts/nodules or scarring",
            "No improvement after ~6–8 weeks of basic care",
            "Eye irritation or vision symptoms (possible ocular rosacea)"
        ],
    ),

    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": Advice(
        urgency="urgent",
        contagious=False,
        base_see_doctor=True,
        summary=(
            "This category includes sun-damage lesions and possible skin cancers. "
            "Please arrange a dermatologist evaluation promptly."
        ),
        self_care=[
            "Avoid sun exposure; use SPF 30+ daily and protective clothing.",
            "Do not pick/peel lesions.",
            "Photograph the lesion to track changes (size/color/bleeding)."
        ],
        red_flags=[
            "Bleeding, ulceration, or rapid growth",
            "New lesion that changes quickly",
            "Persistent non-healing sore"
        ],
    ),

    "Atopic Dermatitis Photos": Advice(
        urgency="monitor",
        contagious=False,
        base_see_doctor=False,
        summary=(
            "This may be consistent with atopic dermatitis (eczema-type). Focus on moisture repair and avoiding irritants. "
            "If frequent flares or severe itch, a clinician can guide treatment."
        ),
        self_care=[
            "Moisturize 2–3×/day with fragrance-free emollient (especially after bathing).",
            "Use mild soap; avoid fragrance and hot showers.",
            "Avoid scratching; keep nails short."
        ],
        red_flags=[
            "Crusting, pus, increasing pain (possible infection)",
            "Large area involvement or sleep disruption",
            "No improvement in 1–2 weeks"
        ],
    ),

    "Bullous Disease Photos": Advice(
        urgency="urgent",
        contagious=False,
        base_see_doctor=True,
        summary=(
            "This category includes blistering (bullous) disorders, some of which can be serious. "
            "Please seek medical assessment promptly."
        ),
        self_care=[
            "Do not pop blisters; keep areas clean and covered.",
            "Avoid new topical products until evaluated."
        ],
        red_flags=[
            "Blisters in mouth/eyes/genitals",
            "Fever or feeling very unwell",
            "Rapidly spreading blisters or skin peeling"
        ],
    ),

    "Cellulitis Impetigo and other Bacterial Infections": Advice(
        urgency="urgent",
        contagious=True,  # impetigo contagious; cellulitis not usually, but safe to warn hygiene
        base_see_doctor=True,
        summary=(
            "This category includes bacterial infections like cellulitis/impetigo that often need medical treatment. "
            "Seek care promptly, especially if spreading or painful."
        ),
        self_care=[
            "Keep area clean; cover open lesions.",
            "Avoid sharing towels/clothing; wash hands frequently.",
            "Mark the edge of redness to monitor spread."
        ],
        red_flags=[
            "Fever/chills",
            "Rapidly spreading redness or severe pain",
            "Pus, swelling, or red streaks"
        ],
    ),

    "Eczema Photos": Advice(
        urgency="monitor",
        contagious=False,
        base_see_doctor=False,
        summary=(
            "This may be consistent with eczema/dermatitis. Restore the skin barrier with regular moisturizing and avoid irritants. "
            "See a clinician if persistent or severe."
        ),
        self_care=[
            "Moisturize frequently with fragrance-free cream/ointment.",
            "Use gentle cleanser; avoid harsh soaps and fragrances.",
            "Identify triggers (new products, detergents, stress)."
        ],
        red_flags=[
            "Weeping/crusting or increasing pain (infection concern)",
            "No improvement in 1–2 weeks",
            "Severe widespread rash"
        ],
    ),

    "Exanthems and Drug Eruptions": Advice(
        urgency="urgent",
        contagious=False,
        base_see_doctor=True,
        summary=(
            "This category can include viral rashes or drug-related eruptions. Some drug rashes can be serious. "
            "If you started a new medication recently, seek medical advice promptly."
        ),
        self_care=[
            "Stop and review any new medications with a clinician (do not stop critical meds without advice).",
            "Avoid new skincare products until assessed."
        ],
        red_flags=[
            "Fever, facial swelling, or trouble breathing",
            "Blistering or skin peeling",
            "Mouth/eye/genital involvement"
        ],
    ),

    "Hair Loss Photos Alopecia and other Hair Diseases": Advice(
        urgency="soon",
        contagious=False,
        base_see_doctor=True,
        summary=(
            "This category includes hair loss conditions with different causes. A clinician can help confirm the type and treatment."
        ),
        self_care=[
            "Avoid tight hairstyles and harsh chemicals/heat.",
            "Ensure adequate nutrition and reduce scalp irritation.",
            "Track pattern and timeline with photos."
        ],
        red_flags=[
            "Sudden patchy hair loss",
            "Scalp pain, scaling, or pus",
            "Associated fatigue/weight change (possible systemic cause)"
        ],
    ),

    "Herpes HPV and other STDs Photos": Advice(
        urgency="soon",
        contagious=True,
        base_see_doctor=True,
        summary=(
            "This category may include sexually transmitted infections. Medical testing is important for confirmation and treatment. "
            "Avoid sexual contact until assessed."
        ),
        self_care=[
            "Avoid sexual contact until evaluated.",
            "Do not apply strong irritants to lesions.",
            "Practice hygiene; avoid touching lesions then eyes."
        ],
        red_flags=[
            "Severe pain, spreading ulcers",
            "Fever or swollen lymph nodes",
            "Eye involvement"
        ],
    ),

    "Light Diseases and Disorders of Pigmentation": Advice(
        urgency="soon",
        contagious=False,
        base_see_doctor=False,
        summary=(
            "This category involves pigmentation changes or light-related issues. Many are not urgent, but evaluation helps confirm the cause."
        ),
        self_care=[
            "Use sunscreen daily; avoid excessive sun exposure.",
            "Avoid harsh bleaching/irritating products."
        ],
        red_flags=[
            "Rapidly changing pigmented lesion",
            "New pigment change with pain/bleeding",
            "Widespread sudden color change"
        ],
    ),

    "Lupus and other Connective Tissue diseases": Advice(
        urgency="urgent",
        contagious=False,
        base_see_doctor=True,
        summary=(
            "This category can be linked to autoimmune/connective tissue conditions. Please seek medical evaluation for proper diagnosis."
        ),
        self_care=[
            "Use sun protection (UV can trigger flares).",
            "Track symptoms (joint pain, fatigue, fevers) with dates."
        ],
        red_flags=[
            "Chest pain/shortness of breath",
            "New swelling, severe fatigue, or fevers",
            "Widespread rash with systemic symptoms"
        ],
    ),

    "Melanoma Skin Cancer Nevi and Moles": Advice(
        urgency="urgent",
        contagious=False,
        base_see_doctor=True,
        summary=(
            "This category includes moles and possible melanoma-related lesions. A dermatologist should assess suspicious changes promptly."
        ),
        self_care=[
            "Photograph the lesion and monitor ABCDE changes.",
            "Use sunscreen and avoid sunburn."
        ],
        red_flags=[
            "Asymmetry, irregular border, multiple colors",
            "Bleeding, ulceration, or rapid growth",
            "New changing mole"
        ],
    ),

    "Nail Fungus and other Nail Disease": Advice(
        urgency="monitor",
        contagious=True,  # nail fungus can spread
        base_see_doctor=False,
        summary=(
            "This category includes nail fungus and other nail disorders. Nail fungus is common but slow to treat. "
            "A clinician can confirm and guide treatment if needed."
        ),
        self_care=[
            "Keep nails trimmed; keep feet/hands dry.",
            "Avoid sharing nail clippers; disinfect tools.",
            "Wear breathable footwear and change socks daily."
        ],
        red_flags=[
            "Severe nail pain, swelling, or pus",
            "Diabetes/immunosuppression with worsening",
            "No improvement after trying basic care"
        ],
    ),

    "Poison Ivy Photos and other Contact Dermatitis": Advice(
        urgency="monitor",
        contagious=False,
        base_see_doctor=False,
        summary=(
            "This may be consistent with contact dermatitis (including poison ivy-type reactions). "
            "Identify and avoid the trigger; symptoms often improve with supportive care."
        ),
        self_care=[
            "Wash skin/clothes that may have contacted the irritant.",
            "Use gentle moisturizer; avoid scratching.",
            "Avoid new products until rash settles."
        ],
        red_flags=[
            "Face/eye/genital involvement",
            "Rapid spreading or severe swelling",
            "Signs of infection (pus, increasing pain)"
        ],
    ),

    "Psoriasis pictures Lichen Planus and related diseases": Advice(
        urgency="soon",
        contagious=False,
        base_see_doctor=True,
        summary=(
            "This category includes psoriasis/lichen planus-type inflammatory diseases. Diagnosis and treatment often benefit from a clinician."
        ),
        self_care=[
            "Moisturize daily; avoid harsh soaps.",
            "Track triggers (stress, infections, certain meds)."
        ],
        red_flags=[
            "Painful/swollen joints",
            "Widespread rapidly worsening rash",
            "Mouth/genital involvement (lichen planus can affect mucosa)"
        ],
    ),

    "Scabies Lyme Disease and other Infestations and Bites": Advice(
        urgency="soon",
        contagious=True,  # scabies contagious
        base_see_doctor=True,
        summary=(
            "This category includes infestations/bites (like scabies) and tick-related issues. "
            "Evaluation is important because treatment may be prescription-based and contacts may need treatment."
        ),
        self_care=[
            "Avoid close contact and wash bedding/clothes if scabies suspected.",
            "Use insect bite precautions; check for ticks after outdoor exposure."
        ],
        red_flags=[
            "Fever, severe headache, or expanding rash",
            "Multiple household members itching",
            "Rapidly spreading rash"
        ],
    ),

    "Seborrheic Keratoses and other Benign Tumors": Advice(
        urgency="monitor",
        contagious=False,
        base_see_doctor=False,
        summary=(
            "This category includes benign growths. Many are not urgent, but new or changing lesions should be checked."
        ),
        self_care=[
            "Avoid picking or irritating the lesion.",
            "Use sun protection to reduce further skin damage."
        ],
        red_flags=[
            "Bleeding, ulceration, or rapid change",
            "New lesion that looks very different from others"
        ],
    ),

    "Systemic Disease": Advice(
        urgency="urgent",
        contagious=False,
        base_see_doctor=True,
        summary=(
            "This category suggests the skin findings may relate to an internal/systemic condition. "
            "Medical evaluation is recommended."
        ),
        self_care=[
            "Track symptoms (fever, fatigue, weight change) and timing.",
            "Avoid new self-treatments until assessed."
        ],
        red_flags=[
            "Fever, severe weakness, or dehydration",
            "Rapidly spreading rash with systemic symptoms",
            "Trouble breathing or swelling"
        ],
    ),

    "Tinea Ringworm Candidiasis and other Fungal Infections": Advice(
        urgency="monitor",
        contagious=True,
        base_see_doctor=False,
        summary=(
            "This may be consistent with a fungal infection (tinea/ringworm/candidiasis). "
            "Keep the area clean and dry; avoid sharing towels/clothing. Many cases improve with appropriate antifungal treatment."
        ),
        self_care=[
            "Keep area dry; change clothes/socks daily if relevant.",
            "Avoid sharing towels/clothes; wash bedding regularly.",
            "If using OTC antifungal, use consistently as directed."
        ],
        red_flags=[
            "Scalp involvement (often needs prescription treatment)",
            "Face/genital involvement with worsening",
            "No improvement in 1–2 weeks"
        ],
    ),

    "Urticaria Hives": Advice(
        urgency="soon",
        contagious=False,
        base_see_doctor=False,
        summary=(
            "This may be consistent with hives (urticaria). Triggers can include infections, foods, medications, or stress. "
            "Most cases resolve, but recurrent or severe cases benefit from clinician guidance."
        ),
        self_care=[
            "Avoid known triggers if identified.",
            "Use gentle skincare; avoid heat which may worsen itching."
        ],
        red_flags=[
            "Lip/tongue swelling or trouble breathing (emergency)",
            "Hives lasting > 6 weeks (recurrent chronic hives)",
            "Severe widespread symptoms"
        ],
    ),

    "Vascular Tumors": Advice(
        urgency="soon",
        contagious=False,
        base_see_doctor=True,
        summary=(
            "This category includes vascular growths. Many are benign, but evaluation helps confirm type and treatment options."
        ),
        self_care=[
            "Avoid irritating or scratching the lesion.",
            "Take photos to track size/color changes."
        ],
        red_flags=[
            "Rapid growth or bleeding",
            "Painful lesion or ulceration"
        ],
    ),

    "Vasculitis Photos": Advice(
        urgency="urgent",
        contagious=False,
        base_see_doctor=True,
        summary=(
            "This category can include vasculitis, which may be linked to systemic inflammation. "
            "Medical evaluation is recommended promptly."
        ),
        self_care=[
            "Track symptoms (joint pain, fever, fatigue).",
            "Avoid new medications/products until assessed."
        ],
        red_flags=[
            "Fever, abdominal pain, blood in urine",
            "Widespread purplish spots or rapidly worsening rash",
            "Severe pain or ulceration"
        ],
    ),

    "Warts Molluscum and other Viral Infections": Advice(
        urgency="monitor",
        contagious=True,
        base_see_doctor=False,
        summary=(
            "This category includes viral lesions like warts or molluscum, which can spread by contact. "
            "Many resolve, but persistent lesions may need treatment."
        ),
        self_care=[
            "Avoid picking; cover lesions if rubbing against others.",
            "Don’t share towels/razors; wash hands after touching lesions."
        ],
        red_flags=[
            "Rapid spreading or painful lesions",
            "Face/genital involvement with worsening",
            "Immunocompromised with persistent outbreaks"
        ],
    ),
}


def get_recommendation_rich(label: str, confidence: float) -> dict:
    """
    Returns a rich dict for UI:
    {
      urgency, contagious, see_doctor,
      summary, self_care(list), red_flags(list),
      confidence_band
    }
    """
    advice = CLASS_ADVICE.get(label)
    band = _confidence_band(float(confidence))

    if advice is None:
        # Fallback for unexpected labels
        see_doctor = (band != "high")
        return {
            "urgency": "soon" if see_doctor else "monitor",
            "contagious": False,
            "see_doctor": see_doctor,
            "summary": (
                "This label is not mapped yet. If symptoms are new, worsening, painful, spreading, "
                "or your confidence score is low, consult a clinician."
            ),
            "self_care": [
                "Avoid new skincare products until symptoms settle.",
                "Take a clear photo in good lighting to track changes."
            ],
            "red_flags": [
                "Fever or feeling very unwell",
                "Rapidly spreading rash, severe pain, or facial/eye involvement",
                "Blistering or skin peeling"
            ],
            "confidence_band": band,
        }

    # Confidence adjustment:
    # - low confidence: see_doctor=True (safety)
    # - medium: keep baseline
    # - high: keep baseline
    see_doctor = advice.base_see_doctor or (band == "low")

    # If it's "urgent" category, always yes
    if advice.urgency == "urgent":
        see_doctor = True

    return {
        "urgency": advice.urgency,
        "contagious": advice.contagious,
        "see_doctor": see_doctor,
        "summary": advice.summary,
        "self_care": advice.self_care,
        "red_flags": advice.red_flags,
        "confidence_band": band,
    }