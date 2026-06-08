from __future__ import annotations

def validate_section(section_name: str, content: str | None) -> bool:
    bad_phrases = [
        "no analyst", "not provided", "no narrative",
        "observations are", "placeholder", "n/a"
    ]
    if not content or len(content.strip()) < 80:
        raise ValueError(f"REPORT BLOCKED: {section_name} is too short or empty.")
    for phrase in bad_phrases:
        if phrase.lower() in content.lower():
            raise ValueError(f"REPORT BLOCKED: {section_name} contains placeholder text: '{phrase}'")
    return True
