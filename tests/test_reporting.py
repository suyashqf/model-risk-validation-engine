from __future__ import annotations

import pytest

from mrm_os.quality_gate import validate_section


def test_quality_gate_rejects_short_or_placeholder_sections() -> None:
    with pytest.raises(ValueError, match="too short"):
        validate_section("Conceptual Soundness", "Too short.")

    with pytest.raises(ValueError, match="placeholder"):
        validate_section(
            "Overrides",
            "This placeholder section contains enough characters but should be rejected by the quality gate.",
        )


def test_quality_gate_accepts_specific_llm_narrative() -> None:
    assert validate_section(
        "Stress Test",
        "The stressed loss result is directionally consistent with the configured macroeconomic shock and portfolio sensitivity. The ECL movement should be reviewed against observed credit migration before approval.",
    )
