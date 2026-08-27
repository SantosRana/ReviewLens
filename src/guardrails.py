import re
import logging
import unicodedata
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class GuardrailReason(str, Enum):
    PASSED = "PASSED"
    EMPTY_INPUT = "EMPTY_INPUT"
    INPUT_TOO_LONG = "INPUT_TOO_LONG"

    INSTRUCTION_OVERRIDE = "instruction_override"
    PROMPT_EXTRACTION = "prompt_extraction"
    ROLE_MANIPULATION = "role_manipulation"
    JAILBREAK_ATTEMPT = "jailbreak_attempt"
    CODE_EXECUTION = "code_execution"

    REVIEW_DATA_INJECTION = "review_data_injection"
    REVIEW_DATA_TOO_LONG = "review_data_too_long"


class RiskLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GuardrailResult(BaseModel):
    is_safe: bool = Field(
        description="True if the query passed security inspection"
    )
    reason: GuardrailReason = Field(
        description="Structured category code describing outcome"
    )
    risk: RiskLevel = Field(
        description="Security risk assessment level"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="User-facing response if blocked"
    )


class SecurityGuardrails:

    def __init__(
        self,
        max_input_length: int = 1000,
        max_review_length: int = 5000,
    ):
        self.max_input_length = max_input_length
        self.max_review_length = max_review_length

        # ---------------------------------------------------------
        # Direct user-query security patterns
        # ---------------------------------------------------------

        self.user_query_patterns = {
            GuardrailReason.INSTRUCTION_OVERRIDE: [
                r"\bignore\b.{0,80}\b(previous|prior|above|earlier|system)\b.{0,80}\binstructions?\b",
                r"\b(disregard|forget|override|bypass)\b.{0,80}\b(instructions?|rules?|prompt|policy)\b",
                r"\bignore\s+all\s+rules\b",
                r"\bignore\s+everything\s+(above|before)\b",
                r"\byour\s+new\s+(task|instruction|role)\s+is\b",
                r"\bfrom\s+now\s+on\b.{0,50}\b(you\s+must|you\s+are)\b",
            ],

            GuardrailReason.PROMPT_EXTRACTION: [
                r"\b(reveal|show|print|display|give\s+me|output|repeat)\b"
                r".{0,60}\b(system|hidden|secret|internal)\b"
                r".{0,40}\b(prompt|instructions?|rules?)\b",

                r"\bshow\s+me\s+your\s+(system\s+)?instructions\b",
                r"\bwhat\s+are\s+your\s+(system\s+)?instructions\b",
                r"\brepeat\s+the\s+system\s+prompt\b",
            ],

            GuardrailReason.ROLE_MANIPULATION: [
                r"\byou\s+are\s+now\b.{0,50}\b(a|an|the)\b",
                r"\bact\s+as\s+(the\s+)?(system|developer|admin|root)\b",
                r"\bpretend\s+you\s+are\b.{0,50}\b(system|developer|admin|root)\b",
                r"\broleplay\s+as\s+(a\s+)?(system|developer|admin)\b",
            ],

            GuardrailReason.JAILBREAK_ATTEMPT: [
                r"\bjailbreak\b",
                r"\bdeveloper\s+mode\b",
                r"\bdan\s+mode\b",
                r"\bdo\s+anything\s+now\b",
                r"\bno\s+restrictions?\b",
                r"\bwithout\s+(any\s+)?rules\b",
            ],

            GuardrailReason.CODE_EXECUTION: [
                r"<script\b",
                r"\beval\s*\(",
                r"\bexec\s*\(",
                r"\b__import__\s*\(",
                r"\bimport\s+(os|sys|subprocess|shutil)\b",
                r"\bos\.system\s*\(",
                r"\bsubprocess\.(run|call|Popen)\s*\(",
            ],
        }

        # ---------------------------------------------------------
        # Review-data injection patterns
        #
        # More conservative than user-query patterns because
        # legitimate review text can contain unusual language.
        # ---------------------------------------------------------

        self.review_injection_patterns = [
            r"\bignore\s+(all\s+)?previous\s+instructions\b",
            r"\bignore\s+(the\s+)?system\s+prompt\b",
            r"\bdisregard\s+(all\s+)?previous\s+instructions\b",
            r"\bforget\s+(all\s+)?previous\s+instructions\b",
            r"\byou\s+are\s+now\s+(a|an|the)\b",
            r"\bact\s+as\s+(the\s+)?(system|developer|admin)\b",
            r"\breveal\s+(your\s+)?system\s+prompt\b",
            r"\bshow\s+(me\s+)?your\s+(hidden\s+)?instructions\b",
            r"\bdeveloper\s+mode\b",
            r"\bjailbreak\b",
            r"<\s*(system|assistant|developer|tool|instruction)\s*>",
            r"\[\s*(SYSTEM|ASSISTANT|DEVELOPER|TOOL)\s*\]",
        ]

        self.compiled_user_patterns = {
            reason: re.compile(
                "|".join(patterns),
                re.IGNORECASE,
            )
            for reason, patterns in self.user_query_patterns.items()
        }

        self.compiled_review_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.review_injection_patterns
        ]

    # =============================================================
    # Shared normalization
    # =============================================================

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize Unicode and remove ASCII control characters while
        preserving ordinary textual content.
        """
        normalized = unicodedata.normalize("NFKC", text.strip())

        return re.sub(
            r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]",
            "",
            normalized,
        )

    # =============================================================
    # Boundary 1: Direct user query
    # =============================================================

    def validate_user_input(self, user_input: str) -> GuardrailResult:
        """
        Security boundary for direct user input.

        This method runs BEFORE the user query is sent to the LLM.
        It handles malicious instructions, prompt extraction,
        role manipulation, jailbreaks, and obvious code execution
        requests.
        """

        if not isinstance(user_input, str):
            return GuardrailResult(
                is_safe=False,
                reason=GuardrailReason.EMPTY_INPUT,
                risk=RiskLevel.HIGH,
                error_message="Invalid input type.",
            )

        if not user_input.strip():
            return GuardrailResult(
                is_safe=False,
                reason=GuardrailReason.EMPTY_INPUT,
                risk=RiskLevel.NONE,
                error_message="Input query cannot be empty.",
            )

        cleaned_text = self._normalize_text(user_input)

        if len(cleaned_text) > self.max_input_length:
            logger.warning(
                "[Guardrails] Category=INPUT_TOO_LONG length=%d",
                len(cleaned_text),
            )

            return GuardrailResult(
                is_safe=False,
                reason=GuardrailReason.INPUT_TOO_LONG,
                risk=RiskLevel.LOW,
                error_message=(
                    f"Input is too long ({len(cleaned_text)} characters). "
                    f"Limit requests to {self.max_input_length} characters."
                ),
            )

        for reason, regex in self.compiled_user_patterns.items():
            if regex.search(cleaned_text):
                logger.warning(
                    "[Guardrails] Blocked user query category=%s",
                    reason.value,
                )

                return GuardrailResult(
                    is_safe=False,
                    reason=reason,
                    risk=RiskLevel.HIGH,
                    error_message=(
                        "Your request was flagged by security guardrails. "
                        "Please rephrase without instruction overrides, "
                        "system-prompt requests, or execution commands."
                    ),
                )

        return GuardrailResult(
            is_safe=True,
            reason=GuardrailReason.PASSED,
            risk=RiskLevel.NONE,
        )

    # =============================================================
    # Boundary 2: External / user-provided review data
    # =============================================================

    def validate_review_data(self, review_text: str) -> Optional[str]:
        """
        Treat user-provided review text as untrusted DATA.

        Unlike validate_user_input(), this method does not try to
        interpret the review as a user instruction.

        It:
        1. Normalizes Unicode.
        2. Removes control characters.
        3. Enforces a review-size limit.
        4. Detects obvious prompt-injection markers.
        5. Returns the original textual content otherwise so that
           ABSA evidence matching remains intact.

        Returns:
            Sanitized review text if safe.
            None if the review should be rejected.
        """

        if not isinstance(review_text, str):
            return None

        if not review_text.strip():
            return None

        cleaned_text = self._normalize_text(review_text)

        if not cleaned_text:
            return None

        if len(cleaned_text) > self.max_review_length:
            logger.warning(
                "[Guardrails] Review rejected: length=%d",
                len(cleaned_text),
            )
            return None

        for regex in self.compiled_review_patterns:
            if regex.search(cleaned_text):
                logger.warning(
                    "[Guardrails] Suspicious review data rejected"
                )
                return None

        return cleaned_text

    # =============================================================
    # Display sanitization
    # =============================================================

    def sanitize_for_display(self, text: str) -> str:
        """
        Prepares text safely for Streamlit rendering and logging.
        This is presentation sanitization, not security validation.
        """

        if not text:
            return ""

        normalized = self._normalize_text(text)

        sanitized = re.sub(
            r"\[SYSTEM\]|\[USER\]|\[ASSISTANT\]|\[TOOL\]",
            "",
            normalized,
            flags=re.IGNORECASE,
        )

        return sanitized.strip()