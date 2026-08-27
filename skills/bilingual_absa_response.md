# Skill: Bilingual ABSA Response

## Purpose
Generate natural conversational responses while preserving the linguistic
integrity of Korean restaurant entities and validated review evidence.

## Primary Responsibilities

1. **Language Alignment**
  - Respond in the primary language of the user's query.
   - DEFAULT TO ENGLISH for brief greetings, ambiguous inputs, or short non-Korean prompts (e.g., "hi", "hy", "hello").
   - Only respond in Korean if the user explicitly writes their prompt in Korean (e.g., "안녕하세요", "분석해줘").

2. **Entity Preservation**
   - ALWAYS preserve restaurant names, dish names, and meaningful Korean food terminology in their original Hangul script.
   - Examples: `삼청동수제비`, `수제비`, `물냉면`, `밑반찬`.
   - Do not romanize (e.g., avoid writing "Sujebi" alone) or translate entity names unless providing an optional parenthetical explanation on first mention (e.g., `삼청동수제비 (Samcheongdong Sujebi)`).

3. **Evidence Integrity**
   - Treat validated evidence quotes as immutable text payloads.
   - NEVER translate, rewrite, paraphrase, shorten, expand, or correct typos in validated evidence quotes.
   - Render evidence exactly as supplied by the analysis tool, enclosed in verbatim double quotation marks.
   - Example: `"국물이 깊고 맛있어요"`

4. **Contextual Explanation**
   - When the response language is not Korean, explain the meaning of the Korean evidence in the user's target language immediately following or preceding the quote.
   - Always present the raw Korean quote alongside its translation/explanation so evidence remains transparent and verifiable.

5. **Missing or Null Evidence Handling**
   - If an aspect score exists but contains no positive/negative evidence quotes (or an empty list `[]`), state the sentiment finding clearly without fabricating or guessing quotes.

## Important Boundary

- This skill controls **response language, formatting, and presentation policy**.
- `ABSAOutputValidator` is solely responsible for **deterministic evidence validation in Python**.
- This skill MUST NEVER be treated as a substitute for Python-level evidence-grounding verification.