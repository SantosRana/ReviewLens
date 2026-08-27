# app.py
import html
import logging
import streamlit as st
from src.agent import ReviewLensAgent

# Configure root logger to output INFO level logs to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="ReviewLens | Restaurant Review Intelligence",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------
# Design system
# ------------------------------------------------------------------
# Palette drawn from Goryeo celadon ceramics (muted jade) and the
# cinnabar ink of a Korean dojang (personal seal stamp). The seal is
# the app's signature: every piece of evidence the agent surfaces has
# been checked against the source text in Python, so a stamp motif —
# marked "검증" ("verified") — is used wherever that guarantee applies.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@500;700&family=IBM+Plex+Sans+KR:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg: #EEF2EA;
    --surface: #FBFBF6;
    --surface-alt: #E3EAE0;
    --ink: #202821;
    --ink-soft: #55604F;
    --cinnabar: #AC3428;
    --cinnabar-soft: #C65344;
    --celadon-deep: #7C9A82;
    --brass: #B08A4E;
    --line: #D2DACB;
}

html, body, .stApp {
    background-color: var(--bg);
    background-image:
        radial-gradient(circle at 15% 10%, rgba(124,154,130,0.06) 0, transparent 45%),
        radial-gradient(circle at 85% 70%, rgba(176,138,78,0.06) 0, transparent 50%);
    font-family: 'IBM Plex Sans KR', sans-serif;
    color: var(--ink);
}

h1, h2, h3, h4 { font-family: 'Noto Serif KR', serif; color: var(--ink); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--surface-alt);
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] h3 {
    font-size: 0.95rem;
    letter-spacing: .02em;
}

/* Buttons */
.stButton > button {
    background: var(--ink);
    color: var(--surface);
    border: 1px solid var(--ink);
    border-radius: 3px;
    font-family: 'IBM Plex Sans KR', sans-serif;
    letter-spacing: .01em;
}
.stButton > button:hover {
    background: var(--cinnabar);
    border-color: var(--cinnabar);
    color: var(--surface);
}

/* Metrics */
div[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 0.85rem 1rem 0.7rem 1rem;
}
div[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--cinnabar);
    font-size: 1.35rem;
}
div[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Sans KR', sans-serif;
    color: var(--ink-soft);
    text-transform: uppercase;
    letter-spacing: .05em;
    font-size: 0.7rem;
}

/* Progress bars (aspect sentiment) */
div[data-testid="stProgress"] > div > div {
    background-color: var(--line) !important;
}
div[data-testid="stProgress"] > div > div > div {
    background-color: var(--celadon-deep) !important;
}

/* Chat */
div[data-testid="stChatMessage"] {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 6px;
}

/* Hero */
.hero {
    padding: 0.25rem 0 1.75rem 0;
    border-bottom: 1px solid var(--line);
    margin-bottom: 1.5rem;
}
.hero-eyebrow {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: .09em;
    text-transform: uppercase;
    color: var(--ink-soft);
    margin-bottom: 0.6rem;
}
.hero-title {
    font-family: 'Noto Serif KR', serif;
    font-size: 2.6rem;
    font-weight: 700;
    margin: 0 0 0.55rem 0;
    color: var(--ink);
}
.hero-sub {
    font-family: 'IBM Plex Sans KR', sans-serif;
    font-size: 1.02rem;
    line-height: 1.6;
    color: var(--ink-soft);
    max-width: 640px;
    margin-bottom: 1.1rem;
}
.hero-chips { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.chip {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    background: var(--surface);
    border: 1px solid var(--line);
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    color: var(--ink);
}

/* Seal / dojang badge — the signature element */
.seal {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border: 2px solid var(--cinnabar);
    color: var(--cinnabar);
    background: rgba(172, 52, 40, 0.05);
    font-family: 'Noto Serif KR', serif;
    font-weight: 700;
    letter-spacing: .04em;
    border-radius: 2px;
    transform: rotate(-4deg);
    line-height: 1;
    cursor: default;
}
.seal-sm { font-size: 0.68rem; padding: 0.18rem 0.42rem; }
.seal-xs { font-size: 0.58rem; padding: 0.1rem 0.32rem; }

/* Evidence quote cards */
.quote {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    border-radius: 4px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.6rem;
    border-left: 3px solid transparent;
}
.quote p { margin: 0; font-size: 0.92rem; line-height: 1.5; color: var(--ink); }
.quote-pro { background: rgba(124, 154, 130, 0.14); border-left-color: var(--celadon-deep); }
.quote-con { background: rgba(172, 52, 40, 0.08); border-left-color: var(--cinnabar); }

.section-divider { border: none; border-top: 1px solid var(--line); margin: 1.75rem 0; }
</style>
""", unsafe_allow_html=True)


def seal(label: str = "검증", title: str = "Checked against the source review in Python", size: str = "sm") -> str:
    """Return an inline dojang-style stamp used to mark verified content."""
    return f'<span class="seal seal-{size}" title="{html.escape(title)}">{html.escape(label)}</span>'


def render_quote(text: str, kind: str = "pro") -> None:
    """Render a single grounded evidence quote as a stamped card."""
    css_class = "quote-pro" if kind == "pro" else "quote-con"
    st.markdown(
        f'<div class="quote {css_class}">{seal(size="xs")}<p>&ldquo;{html.escape(text)}&rdquo;</p></div>',
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------------
if "agent" not in st.session_state:
    st.session_state.agent = ReviewLensAgent()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_data" not in st.session_state:
    st.session_state.active_data = None


# ------------------------------------------------------------------
# Sidebar: System Controls & Integrity Architecture
# ------------------------------------------------------------------
with st.sidebar:
    st.title("🍲 ReviewLens")
    st.caption("AI-Powered Naver Blog Review ABSA Engine")
    st.markdown("---")

    # Analysis Integrity Section
    st.markdown(f"### Analysis Integrity {seal(size='xs')}", unsafe_allow_html=True)
    st.markdown("""
- **Untrusted Data Isolation**: Treats Naver reviews strictly as raw text, blocking prompt injection
- **Strict Schema Enforcement**: Every extraction is checked against a strict Pydantic schema
- **Verbatim Grounding**: Verifies evidence quotes against the source review
- **Hallucination Filtering**: Aspects that can't be traced back to real text are filtered out
""")

    st.markdown("---")
    st.markdown("### Try asking")
    st.markdown("""
- **Search & analyze** — `Search 봉피양 강남점 and analyze`
- **Korean Analysis**: `삼청동수제비 분석해줘`
- **English Analysis**: `Analyze Samcheongdong Sujebi`
- **Follow-up insight** — `서비스 상태는 어때?`
- **Side-by-side** — `Compare 봉피양이랑 삼청동수제비`
- **Recommendation** — `Which is the best restaurant for a date in Seoul?`
""")

    st.markdown("---")
    if st.button("Reset session chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.active_data = None
        st.session_state.agent = ReviewLensAgent()
        st.rerun()


# ------------------------------------------------------------------
# Hero / Intro
# ------------------------------------------------------------------
st.markdown(f"""
<div class="hero">
    <div class="hero-eyebrow">Restaurant Review Intelligence {seal()}</div>
    <h1 class="hero-title">ReviewLens</h1>
    <p class="hero-sub">
        Read what a menu actually tastes like, before you go. Name a restaurant and
        ReviewLens pulls its Naver Blog reviews, scores food, price, service, and ambience,
        and shows the exact lines each score came from — nothing invented, everything traceable.
    </p>
    <div class="hero-chips">
        <span class="chip">봉피양 강남점 분석해줘</span>
        <span class="chip">서비스 상태는 어때?</span>
        <span class="chip">봉피양이랑 삼청동수제비 비교해줘</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Main UI Layout
# ------------------------------------------------------------------
# Defensive Data Check for Dashboard Rendering
REQUIRED_DASHBOARD_KEYS = {
    "overall_score",
    "aspect_scores",
    "pros",
    "cons",
    "parsed_count",
    "total_reviews",
    "parse_rate",
    "retention_rate"
}

active_data = st.session_state.active_data

if (
    isinstance(active_data, dict)
    and REQUIRED_DASHBOARD_KEYS.issubset(active_data.keys())
    and active_data.get("status") == "success"
):
    restaurant_name = active_data.get("restaurant_name", "Analyzed Restaurant")
    st.markdown(f"## Analytics Dashboard — **{restaurant_name}**")

    # Top Metric Bar: 5 Metrics Exposing Extraction & Validation Integrity
    col1, col2, col3, col4, col5 = st.columns(5)

    parsed_count = active_data.get("parsed_count", 0)
    total_reviews = active_data.get("total_reviews", 0)
    parse_rate = active_data.get("parse_rate", 0.0)
    retention_rate = active_data.get("retention_rate", 0.0)
    overall_score = active_data.get("overall_score", 0.0)

    with col1:
        st.metric(label="Overall rating", value=f"{overall_score:.2f} / 5.0")
    with col2:
        st.metric(label="Reviews processed", value=f"{parsed_count} / {total_reviews}")
    with col3:
        st.metric(label="Parse rate", value=f"{parse_rate:.1f}%")
    with col4:
        st.metric(label="Extraction retention", value=f"{retention_rate:.1f}%")
    with col5:
        st.metric(label="Evidence grounding", value="100.0%")  # 100% of rendered evidence is verified verbatim in Python

    st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)

    # Aspect Progress Bars (Normalized 0.0 to 1.0 floats preserving precision)
    st.subheader("Aspect Sentiment")
    aspect_scores = active_data.get("aspect_scores", {})

    asp_cols = st.columns(4)
    aspect_keys = ["FOOD", "PRICE", "SERVICE", "AMBIENCE"]

    for i, asp in enumerate(aspect_keys):
        with asp_cols[i]:
            data = aspect_scores.get(asp, {})
            pos_rate = data.get("score", 0.0)  # Positive ratio percentage
            mentions = data.get("total_mentions", 0)

            # Safe float progression between 0.0 and 1.0
            progress_value = min(max(pos_rate / 100.0, 0.0), 1.0)

            st.markdown(f"**{asp.title()}**")
            st.progress(progress_value)
            st.caption(f"{pos_rate:.1f}% positive · {mentions} mentions")

    st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)

    # Grounded Evidence Quotes Panel
    col_pro, col_con = st.columns(2)
    with col_pro:
        st.markdown(f"#### Key Pros {seal(size='xs')}", unsafe_allow_html=True)
        pros = active_data.get("pros", [])
        if pros:
            for quote in pros:
                render_quote(quote, kind="pro")
        else:
            st.info("No grounded positive evidence extracted.")

    with col_con:
        st.markdown(f"#### Key Cons {seal(size='xs')}", unsafe_allow_html=True)
        cons = active_data.get("cons", [])
        if cons:
            for quote in cons:
                render_quote(quote, kind="con")
        else:
            st.info("No grounded negative evidence extracted.")

    st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)


# ------------------------------------------------------------------
# Conversational Interface Section
# ------------------------------------------------------------------
st.subheader("Chat with ReviewLens")

# Render previous chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process User Query
if prompt := st.chat_input("Ask about a restaurant or inspect aspect details..."):
    # Render user prompt in UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Execute Agent Reasoning with Exception Safety Guard
    with st.chat_message("assistant"):
        with st.spinner("Analyzing Naver Blog reviews and validating aspect extractions..."):
            try:
                # Agent encapsulated execution (Agent -> Guardrails -> Groq -> Validator)
                agent_result = st.session_state.agent.run(prompt)

                response_text = agent_result.get("response_text", "No response generated.")
                tool_data = agent_result.get("tool_data")

                # Defensive update: Set active_data only if payload satisfies schema dictionary requirements
                if (
                    isinstance(tool_data, dict)
                    and REQUIRED_DASHBOARD_KEYS.issubset(tool_data.keys())
                ):
                    st.session_state.active_data = tool_data

                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

                # Re-render dashboard if valid analysis tool data returned
                if isinstance(tool_data, dict) and REQUIRED_DASHBOARD_KEYS.issubset(tool_data.keys()):
                    st.rerun()

            except Exception as exc:
                logger.exception("ReviewLens Agent execution failed")
                st.error("An unexpected error occurred while processing Naver Blog reviews. Please try again.")