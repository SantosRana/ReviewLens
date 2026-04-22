# app.py
import streamlit as st
import json
import sys
from config import ROOT

sys.path.append(str(ROOT / "src"))

from src.absa import load_model
from src.agent import run_agent_turn

st.set_page_config(page_title="ReviewLens", page_icon="🔍", layout="wide")

@st.cache_resource
def get_model():
    """Load and cache the KC-ELECTRA model instance."""
    return load_model()

# Initialize model once and reuse cached instance
kc_model = get_model()


# Session state
if "history" not in st.session_state:
    st.session_state.history = []
if "messages_display" not in st.session_state:
    st.session_state.messages_display = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
def calculate_scores(result: dict) -> dict:
    """Calculate weighted aspect-based sentiment scores."""

    # Define weights for each aspect
    weights = {
        "FOOD": 0.40,
        "SERVICE": 0.25,
        "AMBIENCE": 0.20,
        "PRICE": 0.15,
    }

    scores = {}
    weighted_total = 0
    used_weight = 0

    # Iterate through aspects and compute scores
    for aspect, weight in weights.items():
        if aspect not in result:
            continue  # skip if aspect not present

        pos = result[aspect]["positive"]
        neg = result[aspect]["negative"]
        total = pos + neg

        if total == 0:
            score = 48  # mild penalty if aspect not mentioned
        else:
            # Normalize sentiment to 0–100 scale
            score = int(((pos - neg) / total + 1) / 2 * 100)

        scores[aspect] = score
        weighted_total += score * weight
        used_weight += weight

    # Compute overall weighted score (fallback = 50 if no aspects used)
    scores["OVERALL"] = int(weighted_total / used_weight) if used_weight else 50

    return scores



aspect_names = {"FOOD": "Food", "PRICE": "Value", "SERVICE": "Service", "AMBIENCE": "Ambience"}

def get_pros_cons(result: dict) -> tuple:
    """Get positive and negative points."""
    pros, cons = [], []
    
    for aspect, data in result.items():
        if aspect not in aspect_names or not isinstance(data, dict):
            continue
            
        name = aspect_names[aspect]
        pos, neg = data.get("positive", 0), data.get("negative", 0)
        total = pos + neg
        
        if total == 0:
            continue
            
        ratio = pos / total
        
        # Dynamic thresholds based on total review count
        min_mentions = max(2, total // 3)  # Scale threshold with review count
        
        if ratio >= 0.8 and pos >= min_mentions:
            pros.append((aspect, f"✅ Excellent {name.lower()} ({pos} positive reviews)"))
        elif ratio >= 0.7 and pos >= max(1, min_mentions // 2):
            pros.append((aspect, f"👍 Good {name.lower()}"))
        
        if ratio <= 0.3 and neg >= min_mentions:
            cons.append((aspect, f"❌ Poor {name.lower()} ({neg} negative reviews)"))
        elif ratio <= 0.4 and neg >= max(1, min_mentions // 2):
            cons.append((aspect, f"👎 Weak {name.lower()}"))
    
    return pros[:2], cons[:2]


def get_grade(score: int) -> tuple:
    if score >= 90: return "S", "#fbbf24"
    if score >= 80: return "A", "#22c55e"
    if score >= 65: return "B", "#3b82f6"
    if score >= 50: return "C", "#f97316"
    return "D", "#ef4444"


def render_restaurant_result(result, container=None):
    """Render single restaurant result."""
    target = container if container else st
    
    scores = calculate_scores(result)
    overall = scores.get("OVERALL", 0)
    grade, grade_color = get_grade(overall)
    name = result.get("restaurant", "Restaurant")
    total = result.get("total_reviews", 0)

    # Big score display
    target.markdown(
        f"""
        <div style='text-align:center;padding:20px;
                    background:linear-gradient(135deg,#1e293b,#334155);
                    border-radius:12px;color:white;margin-bottom:15px;'>
            <div style='font-size:14px;opacity:0.7;'>OVERALL</div>
            <div style='font-size:48px;font-weight:bold;color:{grade_color};'>{overall}</div>
            <div style='font-size:24px;color:{grade_color};'>Grade {grade}</div>
            <div style='font-size:11px;opacity:0.5;margin-top:5px;'>{name} • {total} reviews</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Aspect scores in one row
    aspect_labels = {"FOOD": "🍽️", "SERVICE": "🛎️", "AMBIENCE": "🏠", "PRICE": "💰"}
    cols = target.columns(4)
    for col_idx, (aspect, emoji) in enumerate(aspect_labels.items()):
        score = scores.get(aspect)
        with cols[col_idx]:
            st.metric(emoji, f"{score}%" if score is not None else "N/A")

    # Pros & Cons
    target.divider()
    pros, cons = get_pros_cons(result)
    pc_cols = target.columns(2)
    with pc_cols[0]:
        st.markdown("**Why Go**")
        if pros:
            for _, p in pros:
                st.markdown(f"<small>{p}</small>", unsafe_allow_html=True)
        else:
            st.caption("No standout positives")
    with pc_cols[1]:
        st.markdown("**Why Skip**")
        if cons:
            for _, c in cons:
                st.markdown(f"<small>{c}</small>", unsafe_allow_html=True)
        else:
            st.caption("No major red flags")

    # Download
    target.divider()
    target.download_button(
        "📥 Export JSON",
        json.dumps(result, ensure_ascii=False),
        file_name=f"{name}.json",
        use_container_width=True,
        key=f"download_{name}_{id(result)}"  # Unique key
    )


def render_comparison_result(result, container=None):
    """Render comparison result."""
    target = container if container else st
    
    r1, r2 = result[0], result[1]
    s1, s2 = calculate_scores(r1), calculate_scores(r2)
    g1, gc1 = get_grade(s1["OVERALL"])
    g2, gc2 = get_grade(s2["OVERALL"])

    # Winner banner
    if s1["OVERALL"] > s2["OVERALL"]:
        target.success(f"🏆 {r1.get('restaurant', 'R1')} wins!")
    elif s2["OVERALL"] > s1["OVERALL"]:
        target.success(f"🏆 {r2.get('restaurant', 'R2')} wins!")
    else:
        target.info("🤝 Tie!")

    # Side-by-side score cards
    score_cols = target.columns(2)
    restaurants_data = [
        (r1, s1, g1, gc1, score_cols[0]),
        (r2, s2, g2, gc2, score_cols[1])
    ]
    
    for idx, (r, s, g, gc, col) in enumerate(restaurants_data):
        with col:
            name = r.get('restaurant', f"R{idx + 1}")
            st.markdown(
                f"""
                <div style='text-align:center;padding:15px;
                            background:#f8fafc;border-radius:8px;
                            border:2px solid #e2e8f0;'>
                    <div style='font-size:12px;color:#64748b;'>{name}</div>
                    <div style='font-size:36px;font-weight:bold;color:{gc};'>{s["OVERALL"]}</div>
                    <div style='font-size:16px;'>Grade {g}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for aspect, emoji in [("FOOD", "🍽️"), ("SERVICE", "🛎️"), ("AMBIENCE", "🏠"), ("PRICE", "💰")]:
                val = s.get(aspect)
                if val:
                    st.caption(f"{emoji} {val}%")

    # Pros summary per restaurant
    target.divider()
    p1, _ = get_pros_cons(r1)
    p2, _ = get_pros_cons(r2)
    comp_cols = target.columns(2)
    with comp_cols[0]:
        st.markdown(f"**{r1.get('restaurant', 'R1')}**")
        best1 = ", ".join([aspect_names[a] for a, _ in p1[:2]]) if p1 else "Nothing specific"
        st.caption(f"Best at: {best1}")
    with comp_cols[1]:
        st.markdown(f"**{r2.get('restaurant', 'R2')}**")
        best2 = ", ".join([aspect_names[a] for a, _ in p2[:2]]) if p2 else "Nothing specific"
        st.caption(f"Best at: {best2}")


def render_review_result(result, container=None):
    """Render single or multiple review sentiment analysis."""
    target = container if container else st

    target.markdown("**Review Sentiment Analysis**")

    reviews = result.get("reviews_data", [result])

    aspect_labels = {
        "FOOD": "🍽️ Food",
        "PRICE": "💰 Value",
        "AMBIENCE": "🏠 Ambience",
        "SERVICE": "🛎️ Service"
    }

    for idx, review in enumerate(reviews, start=1):
        if len(reviews) > 1:
            target.divider()
            target.markdown(f"### Review {idx}")

        if "text" in review:
            with target.expander("View original review"):
                st.write(
                    review["text"][:200] + "..."
                    if len(review["text"]) > 200
                    else review["text"]
                )

        for aspect, label in aspect_labels.items():
            if aspect not in review:
                continue

            aspect_data = review[aspect]

            # If this is a per-review sentiment label
            sentiment = aspect_data.get("sentiment")

            if sentiment:
                if sentiment == "positive":
                    pos = 1
                    neg = 0
                elif sentiment == "negative":
                    pos = 0
                    neg = 1
                else:
                    pos = 0
                    neg = 0

            # Fallback for aggregated counts
            else:
                pos = aspect_data.get("positive", 0)
                neg = aspect_data.get("negative", 0)

            total = pos + neg

            if total > 0:
                target.markdown(f"**{label}**")

                bar_cols = target.columns(2)

                with bar_cols[0]:
                    st.progress(pos / total if total > 0 else 0, text=f"👍 Positive")

                with bar_cols[1]:
                    st.progress(neg / total if total > 0 else 0, text=f"👎 Negative")

            else:
                target.caption(f"{label}: Not mentioned")


def show_result(result, container=None):
    """Route to appropriate renderer based on result type."""
    target = container if container else st
    
    # Explicit type detection - check for review analysis FIRST
    if isinstance(result, dict):
        if result.get("is_review_analysis", False):
            render_review_result(result, target)
            return
        elif result.get("restaurant") is not None:
            render_restaurant_result(result, target)
            return
        elif result.get("error"):
            target.error(result["error"])
            return
    
    # Comparison is a list of 2 results
    if isinstance(result, list) and len(result) == 2:
        render_comparison_result(result, target)
        return
    
    # Fallback: if it has aspect keys, treat as review
    if isinstance(result, dict) and any(key in result for key in ["FOOD", "PRICE", "SERVICE", "AMBIENCE"]):
        render_review_result(result, target)
        return
    
    target.error("Unknown result type")
    target.json(result)  # Show raw result for debugging


# Layout
st.title("🔍 ReviewLens")
st.caption("AI-Powered Korean Restaurant Review Analyzer")

# App description
st.markdown("""
Welcome to **ReviewLens** — your AI companion for exploring Korean restaurant reviews.  
This app automatically collects customer feedback, analyzes it by key aspects (Food, Service, Value, Ambience), 
and summarizes the strengths and weaknesses of each restaurant.  

### How it works:
- **Chat** 💬: Ask about a restaurant, compare two places, or provide your own review text.  
- **Sentiment Analysis** 📊: The AI model classifies reviews into positive, negative, or not mentioned for each aspect.  
- **Pros & Cons** ✅❌: Get a quick snapshot of why you might want to visit or avoid a restaurant.  
- **Comparison Mode** 🏆: See two restaurants side by side, with scores, grades, and highlights.
""")

left_col, right_col = st.columns([1, 1])

# Left - Chat
with left_col:
    st.subheader("💬 Chat")
    
    chat_container = st.container(height=550)
    with chat_container:
        for msg in st.session_state.messages_display:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
    
    user_input = st.chat_input(
        "🍽️ Curious about a Korean restaurant? Ask me to analyze reviews, compare places, or just chat..."
    )

    if user_input:
        # CLEAR PREVIOUS RESULT IMMEDIATELY when starting new analysis
        st.session_state.last_result = None
        
        st.session_state.messages_display.append({"role": "user", "content": user_input})

        progress_bar = st.progress(0, text="Starting analysis...")
        status_text = st.empty()

        # Placeholder in the RIGHT column for live result updates
        with right_col:
            live_result_slot = st.empty()
            # 🆕 Show "analyzing..." placeholder immediately
            with live_result_slot.container():
                st.info("🔍 Analyzing... please wait")

        def on_progress(current: int, total: int, label: str = ""):
            progress_bar.progress(current / total, text=label or f"Processing {current}/{total}...")
            status_text.caption(f"✅ {current} / {total} reviews processed")

        def on_restaurant_done(partial_result):
            """Called by agent.py after each restaurant finishes — updates right col live."""
            st.session_state.last_result = partial_result
            with live_result_slot.container():
                show_result(partial_result)

        updated_history, answer, result = run_agent_turn(
            user_input,
            st.session_state.history,
            kc_model,
            progress_callback=on_progress,
            result_callback=on_restaurant_done,
        )

        progress_bar.empty()
        status_text.empty()

        st.session_state.history = updated_history
        st.session_state.messages_display.append({"role": "assistant", "content": answer})
        
        # ALWAYS update last_result if we got a result, even if callback wasn't triggered
        if result is not None:
            st.session_state.last_result = result
            # If callback wasn't called (e.g., for direct reviews), update display now
            with live_result_slot.container():
                show_result(result)
        
        st.rerun()
        
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.history = []
            st.session_state.messages_display = []
            st.session_state.last_result = None
            st.rerun()
    with c2:
        if st.session_state.last_result and st.button("💬 Chat Only", use_container_width=True):
            st.session_state.last_result = None
            st.rerun()


# Right - Compact Scores & Tips
with right_col:
    st.subheader("📊 Scores")

    if st.session_state.last_result is None:
        st.info("Results appear here after analysis")
    else:
        show_result(st.session_state.last_result)