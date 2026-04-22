# 🔍 ReviewLens
> See Beyond the Star Rating — AI-Powered Korean Restaurant Review Analyzer

ReviewLens is a lightweight agentic AI system for analyzing Korean restaurant reviews with Aspect-Based Sentiment Analysis (ABSA).

This project is a continuation of my earlier benchmark study comparing Classical Machine Learning, Transformer-based models, and LLMs for Korean restaurant review ABSA. After benchmarking different approaches, I selected the strongest practical combination — a fine-tuned KcELECTRA model for structured aspect prediction and Qwen 2.5 for reasoning, summarization, and conversational interaction.

The result is a lightweight local application that can:
- Analyze single or multiple Korean reviews (sentiment bars per aspect)
- Fetch real restaurant reviews from Naver Blog Search (scores + pros/cons + summary)
- Compare restaurants side by side (scores + pros/cons + summary)
- Provide conversational summaries and insights
- Run locally without requiring a GPU

---

# 🧠 Project Background

This project builds directly on my Korean restaurant ABSA benchmark project, where I compared:

- Classical ML models using TF-IDF + Logistic Regression
- Transformer-based models such as KcELECTRA
- Small local LLMs such as Qwen 2.5

The benchmark showed that:

- Classical ML was fast but struggled with contextual understanding
- LLMs were flexible but slower and less consistent for structured prediction
- KcELECTRA achieved the best balance of accuracy, speed, and efficiency

Because of this, ReviewLens uses:
- **KcELECTRA** for aspect detection and sentiment classification
- **Qwen 2.5** for chat responses, explanation, summarization, and agent reasoning
- **Naver Blog API** to retrieve real Korean restaurant reviews

---

# ✨ Features

## Review Analysis

Analyze one or many Korean restaurant reviews and extract sentiment by aspect. Displays progress bars for positive/negative sentiment per aspect.

Example input:

```text
음식은 맛있었지만 직원이 불친절했고 가격이 조금 비쌌어요.
```

## Restaurant Search

Search for a Korean restaurant and automatically fetch recent reviews using the Naver Blog Search API. Displays overall score, grade, aspect scores, pros/cons, and a short summary.

Example:

```text
봉피양 강남점 분석해줘, analyze this restaurant 한옥집 and give your suggestion
```

## Restaurant Comparison

Compare two restaurants side by side across all aspects. Displays winner banner, side-by-side score cards, and best-at highlights.

Example:

```text
봉피양이랑 교촌치킨 비교해줘, compare 삼청동수제비 and 우래옥 
```


## Conversational Chat Interface

Users can ask questions naturally instead of clicking filters or menus.

Examples:

```text
분위기가 좋은 카페형 식당 알려줘,
Hi, Can you recommend some best restaurant on Seoul?,
How can you help me?
```

---

# 🏗️ System Architecture

```
┌─────────────────────────────────────────┐
│              User Input                 │
│    (Chat / Restaurant / Comparison)     │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│             Streamlit                   │
│  • Chat UI  • Progress Bars  • Scores   │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│            Agent Router (Qwen 2.5)      │
│  Intent Classification → Tool Selection │
└──────────────┬──────────────────────────┘
               ↓
         ┌─────┴─────┐
         ↓           ↓
┌─────────────┐  ┌──────────────────────┐
│   Tool:     │  │   Tool:              │
│ Naver Blog  │  │ ABSA Inference       │
│   Search    │  │ (KcELECTRA)          │
│             │  │                      │
│ • Fetch 20  │  │ • Aspect Detection   │
│   reviews   │  │ • Sentiment Classify │
└──────┬──────┘  └──────────┬───────────┘
       │                    │
       └────────┬───────────┘
                ↓
┌─────────────────────────────────────────┐
│         Result Aggregation              │
│  • Scores  • Pros/Cons  • Summaries     │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│      Response Generation (Qwen 2.5)     │
│ Natural language explanation in English │
└─────────────────────────────────────────┘
```

**Data Flow:**
1. User query → Agent classifies intent (chat vs. restaurant vs. comparison)
2. Restaurant queries trigger Naver Blog Search API
3. Raw reviews fed to fine-tuned KcELECTRA for ABSA
4. Structured aspect counts aggregated into scores
5. Qwen 2.5 generates conversational summary
6. Streamlit renders structured tables + chat response side-by-side

---

# 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Qwen 2.5 7B via Ollama |
| ABSA Model | Fine-tuned KcELECTRA |
| Review Source | Naver Blog Search API |
| Frontend | Streamlit |
| Backend | Python |
| NLP Libraries | PyTorch, Transformers, scikit-learn |

---

# 📂 Project Structure

```text
ReviewLens/
├── src/                                 # Source code package
│   ├── kc_electra/                      # Korean ELECTRA model sub-package
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── decode_prediction.py
│   │   └── helper_utils.py
│   ├── absa.py                          # ABSA orchestrator
│   ├── agent.py                         # Agent controller / state manager
│   └── naver.py                         # Naver Place scraper
├── weights/                             # Pretrained model artifacts (not versioned in git)    
│   └── kc_electra/
│       ├── kc_electra.pt
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── vocab.txt
│       ├── special_tokens_map.json
│       ├── aspects.json
│       └── thresholds.json
├── app.py                          # Application entry point(streamlit)
├── config.py
├── README.md
├── requirements.txt
└── .gitignore

```

---

# ⚙️ Setup

## 1. Clone the Repository

```bash
git clone <your-repo-url>
cd ReviewLens
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Install Ollama

```bash
ollama pull qwen2.5:7b-instruct
```

Then make sure Ollama is running locally.

## 4. Configure Naver API Keys

Create a `config.py` file:

```python
NAVER_CLIENT_ID = "your_client_id"
NAVER_CLIENT_SECRET = "your_client_secret"
```

## 5. Run the Application

```bash
streamlit run app.py
```

---

## 6. Setup Requirements

Before running the project, ensure:

- Model weights are placed in the weights/ directory
- Ollama is running locally for Qwen inference
- Naver API keys are generated

## 🎥 Demo
[Watch Demo Video]()


---

# 📈 Future Improvements

- **Dish-level analysis** — identify specific menu items and their sentiment
- **Weighted aspect scoring** — user-customizable priority weights per aspect
- **Sentiment timeline trends** — track sentiment drift over time
- **Multi-source aggregation** — Kakao Map, Google Reviews
- **Session memory** — remember user preferences across the session

---

# 🙏 Acknowledgments

- [beomi/KcELECTRA](https://github.com/beomi/KcELECTRA) for the base Korean language model
- [Qwen](https://github.com/QwenLM/Qwen) for the local LLM
- [Naver Developers](https://developers.naver.com/) for the Blog Search API

# Pretrained KcELECTRA Model
Due to file size limitations, pretrained model weight is not included in this repository.

You can download them here: [Download models here](https://drive.google.com/drive/folders/15_2_i6g6-1LIQcNbElXbHTtILDAv2KYH?preview)

After downloading, place them in the weights/ folder.
