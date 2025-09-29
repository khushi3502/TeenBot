import os, re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import streamlit as st
from dotenv import load_dotenv, find_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables with encoding fallback (handles UTF-8/UTF-16 .env files)
_env_path = find_dotenv()
if _env_path:
    try:
        load_dotenv(_env_path, encoding="utf-8")
    except UnicodeDecodeError:
        load_dotenv(_env_path, encoding="utf-16")
else:
    load_dotenv()

PASTEL_CSS = """
<style>
:root { --pastel-bg:#f6f7ff; --text:#000; --user:#1877f2; --bot:#f1f0f0; }
html,body,[data-testid="stAppViewContainer"], *{ color:var(--text)!important; }
[data-testid="stAppViewContainer"]{ background:var(--pastel-bg)!important; }
[data-testid="stSidebar"]{ background:#f7fbff!important; }

.chat-row{ display:flex; gap:8px; margin:8px 0; align-items:flex-end; }
.chat-row.user{ justify-content:flex-end; }
.chat-row.bot{ justify-content:flex-start; }

.avatar{ width:28px; height:28px; border-radius:50%; display:flex; align-items:center; justify-content:center; background:#ffe2ec; border:1px solid rgba(0,0,0,.05); }
.avatar span{ font-size:18px; }

.bubble{ max-width:78%; padding:10px 12px; border-radius:18px; border:1px solid rgba(0,0,0,.06); box-shadow:0 1px 2px rgba(0,0,0,.05); line-height:1.35; }
.user-bubble{ background:var(--user); color:#fff; }
.bot-bubble{ background:var(--bot); color:#000; }

.small-note{font-size:.9rem;opacity:.95;}
a,.stButton>button,.stMarkdown p{color:#000!important;}

.stTextInput input{ color:#000!important; background:#ffffff!important; border:1px solid #d0d7de!important; border-radius:12px!important; padding:12px 14px!important; }
.stTextInput label{ color:#000!important; }
.stButton>button{ background:#fff!important; border:1px solid rgba(0,0,0,.15)!important; border-radius:10px!important; }
</style>
"""
st.markdown(PASTEL_CSS, unsafe_allow_html=True)

# ---------------- Memory (TF-IDF retrieval) ----------------
@dataclass
class ConversationMemory:
    max_entries: int = 200
    sentences: List[str] = field(default_factory=list)
    _vec: Optional[TfidfVectorizer] = None

    def add(self, text: str) -> None:
        text = (text or "").strip()
        if not text: return
        self.sentences.append(text)
        if len(self.sentences) > self.max_entries:
            self.sentences = self.sentences[-self.max_entries:]
        self._vec = None

    def _ensure(self):
        if self._vec is None and self.sentences:
            self._vec = TfidfVectorizer(ngram_range=(1,2))
            self._vec.fit(self.sentences)

    def snippet(self, query: str, k: int = 5) -> str:
        if not self.sentences: return ""
        self._ensure()
        if not self._vec: return ""
        q = self._vec.transform([query])
        m = self._vec.transform(self.sentences)
        scores = cosine_similarity(q, m).flatten()
        pairs = list(zip(self.sentences, scores))
        pairs.sort(key=lambda x: x[1], reverse=True)
        lines = [f"- {t}" for t, s in pairs[:k] if s > 0.05]
        return ("Earlier you mentioned:\n" + "\n".join(lines)) if lines else ""

# ---------------- Mood Classifiers ----------------
def clean_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s,.!?'’]+", " ", s)
    return re.sub(r"\s+", " ", s)

class LocalMoodClassifier:
    def __init__(self, labels: List[str] = ["negative","neutral","positive"]):
        self.labels = labels
        self.vec = TfidfVectorizer(ngram_range=(1,2), max_features=20000)
        self.clf = LogisticRegression(max_iter=200)

    def fit_demo(self):
        texts = [
            "I feel happy and excited!", "Today was okay, nothing much happened.",
            "I'm really stressed and overwhelmed.", "This is fine.",
            "I’m anxious about school.", "Feeling good about my day!",
            "Not great, kind of down.", "I'm angry", "I feel calm",
            "I'm worried about exams", "I feel proud of myself"
        ]
        labels = ["positive","neutral","negative","neutral","negative","positive","negative","negative","neutral","negative","positive"]
        X = self.vec.fit_transform([clean_text(t) for t in texts])
        self.clf.fit(X, labels)

    def predict(self, text: str) -> Tuple[str, List[Tuple[str, float]]]:
        X = self.vec.transform([clean_text(text)])
        probs = self.clf.predict_proba(X)[0]
        classes = self.clf.classes_
        pairs = list(zip(classes, probs.tolist()))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[0][0], pairs

@st.cache_resource
def load_hf_pipeline():
    try:
        from transformers import pipeline
        clf = pipeline("text-classification",
                       model="bhadresh-savani/distilbert-base-uncased-emotion",
                       return_all_scores=True)
        return clf
    except Exception as e:
        return None

def hf_predict(text: str) -> Tuple[str, List[Tuple[str, float]]]:
    pipe = load_hf_pipeline()
    if not pipe:
        return "neutral", [("neutral", 1.0)]
    out = pipe(text)[0]
    out.sort(key=lambda d: d["score"], reverse=True)
    top = out[0]["label"]
    pairs = [(d["label"], float(d["score"])) for d in out]
    # map emotions -> coarse mood
    emo_to_mood = {"sadness":"negative","fear":"negative","anger":"negative","disgust":"negative",
                   "joy":"positive","surprise":"neutral","neutral":"neutral"}
    mood = emo_to_mood.get(top, "neutral")
    return mood, pairs

# ---------------- LLM providers ----------------
def call_openai(system: str, messages: List[dict], model: str = "gpt-4o-mini") -> str:
    from openai import OpenAI
    key = os.getenv("OPENAI_API_KEY")
    if not key: return "Please set OPENAI_API_KEY."
    client = OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model=model, temperature=0.7,
        messages=[{"role":"system","content":system}] + messages
    )
    return resp.choices[0].message.content

def call_anthropic(system: str, messages: List[dict], model: str = "claude-3-5-sonnet-20241022") -> str:
    import anthropic
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key: return "Please set ANTHROPIC_API_KEY."
    client = anthropic.Anthropic(api_key=key)
    # Flatten to a user message for Claude
    user_parts = []
    for m in messages:
        if m["role"] == "user":
            user_parts.append({"type":"text","text": m["content"]})
        elif m["role"] == "assistant":
            user_parts.append({"type":"text","text": f"(assistant said) {m['content']}"})
    msg = client.messages.create(
        model=model, max_tokens=600, temperature=0.7,
        system=system, messages=[{"role":"user","content": user_parts}]
    )
    return "".join([b.text for b in msg.content if hasattr(b, "text")])

# ---------------- Safety ----------------
CRISIS = {"suicide","self-harm","kill myself","hurt myself","ending it","no reason to live"}
RESOURCES = [
    "The Live Love Laugh Foundation – Blog\nhttps://www.thelivelovelaughfoundation.org/blog",
    "Heart It Out – Mental Health Blogs\nhttps://heartitout.in/blogs/",
    "National Alliance on Mental Illness (NAMI) – Blogs\nhttps://www.nami.org/blogs/",
    "CMHLP (Centre for Mental Health Law & Policy) – Blogs\nhttps://cmhlp.org/blogs/",
    "MHFA India – Blogs\nhttps://www.mhfaindia.com/blogs-list",
]
BASE_PROMPT = (
    "You are a warm, supportive teen mental health chatbot. Be empathetic, non-judgmental, "
    "inclusive, and culturally sensitive. Do not diagnose or give medical instructions. "
    "Encourage reaching out to trusted adults or professionals when appropriate. "
    "Use clear, simple language."
)

def looks_like_crisis(t: str) -> bool:
    t = t.lower()
    return any(k in t for k in CRISIS)

# ---------------- App state ----------------
if "history" not in st.session_state:
    st.session_state.history = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()
if "local_clf" not in st.session_state:
    lc = LocalMoodClassifier()
    lc.fit_demo()
    st.session_state.local_clf = lc

st.title("VibeBuddy ✨")
st.caption("I’m here to listen. I don’t store or share your messages. This isn’t a replacement for professional help.")

with st.sidebar:
    st.markdown("### Settings")
    provider = st.selectbox("LLM Provider", ["OpenAI GPT", "Anthropic Claude"])
    mood_source = st.selectbox("Mood Detection", ["Local TF-IDF (scikit-learn)", "Hugging Face Emotion"])
    st.markdown("### Resources")
    st.write("\n".join(RESOURCES))

if not st.session_state.history:
    st.session_state.history.append({"role":"assistant","content":"Hey! How are you feeling today?"})

for m in st.session_state.history:
    is_user = m["role"] == "user"
    row_cls = "chat-row user" if is_user else "chat-row bot"
    bub_cls = "user-bubble bubble" if is_user else "bot-bubble bubble"
    left_avatar = "<div class=\"avatar\"><span>🤖</span></div>" if not is_user else ""
    right_avatar = "<div class=\"avatar\"><span>🧑</span></div>" if is_user else ""
    st.markdown(f'<div class="{row_cls}">{left_avatar}<div class="{bub_cls}">{m["content"]}</div>{right_avatar}</div>', unsafe_allow_html=True)

user_text = st.text_input("Type your message", key="msg", placeholder="Type here…")

def detect_mood(text: str) -> Tuple[str, List[Tuple[str,float]]]:
    if mood_source.startswith("Local"):
        label, pairs = st.session_state.local_clf.predict(text)
        # map to negative/neutral/positive already
        return label, pairs
    else:
        return hf_predict(text)

def tone_hint(mood: str) -> str:
    if mood == "negative":
        return "be extra gentle, validate feelings, suggest coping strategies and resources"
    if mood == "positive":
        return "celebrate wins and reinforce healthy habits"
    return "ask open, supportive questions and reflect back feelings"

def generate_reply(user_msg: str) -> str:
    st.session_state.memory.add(user_msg)
    mem = st.session_state.memory.snippet(user_msg, k=5)
    mood, scores = detect_mood(user_msg)

    safety_prefix = ""
    if looks_like_crisis(user_msg):
        safety_prefix = (
            "Thank you for sharing something so tough. Your safety matters. "
            "If you feel at risk of harming yourself, please reach out now:\n" + "\n".join(RESOURCES) + "\n\n"
        )

    system = f"{BASE_PROMPT}\nDetected mood: {mood}. Please {tone_hint(mood)}."
    if mem: system += "\n" + mem

    msgs = []
    for h in st.session_state.history[-20:]:
        msgs.append({"role":h["role"], "content":h["content"]})
    msgs.append({"role":"user","content":user_msg})

    if provider == "OpenAI GPT":
        text = call_openai(system, msgs)
    else:
        text = call_anthropic(system, msgs)

    return safety_prefix + text

col1, col2 = st.columns([4,1])
with col1:
    send = st.button("Send", use_container_width=True)
with col2:
    clear = st.button("Clear", use_container_width=True)

if send and user_text.strip():
    st.session_state.history.append({"role":"user","content":user_text.strip()})
    reply = generate_reply(user_text.strip())
    st.session_state.history.append({"role":"assistant","content":reply})
    st.rerun()

if clear:
    st.session_state.history = []
    st.session_state.memory = ConversationMemory()
    st.rerun()

st.markdown('<div class="small-note">Eat Healthy stay happy !!!.</div>', unsafe_allow_html=True)