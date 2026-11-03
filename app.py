"""
AI-Powered Document Question-Answering Assistant (Project 21)
--------------------------------------------------------------
Retrieval: TF-IDF (scikit-learn) + cosine similarity over passages.
Answering: extractive (sentences copied from retrieved passages, with [doc_id] citations).
No LLM is used, so answers can never contain text that is not in the documents.

Run:  streamlit run app.py
Test: pytest -q
"""
import io
import re
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------------
# CONFIGURATION (documented defaults; adjustable in the UI sidebar)
# ----------------------------------------------------------------------------
REQUIRED_COLUMNS = ["doc_id", "title", "section", "text", "source", "license"]
MAX_QUERY_CHARS = 500
MAX_FILE_MB = 10
CHUNK_WORDS = 120          # passage size when splitting PDF / TXT files
CHUNK_OVERLAP = 20
DEFAULT_TOP_K = 3
DEFAULT_MIN_SCORE = 0.15   # below this the assistant says "not found"
NO_ANSWER_MSG = ("I could not find an answer to this question in the provided "
                 "documents. Try rephrasing, or check that the right documents are loaded.")

# Phrases typical of prompt-injection attempts (checked in queries AND passages)
INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
    r"disregard (all |any |the )?(previous|prior|above|earlier)? ?(instructions|prompts?|rules)",
    r"forget (everything|all|your) (instructions|rules|above)",
    r"(reveal|show|print|display|repeat) (me )?(your|the) (system|hidden|initial) (prompt|instructions)",
    r"you are now ",
    r"act as (an? )?(?!student)",
    r"pretend (to be|you are)",
    r"override (the )?(rules|instructions|safety)",
    r"jailbreak|developer mode|do anything now",
    r"api[_ ]?key|password dump|secret key",
]
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


# ----------------------------------------------------------------------------
# DATA VALIDATION & PREPARATION (no Streamlit code here)
# ----------------------------------------------------------------------------
class ValidationError(Exception):
    """Raised with a user-friendly message when input is unusable."""


def normalise_text(text: str) -> str:
    """Collapse whitespace and strip control characters."""
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP):
    """Split long text into overlapping word windows."""
    words = text.split()
    if not words:
        return []
    step = max(size - overlap, 1)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)
            if words[i:i + size]]


def prepare_collection(df: pd.DataFrame, source_name: str):
    """
    Validate and clean a passage table.
    Returns (clean_df, log) where log is a list of human-readable transformation notes.
    """
    log = [f"Source: {source_name}", f"Loaded {len(df)} rows"]
    if df.empty:
        raise ValidationError("The file is empty. Please upload a file with at least one passage.")

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValidationError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Expected columns: {', '.join(REQUIRED_COLUMNS)}.")

    df = df[REQUIRED_COLUMNS].astype(str)
    for col in REQUIRED_COLUMNS:
        df[col] = df[col].map(normalise_text)
    df = df.replace({"nan": "", "None": ""})

    before = len(df)
    df = df[df["text"].str.len() > 0]
    if before - len(df):
        log.append(f"Removed {before - len(df)} row(s) with empty text")

    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    if before - len(df):
        log.append(f"Removed {before - len(df)} duplicate passage(s)")

    dup_ids = df["doc_id"].duplicated().sum()
    if dup_ids:
        raise ValidationError(f"{dup_ids} duplicate doc_id value(s) found. Each passage needs a unique doc_id.")
    if (df["doc_id"] == "").any():
        raise ValidationError("Some rows have an empty doc_id.")

    short = (df["text"].str.split().str.len() < 5).sum()
    if short:
        log.append(f"Warning: {short} passage(s) have fewer than 5 words")

    if df.empty:
        raise ValidationError("No usable passages remain after cleaning.")

    df = df.reset_index(drop=True)
    log.append(f"Final collection: {len(df)} passages")
    return df, log


def passages_from_text(text: str, name: str, prefix: str) -> pd.DataFrame:
    """Turn raw text (from PDF or TXT) into a passage table."""
    chunks = chunk_text(normalise_text(text))
    rows = [{"doc_id": f"{prefix}-{i+1:03d}", "title": name, "section": f"Part {i+1}",
             "text": c, "source": name, "license": "user-supplied"}
            for i, c in enumerate(chunks)]
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def load_uploaded_file(name: str, data: bytes):
    """Read CSV / TXT / PDF bytes and return (clean_df, log)."""
    if len(data) == 0:
        raise ValidationError(f"'{name}' is empty.")
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise ValidationError(f"'{name}' is larger than {MAX_FILE_MB} MB.")
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    prefix = re.sub(r"[^A-Za-z0-9]", "", name.rsplit(".", 1)[0])[:6].upper() or "DOC"

    if ext == "csv":
        try:
            df = pd.read_csv(io.BytesIO(data))
        except Exception:
            raise ValidationError(f"'{name}' could not be read as a CSV file.")
        return prepare_collection(df, name)
    if ext == "txt":
        text = data.decode("utf-8", errors="ignore")
        if not text.strip():
            raise ValidationError(f"'{name}' contains no text.")
        return prepare_collection(passages_from_text(text, name, prefix), name)
    if ext == "pdf":
        from pypdf import PdfReader
        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception:
            raise ValidationError(f"'{name}' is not a readable PDF.")
        if not text.strip():
            raise ValidationError(f"'{name}' has no extractable text (it may be a scanned image).")
        return prepare_collection(passages_from_text(text, name, prefix), name)
    raise ValidationError(f"Unsupported file type '.{ext}'. Use CSV, TXT or PDF.")


# ----------------------------------------------------------------------------
# SAFETY CHECKS
# ----------------------------------------------------------------------------
def detect_injection(text: str) -> bool:
    """True if text looks like an instruction aimed at the AI rather than a question."""
    return bool(_INJECTION_RE.search(text or ""))


def validate_query(query: str):
    """Return (ok, message). Message explains why a query was rejected."""
    q = (query or "").strip()
    if not q:
        return False, "Please type a question before searching."
    if len(q) > MAX_QUERY_CHARS:
        return False, f"Question is too long ({len(q)} characters). Limit: {MAX_QUERY_CHARS}."
    if len(re.findall(r"[A-Za-z0-9]", q)) < 2:
        return False, "Please enter a question containing words."
    if detect_injection(q):
        return False, ("This looks like an instruction to the assistant (possible prompt injection), "
                       "not a question about the documents. It was blocked and not processed.")
    return True, ""


# ----------------------------------------------------------------------------
# RETRIEVAL + EXTRACTIVE ANSWER
# ----------------------------------------------------------------------------
class Retriever:
    """TF-IDF passage retriever. Deterministic: same data -> same index."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                          sublinear_tf=True)
        corpus = (self.df["title"] + " " + self.df["section"] + " " + self.df["text"]).tolist()
        self.matrix = self.vectorizer.fit_transform(corpus)
        self.flagged = np.array([detect_injection(t) for t in self.df["text"]])

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> pd.DataFrame:
        qv = self.vectorizer.transform([query])
        scores = cosine_similarity(qv, self.matrix).ravel()
        scores[self.flagged] = 0.0            # never use passages that contain injected instructions
        order = np.argsort(-scores, kind="stable")[:top_k]
        out = self.df.iloc[order].copy()
        out["score"] = scores[order]
        out["rank"] = range(1, len(out) + 1)
        return out

    def answer(self, query: str, top_k: int = DEFAULT_TOP_K, min_score: float = DEFAULT_MIN_SCORE):
        """
        Returns dict: answered(bool), answer(str), sources(DataFrame), best_score(float).
        Abstains when the best passage score is below min_score.
        """
        hits = self.search(query, top_k)
        best = float(hits["score"].max()) if len(hits) else 0.0
        if best < min_score:
            return {"answered": False, "answer": NO_ANSWER_MSG, "sources": hits, "best_score": best}

        qv = self.vectorizer.transform([query])
        cands = []
        for _, row in hits[hits["score"] >= min_score].iterrows():
            for sent in re.split(r"(?<=[.!?])\s+", row["text"]):
                if len(sent.split()) < 3:
                    continue
                s = float(cosine_similarity(qv, self.vectorizer.transform([sent]))[0, 0])
                cands.append((s, row["rank"], row["doc_id"], sent))
        cands = [c for c in cands if c[0] > 0]
        if not cands:
            return {"answered": False, "answer": NO_ANSWER_MSG, "sources": hits, "best_score": best}
        top = sorted(cands, key=lambda c: -c[0])[:2]
        top = sorted(top, key=lambda c: (c[1], c[3]))      # read in passage order
        text = " ".join(f"{s} [{d}]" for _, _, d, s in top)
        return {"answered": True, "answer": text, "sources": hits, "best_score": best}


# ----------------------------------------------------------------------------
# EVALUATION
# ----------------------------------------------------------------------------
def evaluate(retriever: Retriever, eval_df: pd.DataFrame, top_k: int, min_score: float):
    """
    eval_df columns: question, expected_doc_id (blank = the documents do NOT contain the answer).
    Metrics: Hit@k, MRR (answerable questions); abstention accuracy (unanswerable questions).
    """
    if not {"question", "expected_doc_id"}.issubset(eval_df.columns):
        raise ValidationError("Evaluation file needs columns: question, expected_doc_id")
    rows = []
    for _, r in eval_df.iterrows():
        q = str(r["question"])
        exp = "" if pd.isna(r["expected_doc_id"]) else str(r["expected_doc_id"]).strip()
        res = retriever.answer(q, top_k, min_score)
        ids = res["sources"]["doc_id"].tolist()
        if exp:
            rank = ids.index(exp) + 1 if exp in ids else 0
            correct = res["answered"] and rank > 0
            rows.append({"question": q, "expected": exp, "type": "answerable",
                         "rank_of_expected": rank, "answered": res["answered"],
                         "best_score": round(res["best_score"], 3), "correct": correct})
        else:
            rows.append({"question": q, "expected": "(none)", "type": "unanswerable",
                         "rank_of_expected": 0, "answered": res["answered"],
                         "best_score": round(res["best_score"], 3),
                         "correct": not res["answered"]})
    res_df = pd.DataFrame(rows)
    ans = res_df[res_df["type"] == "answerable"]
    una = res_df[res_df["type"] == "unanswerable"]
    metrics = {
        "answerable_questions": len(ans),
        "unanswerable_questions": len(una),
        "hit_at_k": float((ans["rank_of_expected"] > 0).mean()) if len(ans) else float("nan"),
        "mrr": float(np.mean([1 / r if r > 0 else 0 for r in ans["rank_of_expected"]])) if len(ans) else float("nan"),
        "answered_correctly": float(ans["correct"].mean()) if len(ans) else float("nan"),
        "abstention_accuracy": float(una["correct"].mean()) if len(una) else float("nan"),
    }
    return res_df, metrics


def data_fingerprint(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


# ----------------------------------------------------------------------------
# STREAMLIT USER INTERFACE
# ----------------------------------------------------------------------------
def main():
    import streamlit as st
    import os

    st.set_page_config(page_title="Document Q&A Assistant", page_icon="📄", layout="wide")
    st.title("📄 Document Question-Answering Assistant")
    st.caption("Answers come only from the loaded documents and always show their sources. "
               "If the documents don't contain the answer, the assistant says so.")

    @st.cache_resource(show_spinner="Building search index...")
    def get_retriever(fingerprint: str, df: pd.DataFrame):
        return Retriever(df)

    # ---- Sidebar: data + settings
    with st.sidebar:
        st.header("1. Documents")
        mode = st.radio("Source", ["Sample dataset", "Upload my own files"])
        df, log = None, []
        try:
            if mode == "Sample dataset":
                path = "data/sample_documents.csv"
                if not os.path.exists(path):
                    raise ValidationError(f"Sample file '{path}' not found. Upload the data folder next to app.py.")
                with open(path, "rb") as f:
                    df, log = load_uploaded_file("sample_documents.csv", f.read())
            else:
                files = st.file_uploader("CSV (doc_id,title,section,text,source,license), TXT or PDF",
                                         type=["csv", "txt", "pdf"], accept_multiple_files=True)
                if files:
                    parts = []
                    for uf in files:
                        part, plog = load_uploaded_file(uf.name, uf.getvalue())
                        parts.append(part)
                        log += plog
                    df = pd.concat(parts, ignore_index=True)
                    if df["doc_id"].duplicated().any():
                        raise ValidationError("doc_id values clash between files. Rename files or ids.")
        except ValidationError as e:
            st.error(str(e))
            df = None

        st.header("2. Settings")
        top_k = st.slider("Passages to retrieve (top-k)", 1, 8, DEFAULT_TOP_K)
        min_score = st.slider("Minimum relevance score", 0.0, 0.6, DEFAULT_MIN_SCORE, 0.01,
                              help="Below this score the assistant answers 'not found'.")
        if df is not None:
            st.success(f"{len(df)} passages loaded")
            with st.expander("Data preparation log"):
                st.write("\n".join(f"- {l}" for l in log))

    if df is None:
        st.info("Load the sample dataset or upload documents in the sidebar to begin.")
        return

    retriever = get_retriever(data_fingerprint(df), df)
    if "history" not in st.session_state:
        st.session_state.history = []

    tab_ask, tab_docs, tab_eval = st.tabs(["💬 Ask", "📚 Documents", "🧪 Evaluation"])

    # ---- Ask tab
    with tab_ask:
        query = st.text_input("Your question", placeholder="e.g. What is the late fee for tuition payment?")
        if st.button("Search", type="primary"):
            ok, msg = validate_query(query)
            if not ok:
                st.warning(msg)
                st.session_state.history.append({
                    "time": datetime.now().strftime("%H:%M:%S"), "question": query,
                    "status": "blocked/invalid", "answer": msg, "sources": "", "best_score": 0})
            else:
                try:
                    with st.spinner("Searching documents..."):
                        res = retriever.answer(query.strip(), top_k, min_score)
                    if res["answered"]:
                        st.success("Answer")
                        st.write(res["answer"])
                    else:
                        st.warning(res["answer"])
                    st.caption(f"Best relevance score: {res['best_score']:.3f} "
                               f"(threshold {min_score:.2f})")
                    st.subheader("Sources")
                    for _, r in res["sources"].iterrows():
                        with st.expander(f"[{r['doc_id']}] {r['title']} – {r['section']}  "
                                         f"(score {r['score']:.3f})"):
                            st.write(r["text"])
                            st.caption(f"Source: {r['source']} | License: {r['license']}")
                    st.bar_chart(res["sources"].set_index("doc_id")["score"])
                    st.session_state.history.append({
                        "time": datetime.now().strftime("%H:%M:%S"), "question": query,
                        "status": "answered" if res["answered"] else "not found",
                        "answer": res["answer"],
                        "sources": ", ".join(res["sources"]["doc_id"]),
                        "best_score": round(res["best_score"], 3)})
                except Exception:
                    st.error("Something went wrong while searching. Please try again or reload the documents.")

        if st.session_state.history:
            st.divider()
            st.subheader("Session history")
            hist = pd.DataFrame(st.session_state.history)
            st.dataframe(hist, use_container_width=True)
            st.download_button("⬇️ Export history (CSV)", hist.to_csv(index=False).encode("utf-8"),
                               "qa_history.csv", "text/csv")

    # ---- Documents tab
    with tab_docs:
        st.dataframe(df, use_container_width=True)
        flagged = int(retriever.flagged.sum())
        if flagged:
            st.warning(f"{flagged} passage(s) contain instruction-like text and are excluded from answers: "
                       + ", ".join(df.loc[retriever.flagged, "doc_id"]))
        else:
            st.caption("No instruction-like (injection) text detected in the passages.")

    # ---- Evaluation tab
    with tab_eval:
        st.write("Upload a CSV with columns `question,expected_doc_id` (leave expected_doc_id blank for "
                 "questions the documents cannot answer), or use the sample set.")
        eval_up = st.file_uploader("Evaluation CSV (optional)", type=["csv"], key="eval")
        eval_df = None
        try:
            if eval_up is not None:
                eval_df = pd.read_csv(eval_up)
            elif mode == "Sample dataset" and os.path.exists("data/eval_questions.csv"):
                eval_df = pd.read_csv("data/eval_questions.csv")
        except Exception:
            st.error("Could not read the evaluation file.")
        if eval_df is not None and st.button("Run evaluation"):
            try:
                res_df, m = evaluate(retriever, eval_df, top_k, min_score)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(f"Hit@{top_k}", f"{m['hit_at_k']:.0%}")
                c2.metric("MRR", f"{m['mrr']:.2f}")
                c3.metric("Answered correctly", f"{m['answered_correctly']:.0%}")
                c4.metric("Correct 'not found'", f"{m['abstention_accuracy']:.0%}")
                st.dataframe(res_df, use_container_width=True)
                st.caption("Results are for this small test set only and do not predict performance "
                           "on other documents or question styles.")
                wrong = res_df[~res_df["correct"]]
                if len(wrong):
                    st.subheader("Errors to inspect")
                    st.dataframe(wrong, use_container_width=True)
                st.download_button("⬇️ Export evaluation (CSV)", res_df.to_csv(index=False).encode("utf-8"),
                                   "evaluation_results.csv", "text/csv")
            except ValidationError as e:
                st.error(str(e))


if __name__ == "__main__":
    main()
