# =====================================================
# 🌟 Gemini 2.5 Pro All-in-One AI Career Suite (Complete)
# =====================================================
# Full app: Chat, Resume, LinkedIn, Job-fit, ATS/Plagiarism,
# Data summarization, PDF Q&A, Image Gen, Interview Simulator,
# QuestionPaper->PDF Answer Generator (PDF/JPG/PNG/TXT) with
# Tesseract OCR + Gemini Vision OCR fallback (auto).
#
# Run in Google Colab for easiest setup.
# =====================================================

# -----------------------
# Install dependencies (Colab-friendly)
# -----------------------
import sys, subprocess, os

def sh(cmd):
    subprocess.run(cmd, shell=True, check=False)

# Install Python packages and Tesseract (Colab / Debian)
sh("pip install -q google-generativeai gradio PyPDF2 Pillow pytesseract pandas scikit-learn openpyxl")
sh("apt-get update -qq && apt-get install -y -qq tesseract-ocr libtesseract-dev")

# -----------------------
# Imports
# -----------------------
import time
import random
import threading
import difflib
import base64
from io import BytesIO

import pandas as pd
from PIL import Image
import pytesseract
import gradio as gr
from PyPDF2 import PdfReader
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Optional FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

# -----------------------
# Configuration / API Key
# -----------------------
# <-- Your API key (as provided) -->
API_KEY = "AIzaSyB5roew7_J1jrcDra4Td8l-M9LBFduXyeE"
if API_KEY and API_KEY.strip():
    try:
        genai.configure(api_key=API_KEY)
    except Exception:
        pass

# Directory for generated images
GENERATED_DIR = os.path.abspath("generated_images")
os.makedirs(GENERATED_DIR, exist_ok=True)

# -----------------------
# Utilities (file reading, safe)
# -----------------------
def safe_read_file_obj(file_obj):
    """
    Returns (bytes_data, filename) for Gradio file objects or file-like objects.
    """
    if file_obj is None:
        return None, None
    filename = getattr(file_obj, "name", None) or getattr(file_obj, "filename", None)
    file_like = getattr(file_obj, "file", None) or file_obj
    try:
        file_like.seek(0)
    except Exception:
        pass
    data = None
    try:
        data = file_like.read()
        if data is None and filename and os.path.exists(filename):
            with open(filename, "rb") as f:
                data = f.read()
    except Exception:
        try:
            if filename and os.path.exists(filename):
                with open(filename, "rb") as f:
                    data = f.read()
        except Exception:
            data = None
    return data, filename

# -----------------------
# PDF/Text/Image extraction
# -----------------------
def extract_text_from_pdf(file_obj) -> str:
    """Extract text from PDF bytes or Gradio file object."""
    try:
        data, filename = safe_read_file_obj(file_obj)
        if data is None:
            return "⚠️ No data in PDF file."
        reader = PdfReader(BytesIO(data))
        pages = []
        for p in reader.pages:
            t = p.extract_text()
            if t:
                pages.append(t)
        return "\n".join(pages).strip()
    except Exception as e:
        return f"⚠️ Failed to read PDF: {e}"

def extract_text_from_txt(file_obj) -> str:
    data, filename = safe_read_file_obj(file_obj)
    if data is None:
        return "⚠️ No data in text file."
    try:
        return data.decode("utf-8", errors="ignore").strip()
    except Exception:
        return str(data)

def ocr_image_tesseract(file_obj) -> str:
    """Primary OCR using Tesseract."""
    try:
        data, filename = safe_read_file_obj(file_obj)
        if data is None:
            return "⚠️ No image data."
        img = Image.open(BytesIO(data)).convert("RGB")
        # Optional config -- treat as printed text
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        return f"⚠️ Tesseract OCR failed: {e}"

def gemini_vision_ocr_fallback(image_bytes: bytes) -> str:
    """
    Fallback OCR using Gemini generative model: embed image as base64 and ask to transcribe.
    Note: This is a best-effort fallback and may consume more quota.
    """
    try:
        if not image_bytes:
            return "⚠️ No image bytes for Gemini OCR."
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        # Prompt the model to extract raw text from the base64 image
        prompt = (
            "You are an OCR assistant. The user will provide an image encoded in base64. "
            "Extract and return only the textual content in the image. Do NOT add commentary.\n\n"
            f"IMAGE_BASE64:\n{b64}\n\nReturn only the extracted text."
        )
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt)
        return resp.text or ""
    except Exception as e:
        return f"⚠️ Gemini OCR failed: {e}"

def ocr_image_auto(file_obj) -> str:
    """
    Try Tesseract OCR; if output is low quality, fallback to Gemini Vision OCR.
    Heuristic: if tesseract output length < 10 or contains almost no alpha-numeric, fallback.
    """
    t_out = ocr_image_tesseract(file_obj)
    if not t_out or t_out.startswith("⚠️") or len(t_out) < 20:
        # attempt fallback
        data, filename = safe_read_file_obj(file_obj)
        if data:
            g_out = gemini_vision_ocr_fallback(data)
            # if Gemini returns something reasonable, use it
            if g_out and not g_out.startswith("⚠️") and len(g_out.strip()) > len(t_out.strip()):
                return g_out.strip()
            # otherwise return tesseract result (even if small)
        return t_out
    # otherwise return Tesseract result
    return t_out

def highlight_similarities(text1, text2):
    matcher = difflib.SequenceMatcher(None, text1.split(), text2.split())
    output = []
    t2_words = text2.split()
    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == "equal":
            output.append(f"<mark style='background-color:yellow'>{' '.join(t2_words[b0:b1])}</mark>")
        else:
            output.append(' '.join(t2_words[b0:b1]))
    return " ".join(output)




# -----------------------
# Resume / LinkedIn / Job-Fit
# -----------------------
def resume_builder(text):
    prompt = f"Create a professional resume based on this information:\n{text}"
    return gemini_text_response_sync(prompt)

def linkedin_builder(text):
    prompt = f"Generate an optimized LinkedIn profile summary for this person:\n{text}"
    return gemini_text_response_sync(prompt)

# -----------------------
# PDF Q&A manager (TF-IDF optional)
# -----------------------
class PDFQAManager:
    def __init__(self):
        self.chunks = []
        self.sources = []
        self.vectorizer = None
        self.tfidf = None
        self.faiss_index = None

    def add_pdf(self, file_obj, filename="uploaded.pdf", chunk_size_words=400, overlap=50):
        text = extract_text_from_pdf(file_obj)
        if text.startswith("⚠️"):
            return text
        words = text.split()
        i = 0
        n = len(words)
        count = 0
        while i < n:
            chunk = " ".join(words[i:i+chunk_size_words])
            self.chunks.append(chunk)
            self.sources.append(f"{filename} :: words {i}-{min(i+chunk_size_words, n)}")
            count += 1
            i += chunk_size_words - overlap
        self.vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)
        self.tfidf = self.vectorizer.fit_transform(self.chunks)
        if FAISS_AVAILABLE:
            try:
                import numpy as np
                mat = self.tfidf.toarray().astype('float32')
                dim = mat.shape[1]
                self.faiss_index = faiss.IndexFlatL2(dim)
                self.faiss_index.add(mat)
            except Exception:
                self.faiss_index = None
        return f"Indexed {count} chunks from {filename}."

    def query(self, question, top_k=3):
        if self.tfidf is None:
            return []
        q_vec = self.vectorizer.transform([question])
        cos_sim = linear_kernel(q_vec, self.tfidf).flatten()
        top_idx = cos_sim.argsort()[::-1][:top_k]
        return [(float(cos_sim[i]), self.chunks[i], self.sources[i]) for i in top_idx]

pdfqa_manager = PDFQAManager()

# -----------------------
# Timeout runner (10s)
# -----------------------
class TimeoutResult:
    def __init__(self):
        self.value = None
        self.error = None
        self.timed_out = False

def run_with_timeout(fn, args=(), kwargs=None, timeout=10):
    kwargs = kwargs or {}
    res = TimeoutResult()
    def target():
        try:
            res.value = fn(*args, **kwargs)
        except Exception as e:
            res.error = str(e)
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        res.timed_out = True
    return res

def pdf_qa_answer(question: str, use_gemini_api_key: str = ""):
    if not question or not question.strip():
        return "Please type a question."
    if pdfqa_manager.tfidf is None:
        return "No documents indexed. Upload PDFs first."

    def do_answer():
        hits = pdfqa_manager.query(question, top_k=4)
        if not hits:
            return "No relevant content found in indexed PDFs."
        context = "\n\n---\n\n".join([h[1] for h in hits])
        if use_gemini_api_key and use_gemini_api_key.strip():
            try:
                genai.configure(api_key=use_gemini_api_key)
                model = genai.GenerativeModel("gemini-2.5-flash")
                prompt = (f"You are a helpful assistant. Use the context below (from PDFs) to answer the question concisely.\n\n"
                          f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:\n")
                resp = model.generate_content(prompt)
                text = resp.text or ""
                evidence = "\n\n---\n\n" + "\n\n".join([f"Source: {h[2]} (score {h[0]:.3f})\n\n{h[1][:600]}..." for h in hits])
                return text + evidence
            except Exception:
                pass
        snippets = []
        for score, snippet, src in hits:
            snippets.append(f"**Source:** {src}  \n**Score:** {score:.3f}  \n> {snippet[:800]}\n")
        return "### Extractive Answer (fast)\n\n" + "\n\n".join(snippets)

    result = run_with_timeout(do_answer, timeout=10)
    if result.timed_out:
        return "⏰ Timeout: answer exceeded 10 seconds. Try a shorter question or fewer documents."
    if result.error:
        return f"⚠️ Error: {result.error}"
    return result.value

# -----------------------
# Interview Simulator
# -----------------------
def advanced_interview_simulator(topic: str, difficulty: str = "medium"):
    if not topic or not topic.strip():
        return "Please provide a topic."
    prompt = (f"Generate 5 interview questions on the topic: {topic}\n"
              f"Difficulty: {difficulty}\n\nFor each question include:\n"
              "- The question\n- A concise ideal answer\n- Two tips for improvement\n- A skill rating 0-10\nReturn as a numbered list.")
    return gemini_text_response_sync(prompt)

# -----------------------
# Gamification & chat history
# -----------------------
xp_points = 0
def update_xp():
    global xp_points
    xp_points += random.randint(5, 15)
    badges = ["🌱 Beginner", "🚀 Skilled", "🌟 Expert", "🏆 Legend"]
    level = badges[min(len(badges)-1, xp_points // 40)]
    return f"🎮 XP: {xp_points} | Level: {level}"

chat_history = []

# -----------------------
# Data Summarization functions
# -----------------------
def summarize_pdf(file_obj, mode="short"):
    text = extract_text_from_pdf(file_obj)
    if text.startswith("⚠️"):
        return text
    return summarize_text_sync(text, mode=mode)

def summarize_text_input(text, length="short", style="abstractive"):
    if not text or not text.strip():
        return "Please provide text to summarize."
    if style == "extractive":
        try:
            sents = [s.strip() for s in text.split('.') if s.strip()]
            if not sents:
                return "No sentences to summarize."
            vec = TfidfVectorizer(stop_words='english')
            mat = vec.fit_transform(sents)
            scores = mat.sum(axis=1).A1
            top_n = min(len(sents), 5 if length=="short" else 8 if length=="medium" else 12)
            idxs = scores.argsort()[::-1][:top_n]
            bullets = ["- " + sents[i].strip() + "." for i in idxs]
            return "\n".join(bullets)
        except Exception as e:
            return f"⚠️ Extractive summarization failed: {e}"
    else:
        return summarize_text_sync(text, mode=length)

def summarize_dataframe(file_obj):
    try:
        if hasattr(file_obj, "name") and file_obj.name.lower().endswith(".csv"):
            df = pd.read_csv(file_obj.name)
        else:
            try:
                df = pd.read_csv(file_obj.file)
            except Exception:
                df = pd.read_excel(file_obj.file)
    except Exception as e:
        return f"⚠️ Failed to read table: {e}"
    cols = df.columns.tolist()
    missing = df.isnull().sum().to_dict()
    sample = df.head(3).to_dict(orient="records")
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    numeric_summary = []
    for c in num_cols:
        numeric_summary.append(f"{c}: mean={df[c].mean():.3f}, median={df[c].median():.3f}, std={df[c].std():.3f}")
    base_text = f"Columns ({len(cols)}): {cols}\nMissing: {missing}\nSample rows: {sample}\nNumeric summary:\n" + "\n".join(numeric_summary)
    try:
        prompt = f"Provide 3 short insights and 3 suggested actions for this dataset summary:\n\n{base_text}"
        return gemini_text_response_sync(prompt)
    except Exception:
        return base_text

def summarize_conversation(mode="short"):
    if not chat_history:
        return "No conversation history yet."
    convo_text = "\n".join([f"User: {h.get('q','')} Assistant: {h.get('a','')}" for h in chat_history[-30:]])
    return summarize_text_sync(convo_text, mode=mode)

# -----------------------
# NEW: question paper extractor that supports JPG/PNG/PDF/TXT (uses OCR auto)
# -----------------------
def extract_question_text(file_obj):
    if file_obj is None:
        return ""
    data, filename = safe_read_file_obj(file_obj)
    name = (filename or "").lower()
    # image heuristics (extension or JPEG magic bytes)
    if name.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")) or (isinstance(data, (bytes, bytearray)) and data[:4].startswith(b'\xff\xd8')):
        return ocr_image_auto(file_obj)
    if name.endswith(".pdf"):
        return extract_text_from_pdf(file_obj)
    # try text
    try:
        return extract_text_from_txt(file_obj)
    except Exception:
        return "⚠️ Unsupported question paper format."

# -----------------------
# NEW: answer_from_pdf (full feature)
# -----------------------
def answer_from_pdf(question_paper_file, study_pdf_file, marks):
    qp_text = extract_question_text(question_paper_file)
    if qp_text.startswith("⚠️"):
        return qp_text
    study_text = extract_text_from_pdf(study_pdf_file)
    if study_text.startswith("⚠️"):
        return study_text

    mark_rules = {
        "2": "Answer in 2–3 lines only.",
        "7": "Answer in one detailed paragraph suitable for 7 marks.",
        "14": "Answer in multiple long paragraphs suitable for 14 marks with proper explanation."
    }

    prompt = f"""
You MUST use ONLY the content from the Study Material PDF below.
Do NOT introduce facts not present in the Study Material.
Do NOT change the meaning of the original text.

Study Material PDF:
{study_text}

Question Paper:
{qp_text}

Required Answer Format: {mark_rules.get(marks, 'Answer briefly.')}

Please generate clear answers to the questions in the Question Paper using only the Study Material. If an answer cannot be found in the Study Material, explicitly say: "Answer not available in the Study Material."
"""

    try:
        model = genai.GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(prompt)
        result = response.text or "⚠️ No response from Gemini."
    except Exception as e:
        result = f"⚠️ Gemini error: {e}"
    return result

# -----------------------
# UI: Build Gradio App (with effective background & animations)
# -----------------------
css = """
/* Background gradient + subtle animated shapes */
body{
  background: radial-gradient(circle at 10% 20%, rgba(255,255,255,0.03), transparent 10%),
              radial-gradient(circle at 90% 80%, rgba(255,255,255,0.02), transparent 10%),
              linear-gradient(120deg,#06133a 0%, #1a2b6f 50%, #5b2ff7 100%);
  color: #e6eef8;
  font-family: 'Helvetica Neue', Arial, sans-serif;
}

/* floating blob animation */
.blob {
  position:absolute;
  width:420px;
  height:420px;
  filter: blur(80px);
  opacity:0.08;
  border-radius:50%;
  animation: floaty 12s ease-in-out infinite;
}
.blob.blob1 { background: linear-gradient(90deg,#00b4d8,#90f7ec); left:-120px; top:-80px; }
.blob.blob2 { background: linear-gradient(90deg,#ff0099,#ffb347); right:-140px; bottom:-100px; animation-duration:16s; opacity:0.06; }

@keyframes floaty {
  0% { transform: translateY(0) translateX(0) scale(1); }
  50% { transform: translateY(-30px) translateX(20px) scale(1.05); }
  100% { transform: translateY(0) translateX(0) scale(1); }
}

/* glass cards */
.card {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.04);
  backdrop-filter: blur(8px);
  border-radius: 12px;
  padding: 12px;
}

/* headers */
h1, h2, h3 { color: #f0f9ff; }

/* buttons */
button.gradio-button {
  background: linear-gradient(90deg,#00b4d8,#ff0099);
  border: none;
  color: white;
  box-shadow: 0 6px 18px rgba(0,0,0,0.25);
}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft(), title="Career Suite") as app:
    # animated blobs (decorative)
    app.launch_kwargs = {"show_error": True}
    gr.HTML("<div class='blob blob1'></div><div class='blob blob2'></div>")
    gr.Markdown("<h1 style='text-align:center;margin-top:8px'>Career Suite-RGMCET</h1>")

    with gr.Tabs():
        # Chat & Summarizer
        with gr.TabItem("💬 Chat & Summarizer"):
            with gr.Row():
                with gr.Column(scale=8):
                    chat_in = gr.Textbox(label="Ask anything:", placeholder="e.g., Explain RAG in 2 sentences")
                    chat_out = gr.Markdown()
                    chat_btn = gr.Button("Generate")
                with gr.Column(scale=2):
                    xp_md = gr.Markdown()
            chat_btn.click(lambda q: (gemini_text_response_sync(q), update_xp()), inputs=chat_in, outputs=[chat_out, xp_md])

        # Resume Builder
        with gr.TabItem("🧠 Resume Builder"):
            resume_in = gr.Textbox(label="Enter your details:", lines=8)
            resume_out = gr.Markdown()
            btn_r = gr.Button("📄 Build Resume")
            btn_r.click(lambda t: (resume_builder(t), update_xp()), inputs=resume_in, outputs=resume_out)

        # LinkedIn Builder
        with gr.TabItem("🔗 LinkedIn Builder"):
            ln_in = gr.Textbox(label="Professional details:", lines=8)
            ln_out = gr.Markdown()
            btn_ln = gr.Button("💼 Generate LinkedIn Summary")
            btn_ln.click(lambda t: (linkedin_builder(t), update_xp()), inputs=ln_in, outputs=ln_out)

        # Data Summarization
        with gr.TabItem("📊 Data Summarization"):
            gr.Markdown("### PDF Summarization")
            pdf_file = gr.File(label="Upload PDF for summarization")
            pdf_mode = gr.Radio(["short","medium","detailed"], value="short", label="Length")
            pdf_sum_out = gr.Markdown()
            pdf_sum_btn = gr.Button("Summarize PDF")
            pdf_sum_btn.click(lambda f,m: summarize_pdf(f,m), inputs=[pdf_file, pdf_mode], outputs=pdf_sum_out)

            gr.Markdown("### Text Summarization")
            text_in = gr.Textbox(lines=6, label="Paste text to summarize")
            text_len = gr.Radio(["short","medium","detailed"], value="short", label="Length")
            text_style = gr.Radio(["abstractive","extractive"], value="abstractive", label="Style")
            text_sum_out = gr.Markdown()
            text_sum_btn = gr.Button("Summarize Text")
            text_sum_btn.click(lambda t,l,s: summarize_text_input(t, l, s), inputs=[text_in, text_len, text_style], outputs=text_sum_out)

            gr.Markdown("### Table / CSV / Excel Summarization")
            table_file = gr.File(label="Upload CSV or Excel file")
            table_out = gr.Markdown()
            table_btn = gr.Button("Summarize Table")
            table_btn.click(lambda f: summarize_dataframe(f), inputs=table_file, outputs=table_out)

            gr.Markdown("### Conversation Summarization")
            convo_mode = gr.Radio(["short","medium","detailed"], value="short", label="Length")
            convo_out = gr.Markdown()
            convo_btn = gr.Button("Summarize Conversation")
            convo_btn.click(lambda m: summarize_conversation(m), inputs=convo_mode, outputs=convo_out)


        # PDF Q&A
        with gr.TabItem("📘 PDF Q&A"):
            pdf_upload_files = gr.File(file_count="multiple", label="Upload PDFs")
            pdf_upload_status = gr.Markdown()
            pdf_upload_btn = gr.Button("Upload & Index PDFs")
            pdf_key_for_ans = gr.Textbox(label="(Optional) Gemini API Key for concise generative answer", type="password")
            ask_box = gr.Textbox(label="Ask a question about uploaded PDFs")
            answer_out = gr.Markdown()
            ask_btn = gr.Button("❓ Get Answer (≤10s)")

            def handle_upload(files):
                if not files:
                    return "No files uploaded."
                msgs = []
                for f in files:
                    fname = getattr(f, "name", getattr(f, "filename", "uploaded.pdf"))
                    try:
                        msgs.append(pdfqa_manager.add_pdf(f, filename=os.path.basename(fname)))
                    except Exception as e:
                        msgs.append(f"Failed to index {fname}: {e}")
                return "\n\n".join(msgs)

            pdf_upload_btn.click(handle_upload, inputs=[pdf_upload_files], outputs=[pdf_upload_status])
            ask_btn.click(pdf_qa_answer, inputs=[ask_box, pdf_key_for_ans], outputs=[answer_out])

        # Question Paper → PDF Answer Generator (supports JPG/PNG/PDF/TXT)
        with gr.TabItem("📘 Question Paper → PDF Answer Generator"):
            gr.Markdown("### Upload Question Paper (JPG/PNG/PDF/TXT) + Study PDF → Get Answers (2 / 7 / 14 Marks)")
            qp_input = gr.File(label="Upload Question Paper (JPG/PNG/PDF/TXT)")
            study_pdf_input = gr.File(label="Upload Study Material PDF")
            marks_radio = gr.Radio(["2", "7", "14"], label="Select Answer Format", value="2")
            qp_answer_output = gr.Markdown()
            qp_btn = gr.Button("📝 Generate Answers")
            qp_btn.click(answer_from_pdf, inputs=[qp_input, study_pdf_input, marks_radio], outputs=qp_answer_output)

        # Interview Simulator
        with gr.TabItem("🗣️ Interview Simulator"):
            int_topic = gr.Textbox(label="Topic", placeholder="e.g., System Design")
            int_diff = gr.Dropdown(["easy","medium","hard"], value="medium", label="Difficulty")
            int_out = gr.Markdown()
            int_btn = gr.Button("Start Simulation")
            int_btn.click(advanced_interview_simulator, inputs=[int_topic, int_diff], outputs=int_out)


    app.launch(share=True, debug=True)


