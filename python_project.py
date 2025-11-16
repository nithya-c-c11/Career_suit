# =====================================================
# 🌟 Gemini 2.5 Pro All-in-One AI Career Suite (Extended + PDF + Summarization)
# =====================================================

# 📦 Install dependencies
!pip install -q google-generativeai gradio PyPDF2 Pillow

# =====================================================
# 📚 Imports
# =====================================================
import os, time, random, wave, difflib
from io import BytesIO
from PIL import Image
import gradio as gr
from PyPDF2 import PdfReader
import google.generativeai as genai

# =====================================================
# 🔑 API Setup
# =====================================================
API_KEY = "AIzaSyB5roew7_J1jrcDra4Td8l-M9LBFduXyeE"
genai.configure(api_key=API_KEY)

# =====================================================
# 🔧 Helper Functions
# =====================================================
def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)

def extract_text_from_pdf(file_obj):
    try:
        reader = PdfReader(file_obj)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip()
    except Exception as e:
        return f"⚠️ Failed to read PDF: {e}"

def highlight_similarities(text1, text2):
    matcher = difflib.SequenceMatcher(None, text1.split(), text2.split())
    output = []
    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == "equal":
            output.append(f"<mark style='background-color:yellow'>{' '.join(text2.split()[b0:b1])}</mark>")
        else:
            output.append(' '.join(text2.split()[b0:b1]))
    return " ".join(output)

# =====================================================
# 🧠 Gemini Core Wrappers
# =====================================================
def gemini_text_response(prompt):
    try:
        model = genai.GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(prompt)
        answer = response.text or "⚠️ No response received."
        summary = summarize_text(answer)
        return f"🪄 **Answer:**\n{answer}\n\n🧭 **Summary:**\n{summary}"
    except Exception as e:
        return f"⚠️ Gemini failed: {e}"

def summarize_text(text: str):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(f"Summarize this text briefly:\n{text}")
        return response.text
    except Exception as e:
        return f"⚠️ Summary failed: {e}"

# =====================================================
# 🖼️ Image Generation (with explicit path)
# =====================================================
def gemini_image(prompt):
    try:
        # Define explicit image path (works for Colab and local)
        base_path = os.path.abspath("generated_images")
        os.makedirs(base_path, exist_ok=True)

        model = genai.GenerativeModel("gemini-2.5-flash-image")
        resp = model.generate_content(prompt)

        for part in resp.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                image = Image.open(BytesIO(part.inline_data.data))
                filename = f"image_{int(time.time())}.jpg"
                save_path = os.path.join(base_path, filename)
                image.save(save_path, format="JPEG")

                return save_path  # Return full file path
        return "⚠️ No image data returned by Gemini."
    except Exception as e:
        return f"⚠️ Image generation failed: {e}"

# =====================================================
# 📄 Resume, Job & Analysis Tools
# =====================================================
def resume_builder(text):
    prompt = f"Create a professional resume based on this information:\n{text}"
    return gemini_text_response(prompt)

def linkedin_builder(text):
    prompt = f"Generate an optimized LinkedIn profile summary for this person:\n{text}"
    return gemini_text_response(prompt)

def job_fit_analyzer(job_desc, resume_text):
    prompt = f"""Compare this job description and resume.
    Provide a fit score (0–100) and improvement suggestions.

    Job Description:
    {job_desc}

    Resume:
    {resume_text}
    """
    return gemini_text_response(prompt)

# =====================================================
# 🧾 Plagiarism, ATS & PDF Tools
# =====================================================
def plagiarism_checker(text1, text2):
    ratio = difflib.SequenceMatcher(None, text1, text2).ratio()
    return f"🧾 Similarity Score: {ratio*100:.2f}%\n{'⚠️ Possible plagiarism detected!' if ratio>0.8 else '✅ Texts appear unique.'}"

def ats_checker(resume_text, job_desc):
    resume_words = set(resume_text.lower().split())
    job_words = set(job_desc.lower().split())
    match = len(resume_words & job_words)
    score = match / max(1, len(job_words)) * 100
    return f"📊 ATS Match Score: {score:.2f}%\n{'👍 Good match!' if score>50 else '⚠️ Add more job-related keywords.'}"

def pdf_plagiarism_checker(pdf1, pdf2):
    text1 = extract_text_from_pdf(pdf1)
    text2 = extract_text_from_pdf(pdf2)
    ratio = difflib.SequenceMatcher(None, text1, text2).ratio()
    highlighted = highlight_similarities(text1, text2)
    score = f"🧾 Similarity Score: {ratio*100:.2f}%"
    msg = "⚠️ Possible plagiarism detected!" if ratio > 0.8 else "✅ Texts appear unique."
    return f"{score}\n{msg}", highlighted

def question_answer_generator(topic):
    prompt = f"Generate 5 exam-style questions and answers about: {topic}"
    return gemini_text_response(prompt)

def data_summarization(text):
    prompt = f"Summarize and extract key data insights from the following text:\n{text}"
    return gemini_text_response(prompt)

# =====================================================
# 🎮 Gamification
# =====================================================
xp_points = 0
def update_xp():
    global xp_points
    xp_points += random.randint(5, 15)
    badges = ["🌱 Beginner", "🚀 Skilled", "🌟 Expert", "🏆 Legend"]
    level = badges[min(len(badges)-1, xp_points // 40)]
    return f"🎮 XP: {xp_points} | Level: {level}"

# =====================================================
# 🧱 Build Gradio UI
# =====================================================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="pink"),
               title="🌈 Gemini 2.5 Pro Career Suite") as app:

    gr.HTML("""
    <div style='text-align:center; font-size:30px; font-weight:bold;
                background:linear-gradient(90deg,#00b4d8,#ff0099);
                -webkit-background-clip:text; color:transparent;
                animation: glow 3s ease-in-out infinite alternate;'>🌟 Gemini 2.5 Pro – Career Suite (Extended)</div>
    <style>@keyframes glow {from{filter:drop-shadow(0 0 5px #00f);} to{filter:drop-shadow(0 0 20px #ff0);}}</style>
    """)

    # 💬 Chat & Summarizer
    with gr.Tab("💬 Chat & Summarizer"):
        prompt = gr.Textbox(label="Ask anything:")
        output = gr.Markdown()
        xp = gr.Markdown()
        btn = gr.Button("✨ Generate")
        btn.click(fn=lambda p:(gemini_text_response(p), update_xp()), inputs=prompt, outputs=[output, xp])

    # 🧠 Resume Builder
    with gr.Tab("🧠 Resume Builder"):
        resume_in = gr.Textbox(label="Enter your details:", lines=8)
        resume_out = gr.Markdown()
        xp2 = gr.Markdown()
        btn_r = gr.Button("📄 Build Resume")
        btn_r.click(fn=lambda t:(resume_builder(t), update_xp()), inputs=resume_in, outputs=[resume_out, xp2])

    # 🔗 LinkedIn Builder
    with gr.Tab("🔗 LinkedIn Builder"):
        ln_in = gr.Textbox(label="Professional details:", lines=8)
        ln_out = gr.Markdown()
        xp3 = gr.Markdown()
        btn_ln = gr.Button("💼 Generate LinkedIn Summary")
        btn_ln.click(fn=lambda t:(linkedin_builder(t), update_xp()), inputs=ln_in, outputs=[ln_out, xp3])

    # 🎯 Job-Fit Analyzer
    with gr.Tab("🎯 Job-Fit Analyzer"):
        jd = gr.Textbox(label="Job Description:", lines=6)
        rs = gr.Textbox(label="Your Resume Text:", lines=6)
        fit_out = gr.Markdown()
        xp4 = gr.Markdown()
        btn_fit = gr.Button("📊 Analyze Fit")
        btn_fit.click(fn=lambda a,b:(job_fit_analyzer(a,b), update_xp()), inputs=[jd, rs], outputs=[fit_out, xp4])

    # 📊 Data Summarization
    with gr.Tab("📊 Data Summarization"):
        data_in = gr.Textbox(label="Paste your data or report text:", lines=8)
        data_out = gr.Markdown()
        btn_sum = gr.Button("🧠 Summarize Data")
        btn_sum.click(fn=data_summarization, inputs=data_in, outputs=data_out)

    # 🧾 Text Plagiarism
    with gr.Tab("🧾 Plagiarism Checker"):
        txt1 = gr.Textbox(label="Text 1:", lines=6)
        txt2 = gr.Textbox(label="Text 2:", lines=6)
        plag_out = gr.Markdown()
        btn_plag = gr.Button("🔍 Check Similarity")
        btn_plag.click(fn=plagiarism_checker, inputs=[txt1, txt2], outputs=plag_out)

    # 📂 PDF Upload Plagiarism
    with gr.Tab("📂 PDF Resume & ATS Analyzer"):
        pdf_resume = gr.File(label="Upload Resume PDF")
        pdf_job = gr.File(label="Upload Job Description PDF")
        pdf_out = gr.Markdown()
        highlighted_out = gr.HTML()
        btn_pdf = gr.Button("📄 Analyze PDF Similarity")
        btn_pdf.click(fn=pdf_plagiarism_checker, inputs=[pdf_resume, pdf_job], outputs=[pdf_out, highlighted_out])

    # 🧠 ATS Checker
    with gr.Tab("🧠 ATS Resume Checker"):
        ats_resume = gr.Textbox(label="Resume:", lines=6)
        ats_job = gr.Textbox(label="Job Description:", lines=6)
        ats_out = gr.Markdown()
        btn_ats = gr.Button("📈 Analyze ATS Fit")
        btn_ats.click(fn=ats_checker, inputs=[ats_resume, ats_job], outputs=ats_out)

    # 🖼️ Image Generator
    with gr.Tab("🖼️ Image Generator"):
        img_prompt = gr.Textbox(label="Describe image:")
        img_out = gr.Textbox(label="📁 Saved Image Path (Absolute):")
        btn_img = gr.Button("🖌️ Generate Image")
        btn_img.click(fn=gemini_image, inputs=img_prompt, outputs=img_out)

    gr.Markdown("### ✨ Powered by Gemini 2.5 Pro | Auto-saves Images in `/generated_images` ✨")

# =====================================================
# 🚀 Launch App
# =====================================================
app.launch(share=True, debug=True)
