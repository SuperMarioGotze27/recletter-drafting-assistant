import os
import re
import time
from pathlib import Path
from io import BytesIO

import pandas as pd
import streamlit as st
from transformers import pipeline
from openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from docx import Document


# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="AI Recommendation Letter Assistant",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CSS styling
# =============================================================================
st.markdown(
    """
    <style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stMetric { background: #f8f9fa; border-radius: 8px; padding: 10px; }
    .letter-box {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 24px;
        font-family: Georgia, serif;
        line-height: 1.7;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Helpers
# =============================================================================
def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def truncate(text: str, limit: int = 2400) -> str:
    text = clean_text(text)
    return text[:limit]


def read_uploaded_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".txt":
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(uploaded_file)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            st.warning(f"PDF text extraction was not available: {exc}")
            return ""
    return ""


def compose_profile_text(profile: dict) -> str:
    parts = [
        f"Student: {profile.get('student_name', '')}",
        f"GPA: {profile.get('gpa', '')}",
        f"Class percentile: top {profile.get('class_percentile', '')} percent",
        f"Course highlights: {profile.get('course_highlights', '')}",
        f"Relationship: {profile.get('relationship', '')}",
        f"Target: {profile.get('target_program', '')}",
        f"Faculty note: {profile.get('faculty_note', '')}",
        f"Project evidence: {profile.get('project_summary', '')}",
        f"Transcript/profile text: {truncate(profile.get('uploaded_text', ''))}",
    ]
    return clean_text(". ".join(part for part in parts if clean_text(part)))


# =============================================================================
# Model loading (cached)
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_pipelines():
    model_path = Path(__file__).resolve().parent / "recletter_strength_model"
    strength_pipe = pipeline(
        "text-classification",
        model=str(model_path),
        tokenizer=str(model_path),
        top_k=3,
        truncation=True,
    )
    tone_pipe = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
    )
    return strength_pipe, tone_pipe


def analyze_profile(profile: dict) -> dict:
    strength_pipe, tone_pipe = load_pipelines()
    profile_text = compose_profile_text(profile)
    strength_results = strength_pipe(profile_text)[0]
    strength_results = sorted(strength_results, key=lambda item: item["score"], reverse=True)
    best_strength = max(strength_results, key=lambda item: item["score"])
    tone_input = clean_text(profile.get("faculty_note") or profile_text)
    tone = tone_pipe(tone_input[:512])[0]
    return {
        "profile_text": profile_text,
        "strength": best_strength["label"],
        "strength_confidence": float(best_strength["score"]),
        "tone": tone["label"],
        "tone_confidence": float(tone["score"]),
        "top_strengths": "; ".join(
            f"{item['label']} ({item['score']:.1%})" for item in strength_results
        ),
    }


# =============================================================================
# LLM-powered letter generation
# =============================================================================
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        api_key = st.session_state.get("openai_api_key", "")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def build_llm_prompt(profile: dict, analysis: dict) -> str:
    name = profile.get("student_name", "the student")
    recommender = profile.get("recommender", "Professor")
    institution = profile.get("institution", "the university")
    target = profile.get("target_program", "your program")
    relationship = profile.get("relationship", "course instructor")
    courses = profile.get("course_highlights", "my course")
    gpa = profile.get("gpa", "")
    percentile = profile.get("class_percentile", "")
    note = profile.get("faculty_note", "")
    project = profile.get("project_summary", "")
    strength = analysis["strength"]
    tone = analysis["tone"]

    prompt = f"""You are an experienced university professor writing a recommendation letter for a student.

STUDENT PROFILE:
- Name: {name}
- GPA: {gpa}
- Class standing: top {percentile} percent
- Courses taken: {courses}
- Your relationship to student: {relationship}
- Target program: {target}
- Your institution: {institution}
- Faculty observations: {note}
- Project/work evidence: {project}

ML MODEL ANALYSIS:
- Predicted recommendation strength: {strength}
- Faculty note sentiment: {tone}

INSTRUCTIONS:
Write a professional, warm, and specific recommendation letter of approximately 350-450 words.
The letter should:
1. Open with a clear statement of recommendation strength
2. Include specific evidence from the profile (GPA, courses, projects, observations)
3. Connect the student's abilities to the target program
4. Close with a strong endorsement
5. Use natural, human-like language — avoid generic phrases like "hard-working" without evidence
6. Be formatted as a formal letter with Dear Selection Committee and Sincerely signatures

Write the complete letter now:"""
    return prompt


def generate_letter_with_llm(profile: dict, analysis: dict) -> str:
    client = get_openai_client()
    if client is None:
        return ""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert academic recommendation letter writer."},
                {"role": "user", "content": build_llm_prompt(profile, analysis)},
            ],
            temperature=0.7,
            max_tokens=1200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"LLM generation failed: {e}")
        return ""


# =============================================================================
# Post-hoc verification: check if LLM output matches predicted strength
# =============================================================================
def verify_letter_strength(letter_text: str, expected_strength: str) -> dict:
    """Run the fine-tuned model on the generated letter to verify consistency."""
    strength_pipe, _ = load_pipelines()
    results = strength_pipe(letter_text[:512])[0]
    results = sorted(results, key=lambda item: item["score"], reverse=True)
    best = max(results, key=lambda item: item["score"])
    return {
        "predicted_on_letter": best["label"],
        "confidence": float(best["score"]),
        "matches_expected": best["label"] == expected_strength,
        "all_scores": "; ".join(f"{r['label']} ({r['score']:.1%})" for r in results),
    }


# =============================================================================
# Legacy template-based draft (fallback)
# =============================================================================
def strength_phrase(label: str) -> str:
    return {
        "Moderate Recommendation": "recommend",
        "Strong Recommendation": "strongly recommend",
        "Exceptional Recommendation": "recommend with exceptional enthusiasm",
    }.get(label, "recommend")


def draft_letter_template(profile: dict, analysis: dict) -> str:
    name = clean_text(profile.get("student_name")) or "the student"
    recommender = clean_text(profile.get("recommender")) or "Professor"
    institution = clean_text(profile.get("institution")) or "the university"
    target = clean_text(profile.get("target_program")) or "your program"
    relationship = clean_text(profile.get("relationship")) or "course instructor"
    courses = clean_text(profile.get("course_highlights")) or "my course"
    gpa = clean_text(profile.get("gpa")) or "the reported GPA"
    percentile = clean_text(profile.get("class_percentile")) or "the reported class percentile"
    note = clean_text(profile.get("faculty_note")) or "showed consistent academic engagement"
    project = clean_text(profile.get("project_summary")) or "completed an applied analytics project"
    phrase = strength_phrase(analysis["strength"])

    paragraphs = [
        (
            f"Dear Selection Committee,\n\n"
            f"I am pleased to {phrase} {name} for {target}. I know {name} as their "
            f"{relationship} at {institution}, where the student completed {courses}."
        ),
        (
            f"The transcript-like profile indicates a GPA of {gpa} and class standing around "
            f"the top {percentile} percent. In class, {note}. This evidence suggests a student "
            f"who can connect technical work with disciplined academic judgment."
        ),
        (
            f"{project} This project evidence is important because it shows how {name} approaches "
            f"open-ended work: defining a problem, using data carefully, and communicating results "
            f"in a way that a business or academic audience can act on."
        ),
        (
            f"The model classifies this profile as {analysis['strength']} "
            f"with {analysis['strength_confidence']:.1%} confidence. Please treat this as an editable faculty draft. "
            f"After my review, I believe {name} would bring maturity, curiosity, and a constructive "
            f"working style to {target}.\n\nSincerely,\n{recommender}"
        ),
    ]
    return "\n\n".join(paragraphs)


# =============================================================================
# Export helpers
# =============================================================================
def generate_pdf(letter_text: str, student_name: str) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Justify", alignment=TA_JUSTIFY, fontSize=11, leading=16, spaceAfter=12))
    story = []
    for para in letter_text.split("\n\n"):
        if para.strip():
            story.append(Paragraph(para.replace("\n", "<br/>"), styles["Justify"]))
            story.append(Spacer(1, 12))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def generate_docx(letter_text: str) -> bytes:
    doc = Document()
    for para in letter_text.split("\n\n"):
        if para.strip():
            doc.add_paragraph(para)
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


# =============================================================================
# Batch processing
# =============================================================================
def profile_from_row(row: pd.Series) -> dict:
    defaults = {
        "student_name": "Student",
        "gpa": "",
        "class_percentile": "",
        "course_highlights": "",
        "relationship": "course instructor",
        "target_program": "graduate program",
        "faculty_note": "",
        "project_summary": "",
        "institution": "the university",
        "recommender": "Professor",
        "uploaded_text": "",
    }
    profile = defaults | {key: row.get(key, defaults.get(key, "")) for key in defaults}
    return {key: clean_text(value) for key, value in profile.items()}


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.header("⚙️ Settings")
    use_llm = st.toggle("Use LLM (OpenAI) for letter generation", value=True, help="Toggle between AI-generated letters and template-based drafts")
    st.caption("LLM mode produces more natural, personalized letters. Template mode is faster and requires no API key.")
    
    openai_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password", help="Required for LLM mode. Get one at platform.openai.com")
    if openai_key:
        st.session_state["openai_api_key"] = openai_key
    
    st.divider()
    st.markdown("**About this project**")
    st.markdown("""
    - 🧠 Fine-tuned BERT-tiny for recommendation strength classification
    - 😊 DistilBERT sentiment analysis for faculty note tone check
    - 🤖 Optional OpenAI GPT-4o-mini for intelligent letter generation
    - ✅ Post-hoc ML verification ensures generated letter matches predicted strength
    """)
    st.markdown("[GitHub Repo](https://github.com/SuperMarioGotze27/recletter-drafting-assistant)")


# =============================================================================
# Main UI
# =============================================================================
st.title("📝 AI-Assisted Recommendation Letter Drafting System")
st.caption("For university faculty — generate, verify, and refine recommendation letter drafts with ML + LLM assistance")

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.subheader("👤 Student Profile")
    student_name = st.text_input("Student name", "Mina Chen")
    target_program = st.text_input("Application target", "MSc in Business Analytics")
    course_highlights = st.text_input(
        "Course highlights",
        "Machine Learning for Business; Deep Learning Applications; Python Programming",
    )
    c1, c2 = st.columns(2)
    with c1:
        gpa = st.text_input("GPA", "3.82")
    with c2:
        class_percentile = st.text_input("Top percentile", "8")
    relationship = st.text_input("Faculty relationship", "project supervisor in an analytics course")
    project_summary = st.text_area(
        "Project evidence",
        "Mina built an end-to-end NLP application using Python and transformer pipelines, then explained model tradeoffs clearly in the final report.",
        height=110,
    )
    faculty_note = st.text_area(
        "Faculty note",
        "She asked original questions, supported teammates during implementation, and consistently connected technical choices to business impact.",
        height=110,
    )

with right:
    st.subheader("⚙️ Draft Settings")
    recommender = st.text_input("Recommender", "Professor")
    institution = st.text_input("Institution", "HKUST Business School")
    uploaded = st.file_uploader("Transcript/profile text", type=["txt", "pdf"])
    uploaded_text = read_uploaded_text(uploaded)
    
    st.divider()
    st.subheader("📂 Batch Processing")
    batch_file = st.file_uploader("Batch CSV", type=["csv"], key="batch")
    if batch_file is not None:
        st.caption("Batch columns: student_name, gpa, class_percentile, course_highlights, relationship, target_program, faculty_note, project_summary, institution, recommender")


profile = {
    "student_name": student_name,
    "gpa": gpa,
    "class_percentile": class_percentile,
    "course_highlights": course_highlights,
    "relationship": relationship,
    "target_program": target_program,
    "faculty_note": faculty_note,
    "project_summary": project_summary,
    "institution": institution,
    "recommender": recommender,
    "uploaded_text": uploaded_text,
}

# =============================================================================
# Single profile generation
# =============================================================================
analyze_clicked = st.button("✨ Generate Draft", type="primary", use_container_width=True)

if analyze_clicked:
    with st.spinner("Analyzing student profile..."):
        start = time.time()
        analysis = analyze_profile(profile)
        
        # Generate letter
        if use_llm and get_openai_client():
            with st.spinner("Generating letter with LLM..."):
                letter = generate_letter_with_llm(profile, analysis)
            if letter:
                with st.spinner("Verifying letter consistency with ML model..."):
                    verification = verify_letter_strength(letter, analysis["strength"])
            else:
                verification = None
                st.warning("LLM generation failed. Falling back to template mode.")
                letter = draft_letter_template(profile, analysis)
        else:
            letter = draft_letter_template(profile, analysis)
            verification = None
        
        elapsed = time.time() - start

    # Metrics
    st.divider()
    st.subheader("📊 Analysis Results")
    cols = st.columns(4)
    cols[0].metric("Strength", analysis["strength"])
    cols[1].metric("Confidence", f"{analysis['strength_confidence']:.1%}")
    cols[2].metric("Note Tone", analysis["tone"])
    cols[3].metric("Tone Confidence", f"{analysis['tone_confidence']:.1%}")
    
    if verification:
        vcols = st.columns(3)
        vcols[0].metric("Letter Verification", "✅ Match" if verification["matches_expected"] else "⚠️ Mismatch")
        vcols[1].metric("Verified Strength", verification["predicted_on_letter"])
        vcols[2].metric("Runtime", f"{elapsed:.1f}s")
        if not verification["matches_expected"]:
            st.warning(f"The generated letter was classified as **{verification['predicted_on_letter']}** by the ML model, but the student profile suggests **{analysis['strength']}**. Consider reviewing the draft.")
    else:
        st.metric("Runtime", f"{elapsed:.1f}s")
    
    # Detailed probabilities
    with st.expander("View model probabilities"):
        st.write(f"**Top strength probabilities:** {analysis['top_strengths']}")
        st.write(f"**Profile text length:** {len(analysis['profile_text'])} characters")
        if verification:
            st.write(f"**Letter strength scores:** {verification['all_scores']}")

    # Letter display
    st.divider()
    st.subheader("📄 Recommendation Letter Draft")
    st.markdown(f'<div class="letter-box">{letter.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
    
    # Editable text area (hidden behind expander for fine-tuning)
    with st.expander("✏️ Edit letter text directly"):
        edited_letter = st.text_area("Editable draft", letter, height=420)
        if edited_letter != letter:
            letter = edited_letter
            st.success("Letter updated. Download buttons below will use the edited version.")

    # Export buttons
    st.divider()
    st.subheader("💾 Export")
    ec1, ec2, ec3 = st.columns(3)
    safe_name = student_name.replace(" ", "_")
    with ec1:
        st.download_button(
            "📄 Download as TXT",
            data=letter.encode("utf-8"),
            file_name=f"{safe_name}_recommendation_draft.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with ec2:
        st.download_button(
            "📑 Download as PDF",
            data=generate_pdf(letter, student_name),
            file_name=f"{safe_name}_recommendation_draft.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with ec3:
        st.download_button(
            "📝 Download as DOCX",
            data=generate_docx(letter),
            file_name=f"{safe_name}_recommendation_draft.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )


# =============================================================================
# Batch processing
# =============================================================================
if batch_file is not None:
    st.divider()
    st.subheader("📂 Batch Draft Results")
    df = pd.read_csv(batch_file)
    rows = []
    progress = st.progress(0, text="Processing batch...")
    with st.spinner("Generating batch drafts..."):
        for i, (_, row) in enumerate(df.iterrows()):
            batch_profile = profile_from_row(row)
            analysis = analyze_profile(batch_profile)
            if use_llm and get_openai_client():
                letter = generate_letter_with_llm(batch_profile, analysis) or draft_letter_template(batch_profile, analysis)
            else:
                letter = draft_letter_template(batch_profile, analysis)
            rows.append(
                {
                    "student_name": batch_profile["student_name"],
                    "predicted_strength": analysis["strength"],
                    "strength_confidence": analysis["strength_confidence"],
                    "note_tone": analysis["tone"],
                    "tone_confidence": analysis["tone_confidence"],
                    "draft_letter": letter,
                }
            )
            progress.progress((i + 1) / len(df), text=f"Processed {i+1}/{len(df)}...")
    progress.empty()
    
    result_df = pd.DataFrame(rows)
    st.dataframe(result_df.drop(columns=["draft_letter"]), hide_index=True, use_container_width=True)
    st.download_button(
        "Download batch drafts (CSV)",
        data=result_df.to_csv(index=False).encode("utf-8"),
        file_name="recommendation_letter_batch_drafts.csv",
        mime="text/csv",
        use_container_width=True,
    )
