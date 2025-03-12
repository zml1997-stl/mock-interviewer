import os
import json
import logging
import uuid
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import google.generative_ai as genai
from google.generative_ai.types import HarmCategory, HarmBlockThreshold
import docx
import PyPDF2
from pydub import AudioSegment
import whisper

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel(
    "gemini-1.5-flash",
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
)

# Temporary storage for interview data
interviews = {}

# Directory to store audio files
STATIC_DIR = "static/audio"
os.makedirs(STATIC_DIR, exist_ok=True)

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def generate_interview_questions(resume_text, job_title, interview_level, interview_type, custom_questions):
    prompt = f"""
    You are an expert interviewer. Based on the following resume, job title, interview level, and type, generate 10 relevant interview questions.
    
    Resume: {resume_text}
    Job Title: {job_title}
    Interview Level: {interview_level}
    Interview Type: {interview_type}
    Custom Questions/Topics: {custom_questions if custom_questions else 'None'}
    
    Output the questions as a JSON list.
    """
    response = model.generate_content(prompt)
    questions = json.loads(response.text)
    return questions

def text_to_speech(text, filename):
    audio = model.generate_content(f"Convert the following text to speech: {text}")
    with open(filename, "wb") as f:
        f.write(audio.audio_data)
    return filename

def transcribe_audio(audio_path, format="wav"):
    # Load and downsample the audio to 16kHz mono
    audio = AudioSegment.from_file(audio_path, format=format)
    audio = audio.set_frame_rate(16000).set_channels(1)  # Downsample for faster processing
    audio_path_wav = audio_path.rsplit(".", 1)[0] + ".wav"
    audio.export(audio_path_wav, format="wav")
    logger.info(f"Converted to {audio_path_wav}")

    try:
        # Load Whisper model (use 'tiny' for faster transcription on low-resource dynos)
        model = whisper.load_model("tiny")
        result = model.transcribe(audio_path_wav)
        text = result["text"]
        logger.info("Successfully transcribed audio with Whisper")
        return text
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return None
    finally:
        if os.path.exists(audio_path_wav):
            os.remove(audio_path_wav)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/setup", methods=["POST"])
def setup():
    resume = request.files.get("resume")
    job_title = request.form.get("job_title")
    interview_level = request.form.get("interview_level")
    interview_type = request.form.get("interview_type")
    custom_questions = request.form.get("custom_questions")

    resume_text = ""
    if resume:
        if resume.filename.endswith(".pdf"):
            resume_text = extract_text_from_pdf(resume)
        elif resume.filename.endswith((".docx", ".doc")):
            resume_text = extract_text_from_docx(resume)
        elif resume.filename.endswith(".txt"):
            resume_text = resume.read().decode("utf-8")
        else:
            return jsonify({"error": "Unsupported file format"}), 400

    questions = generate_interview_questions(resume_text, job_title, interview_level, interview_type, custom_questions)

    interview_id = str(uuid.uuid4())
    audio_urls = []

    # Pre-generate audio for all questions
    for i, question in enumerate(questions):
        audio_filename = os.path.join(STATIC_DIR, f"{interview_id}_question_{i}.mp3")
        text_to_speech(question, audio_filename)
        audio_url = url_for("static", filename=f"audio/{interview_id}_question_{i}.mp3", t=int(datetime.utcnow().timestamp()))
        audio_urls.append(audio_url)

    interviews[interview_id] = {
        "job_title": job_title,
        "questions": questions,
        "answers": [None] * len(questions),
        "current_question": 0,
        "audio_urls": audio_urls  # Store audio_urls in the interview data
    }

    return jsonify({"interview_id": interview_id, "audio_urls": audio_urls})

@app.route("/interview/<interview_id>")
def interview(interview_id):
    if interview_id not in interviews:
        return "Interview not found", 404
    interview_data = interviews[interview_id]
    return render_template(
        "interview.html",
        interview_id=interview_id,
        job_title=interview_data["job_title"],
        questions=interview_data["questions"],
        audio_urls=interview_data.get("audio_urls", [])
    )

@app.route("/ask_question", methods=["POST"])
def ask_question():
    data = request.get_json()
    interview_id = data["interview_id"]
    question_index = data["question_index"]

    if interview_id not in interviews:
        return jsonify({"error": "Interview not found"}), 404

    interview_data = interviews[interview_id]
    if question_index >= len(interview_data["questions"]):
        return jsonify({"completed": True})

    interview_data["current_question"] = question_index
    question = interview_data["questions"][question_index]
    return jsonify({"question": question})

@app.route("/submit_answer", methods=["POST"])
def submit_answer():
    interview_id = request.form.get("interview_id")
    question_index = int(request.form.get("question_index"))

    if interview_id not in interviews:
        return jsonify({"error": "Interview not found"}), 404

    audio = request.files.get("audio")
    if audio:
        # Save the audio file
        audio_format = audio.filename.rsplit(".", 1)[-1].lower()
        audio_path = f"/tmp/{uuid.uuid4()}.{audio_format}"
        audio.save(audio_path)
        logger.info(f"Saved audio file: {audio_path}, size: {os.path.getsize(audio_path)} bytes")

        # Transcribe the audio using Whisper
        transcribed_text = transcribe_audio(audio_path, format=audio_format)

        # Clean up the audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)

        if transcribed_text:
            interviews[interview_id]["answers"][question_index] = transcribed_text
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Failed to transcribe audio"}), 500
    else:
        text_answer = request.form.get("text_answer")
        if text_answer:
            interviews[interview_id]["answers"][question_index] = text_answer
            return jsonify({"success": True})
        return jsonify({"error": "No audio or text answer provided"}), 400

@app.route("/generate_feedback", methods=["POST"])
def generate_feedback():
    data = request.get_json()
    interview_id = data["interview_id"]

    if interview_id not in interviews:
        return jsonify({"error": "Interview not found"}), 404

    interview_data = interviews[interview_id]
    feedback = []

    for i, (question, answer) in enumerate(zip(interview_data["questions"], interview_data["answers"])):
        if answer:
            prompt = f"""
            You are an expert interviewer providing feedback on a candidate's response.
            Question: {question}
            Answer: {answer}
            Provide constructive feedback on the answer, focusing on clarity, relevance, and how well it addresses the question.
            """
            response = model.generate_content(prompt)
            feedback.append({"question": question, "answer": answer, "feedback": response.text})

    interview_data["feedback"] = feedback
    return jsonify({"success": True})

@app.route("/feedback/<interview_id>")
def feedback(interview_id):
    if interview_id not in interviews:
        return "Interview not found", 404
    interview_data = interviews[interview_id]
    return render_template("feedback.html", interview_id=interview_id, feedback=interview_data.get("feedback", []))

if __name__ == "__main__":
    app.run(debug=True)