# app.py
import os
import tempfile
import time
from flask import Flask, request, jsonify, render_template, send_from_directory
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import speech_recognition as sr
from gtts import gTTS
import PyPDF2
import docx
import json
import uuid
from pydub import AudioSegment

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['INTERVIEW_DATA'] = 'interview_data'

# Create required directories if they don't exist
def ensure_directories():
    with app.app_context():
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['INTERVIEW_DATA'], exist_ok=True)
        os.makedirs('static/audio', exist_ok=True)

ensure_directories()

# Ensure directories exist before first request
# Right after app configuration, before routes
# Create directories immediately at startup
with app.app_context():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['INTERVIEW_DATA'], exist_ok=True)
    os.makedirs('static/audio', exist_ok=True)

# Configure Gemini API
def configure_genai(api_key):
    genai.configure(api_key=api_key)

# Initialize with environment variable if available
try:
    if api_key := os.environ.get('GEMINI_API_KEY'):
        configure_genai(api_key)
    else:
        print("Warning: GEMINI_API_KEY not set in environment")
except Exception as e:
    print(f"Critical Gemini initialization error: {str(e)}")
    raise  # Force crash during startup if config fails

# Text extraction functions
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_resume(file_path):
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_ext in ['.docx', '.doc']:
        return extract_text_from_docx(file_path)
    elif file_ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    else:
        return ""

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/setup', methods=['POST'])
def setup_interview():
    if 'api_key' in request.form and request.form['api_key']:
        api_key = request.form['api_key']
        configure_genai(api_key)
    elif not os.environ.get('GEMINI_API_KEY'):
        return jsonify({'error': 'No API key provided'}), 400
    
    # Generate a unique interview ID
    interview_id = str(uuid.uuid4())
    interview_data = {
        'id': interview_id,
        'resume_text': '',
        'job_title': request.form.get('job_title', ''),
        'custom_questions': request.form.get('custom_questions', ''),
        'interview_level': request.form.get('interview_level', 'intermediate'),
        'interview_type': request.form.get('interview_type', 'general'),
        'questions': [],
        'responses': [],
        'feedback': ''
    }
    
    # Process resume if uploaded
    if 'resume' in request.files:
        resume_file = request.files['resume']
        if resume_file.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{interview_id}_{resume_file.filename}')
            resume_file.save(file_path)
            interview_data['resume_text'] = extract_text_from_resume(file_path)
    
    # Generate interview questions based on resume and parameters
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            },
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        )
        
        prompt = f"""
        Act as a professional job interviewer for a {interview_data['interview_level']} level {interview_data['job_title']} position.
        
        Here is the candidate's resume:
        {interview_data['resume_text']}
        
        Additional custom questions or topics to include:
        {interview_data['custom_questions']}
        
        Please generate 5-7 relevant interview questions for a {interview_data['interview_type']} interview.
        Format your response as a JSON array of strings, each containing one question. 
        Do not include any explanations or other text outside the JSON array.
        """
        
        response = model.generate_content(prompt)
        questions_text = response.text.strip()
        
        # Extract the JSON array from the response
        if '```json' in questions_text:
            questions_text = questions_text.split('```json')[1].split('```')[0].strip()
        elif '```' in questions_text:
            questions_text = questions_text.split('```')[1].strip()
            
        # Clean up any non-JSON content
        if not questions_text.startswith('['):
            questions_text = questions_text[questions_text.find('['):]
        if not questions_text.endswith(']'):
            questions_text = questions_text[:questions_text.rfind(']')+1]
            
        interview_data['questions'] = json.loads(questions_text)
    except Exception as e:
        print(f"Error generating questions: {e}")
        # Fallback questions if generation fails
        interview_data['questions'] = [
            f"Tell me about your experience related to {interview_data['job_title']}.",
            "What are your strengths and weaknesses?",
            "Describe a challenging project you worked on.",
            "Why are you interested in this position?",
            "Where do you see yourself in five years?"
        ]
    
    # Save interview data
    with open(os.path.join(app.config['INTERVIEW_DATA'], f'{interview_id}.json'), 'w') as f:
        json.dump(interview_data, f)
    
    return jsonify({
        'interview_id': interview_id,
        'questions': interview_data['questions']
    })

@app.route('/interview/<interview_id>')
def interview_page(interview_id):
    try:
        with open(os.path.join(app.config['INTERVIEW_DATA'], f'{interview_id}.json'), 'r') as f:
            interview_data = json.load(f)
        return render_template(
            'interview.html',
            interview_id=interview_id,
            job_title=interview_data['job_title'],
            questions=interview_data['questions']
        )
    except Exception as e:
        return f"Interview not found or error: {str(e)}", 404

@app.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.json
    interview_id = data.get('interview_id')
    question_index = int(data.get('question_index', 0))
    
    try:
        with open(os.path.join(app.config['INTERVIEW_DATA'], f'{interview_id}.json'), 'r') as f:
            interview_data = json.load(f)
        
        if question_index < len(interview_data['questions']):
            question = interview_data['questions'][question_index]
            audio_filename = f"{interview_id}_question_{question_index}.mp3"
            audio_path = os.path.join('static/audio', audio_filename)
            
            # Generate TTS for the question
            tts = gTTS(text=question, lang='en')
            tts.save(audio_path)
            
            return jsonify({
                'question': question,
                'audio_url': f'/static/audio/{audio_filename}',
                'question_index': question_index
            })
        else:
            return jsonify({
                'completed': True,
                'message': "Interview completed. Generating feedback..."
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    interview_id = request.form.get('interview_id')
    question_index = int(request.form.get('question_index', 0))
    
    try:
        # Process the audio answer
        if 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file.filename != '':
                # Save the audio file temporarily as WebM
                with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp:
                    audio_file.save(temp.name)
                    temp_filename = temp.name
                
                # Convert WebM to PCM WAV using pydub
                audio = AudioSegment.from_file(temp_filename, format='webm')
                wav_filename = temp_filename.replace('.webm', '.wav')
                audio.export(wav_filename, format='wav')
                
                # Convert speech to text with the WAV file
                recognizer = sr.Recognizer()
                with sr.AudioFile(wav_filename) as source:
                    audio_data = recognizer.record(source)
                    try:
                        answer_text = recognizer.recognize_google(audio_data)
                    except Exception as e:
                        print(f"Speech recognition error: {e}")
                        answer_text = "[Unable to transcribe audio]"
                
                # Clean up temporary files
                os.unlink(temp_filename)
                os.unlink(wav_filename)
        else:
            # If no audio was provided, use text input
            answer_text = request.form.get('text_answer', '')
        
        # Update interview data
        with open(os.path.join(app.config['INTERVIEW_DATA'], f'{interview_id}.json'), 'r') as f:
            interview_data = json.load(f)
        
        # Save the response
        interview_data['responses'].append({
            'question_index': question_index,
            'question': interview_data['questions'][question_index],
            'answer': answer_text
        })
        
        with open(os.path.join(app.config['INTERVIEW_DATA'], f'{interview_id}.json'), 'w') as f:
            json.dump(interview_data, f)
        
        return jsonify({'success': True, 'next_question_index': question_index + 1})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_feedback', methods=['POST'])
def generate_feedback():
    interview_id = request.json.get('interview_id')
    
    try:
        with open(os.path.join(app.config['INTERVIEW_DATA'], f'{interview_id}.json'), 'r') as f:
            interview_data = json.load(f)
        
        # Format Q&A for feedback generation
        qa_pairs = []
        for response in interview_data['responses']:
            qa_pairs.append(f"Question: {response['question']}\nAnswer: {response['answer']}")
        
        qa_text = "\n\n".join(qa_pairs)
        
        # Generate feedback using Gemini
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 4096,
            }
        )
        
        prompt = f"""
        You are an expert job interview coach. Please analyze this mock interview for a {interview_data['interview_level']} level {interview_data['job_title']} position.
        
        Here is the candidate's resume:
        {interview_data['resume_text']}
        
        Here are the questions and answers from the interview:
        {qa_text}
        
        Please provide comprehensive feedback on:
        1. Overall performance and impression
        2. Strengths demonstrated in the responses
        3. Areas for improvement
        4. Specific advice for each question/answer
        5. General interview technique (clarity, conciseness, relevance)
        
        Format your feedback in a structured, easy-to-read format with clear sections and bullet points where appropriate.
        """
        
        response = model.generate_content(prompt)
        feedback = response.text
        
        # Save the feedback
        interview_data['feedback'] = feedback
        with open(os.path.join(app.config['INTERVIEW_DATA'], f'{interview_id}.json'), 'w') as f:
            json.dump(interview_data, f)
        
        return jsonify({'feedback': feedback})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feedback/<interview_id>')
def feedback_page(interview_id):
    try:
        with open(os.path.join(app.config['INTERVIEW_DATA'], f'{interview_id}.json'), 'r') as f:
            interview_data = json.load(f)
        
        return render_template(
            'feedback.html',
            interview_id=interview_id,
            job_title=interview_data['job_title'],
            feedback=interview_data['feedback'],
            qa_pairs=[(r['question'], r['answer']) for r in interview_data['responses']]
        )
    except Exception as e:
        return f"Interview data not found or error: {str(e)}", 404

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
