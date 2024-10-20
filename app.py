from flask import Flask, request, render_template
from PyPDF2 import PdfReader
import re
import pickle
from docx import Document

app = Flask(__name__)
# Load models===========================================================================================================
# rf_classifier_categorization = pickle.load(open('Project 1\rf_classifier_categorization.pkl', 'rb'))
# text='Project 1\'
rf_classifier_categorization = pickle.load(open(r'Project 1\models\rf_classifier_categorization.pkl', 'rb'))
tfidf_vectorizer_categorization =pickle.load(open(r'Project 1\models\tfidf_vectorizer_categorization.pkl', 'rb'))
rf_classifier_job_recommendation = pickle.load(open(r'Project 1\models\rf_classifier_job_recommendation.pkl', 'rb'))
tfidf_vectorizer_job_recommendation =pickle.load(open(r'Project 1\models\tfidf_vectorizer_job_recommendation.pkl', 'rb'))

# Clean resume==========================================================================================================
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# Prediction functions==================================================================================================
def predict_category(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_categorization.transform([resume_text])
    predicted_category = rf_classifier_categorization.predict(resume_tfidf)[0]
    return predicted_category

def job_recommendation(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_job_recommendation.transform([resume_text])
    recommended_job = rf_classifier_job_recommendation.predict(resume_tfidf)[0]
    return recommended_job

def pdf_to_text(file):
    reader = PdfReader(file)
    text = ''
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

def docx_to_text(file):
    doc = Document(file)
    text = '\n'.join([para.text for para in doc.paragraphs])
    return text

# Resume parsing functions (extract phone, email, skills, education, and name)==========================================
def extract_contact_number_from_resume(text):
    pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    match = re.search(pattern, text)
    return match.group() if match else None

def extract_email_from_resume(text):
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text)
    return match.group() if match else None



# Routes===============================================================================================================
@app.route('/')
def resume():
    return render_template("resume.html")

@app.route('/pred', methods=['POST'])
def pred():
    if 'resume' in request.files:
        file = request.files['resume']
        filename = file.filename
        if filename.endswith('.pdf'):
            text = pdf_to_text(file)
        elif filename.endswith('.docx'):
            text = docx_to_text(file)
        elif filename.endswith('.txt'):
            text = file.read().decode('utf-8')
        else:
            return render_template('resume.html', message="Invalid file format. Please upload PDF, DOCX, or TXT.")

        predicted_category = predict_category(text)
        recommended_job = job_recommendation(text)
        phone = extract_contact_number_from_resume(text)
        email = extract_email_from_resume(text)
        # extracted_skills = extract_skills_from_resume(text)
        # extracted_education = extract_education_from_resume(text)
        # name = extract_name_from_resume(text)

        return render_template('resume.html', predicted_category=predicted_category, recommended_job=recommended_job,
                               phone=phone, email=email, 
                               )
    else:
        return render_template("resume.html", message="No resume file uploaded.")

if __name__ == '__main__':
    app.run(debug=True)
