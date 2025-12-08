import google.generativeai as genai
import fitz, docx, json

genai.configure(api_key="AIzaSyDdgfCWEz_8Jq21HfKvL044fmlEfHKPnFc")

def read_file_text(file_path):
    text = ""
    ext = file_path.lower().split(".")[-1]
    if ext == "pdf":
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text += page.get_text() + "\n"
    elif ext == "docx":
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text

def extract_metadata_with_gemini(file_path):
    resume_text = read_file_text(file_path)

    prompt = f"""
    Extract the following metadata from this resume:
    - File Name
    - Name
    - Email
    - Phone Number
    - Education: UG (degree + college), PG (degree + college), PhD (topic + institution if available)
    - Years of Experience: positions + years + location
    - Publications: journal count + names, conference count + names
    - Workshops/programs conducted: count + names
    Return the answer in strict JSON format.
    Resume Text:
    {resume_text}
    """

    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    response = model.generate_content(prompt)

    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON from model", "raw_output": response.text}

file_path = r"C:\Users\Shubha\Downloads\converted_docx_files\Dr Rakesh Kumar AI + Math CV.docx"
metadata = extract_metadata_with_gemini(file_path)
print(json.dumps(metadata, indent=4))
