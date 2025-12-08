from flask import Flask, request, render_template, redirect, url_for
import os

app = Flask(__name__)

# Your PDF QA logic (modified code from your project here)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    pdf_files = request.files.getlist('pdfs')
    if pdf_files:
        ingest_pdfs(pdf_files)
    return redirect(url_for('home'))

@app.route('/ask', methods=['POST'])
def ask():
    query = request.form['query']
    results = query_faiss(query)
    context = "\n\n".join(results)
    prompt = f"""
    You are a helpful assistant. Use the following extracted document snippets to answer the question.

    Context:
    {context}

    Question: {query}

    Answer:
    """
    answer = generate_answer_with_openai(prompt)
    return render_template('index.html', answer=answer, snippets=results)

if __name__ == '__main__':
    app.run(debug=True)
