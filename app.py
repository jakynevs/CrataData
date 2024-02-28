from urllib.parse import urlencode
from flask import Flask, redirect, request, render_template, session
import requests
from scraper import analyze_about_section

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = ""
    about_section_text = ""

    if request.method == 'POST':
        url = request.form['linkedin_url']
        result, about_section_text = analyze_about_section(url)
    return render_template('index.html', analysis_result=result, about_section=about_section_text)

if __name__ == '__main__':
    app.run(debug=True)
