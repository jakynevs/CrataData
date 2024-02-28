from urllib.parse import urlencode
from flask import Flask, redirect, request, render_template, session
import requests
from scraper import scrape_and_analyse

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = ""
    about_section_text = ""

    if request.method == 'POST':
        url = request.form['linkedin_url']
        result, about_section_text = scrape_and_analyse(url)
    return render_template('index.html', scrape_and_analyse_result=result, about_section=about_section_text)

if __name__ == '__main__':
    app.run(debug=True)
