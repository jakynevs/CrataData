from flask import Flask, request, render_template, session
from scraper import scrape_and_analyse

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    result_text = ""
    about_section_text = ""

    if request.method == 'POST':
        url = request.form['linkedin_url']
        result, about_section_text = scrape_and_analyse(url)
        if result[0] == 1:
            result_text = 'Yes'
        else:
            result_text = 'No'

    return render_template('index.html', scrape_and_analyse_result=result_text, about_section=about_section_text)

if __name__ == '__main__':
    app.run(debug=True)
