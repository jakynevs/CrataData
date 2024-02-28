# Sustainability Classifier Web App

## Project Overview

This Flask web application automates the classification of companies based on their sustainability focus by analyzing the "About" section of their LinkedIn profiles. This solution aims to streamline the process of identifying potential investment opportunities in environmentally focused projects for an investment firm.

## Features

- **LinkedIn Profile Analysis**: Determines if a company is focused on sustainability based on its LinkedIn "About" section.
- **User-Friendly Interface**: Provides an easy-to-use interface for submitting LinkedIn URLs for analysis.
- **Immediate Classification Results**: Offers instant feedback on the sustainability focus of the analyzed company.

## Prerequisites

Before you begin, ensure you have:
- Python 3.8 or higher installed.
- Pip for installing Python packages.

## Installation

Follow these steps to set up the project locally:

1. **Clone the Repository**
    ```
    git clone https://github.com/jakynevs/CrataData.git
    cd CrataData
    ```

2. **Set Up a Virtual Environment** (optional but recommended)
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**
    ```
    pip install -r requirements.txt
    ```

4. **Chromedriver Setup**
    - Ensure you have Google Chrome installed on your machine.
    - Download the appropriate version of Chromedriver from [Chromedriver Downloads](https://sites.google.com/a/chromium.org/chromedriver/downloads) that matches your Google Chrome version.
    - Place the `chromedriver` executable in a known directory.
    - In `constants.py`, update the path to where you've stored `chromedriver`.

5. **Environment Variables**
    Create a `.env` file in the root directory and add:
    ```
    FLASK_APP=app.py
    FLASK_ENV=development
    ```

## Running the Application

To run the web app locally:

1. Activate your virtual environment if you haven't already.
2. Set the Flask application environment variables.
3. Run `flask run`.
4. Navigate to `http://127.0.0.1:5000/` in a web browser.

## How to Use

1. **Enter a LinkedIn URL**: On the homepage, input the full URL of a LinkedIn company profile.
2. **Submit for Analysis**: Click "Analyze". The result will indicate whether the company is sustainability-focused.

## Contributing

Feedback and contributions is greatly appreciated. Thanks for the opportunity to participate in this challenge.

## Proposed Next Steps

### Transition to LinkedIn API

- **Leverage LinkedIn API**: To enhance the sustainability classifier's reliability and maintain compliance with LinkedIn's policies, transitioning from using `chromedriver` for web scraping to utilizing LinkedIn's official API is recommended. The API provides structured access to company profiles, including detailed "About" sections, in a manner that respects user privacy and platform guidelines.

- **Implement OAuth for Authentication**: Integrate OAuth authentication to securely access LinkedIn's API. This will allow the application to fetch data programmatically while adhering to user consent protocols.

- **Data Enrichment**: Utilizing the API opens up possibilities for enriching the dataset with more detailed company information, such as industry, size, and other relevant metrics that could improve the model's predictions.

### User Feedback Loop

- **Feedback Mechanism**: Introduce a feature allowing users to provide immediate feedback on the prediction's accuracy. This could involve a simple "Correct" or "Incorrect" button alongside each prediction result.

- **Continuous Learning**: Use the collected feedback to further train and refine the model, incorporating a dynamic learning approach where the model evolves and improves over time based on user inputs. This could be implemented through periodic retraining cycles or more advanced online learning algorithms.

### Model Enhancement

- **Explore Advanced NLP Techniques**: Investigate the use of more advanced natural language processing (NLP) techniques and models, such as BERT or GPT, to improve the accuracy of classifying sustainability-focused companies. These models could provide deeper semantic understanding and context interpretation of company profiles.

- **Feature Engineering**: Dive deeper into feature engineering, exploring additional text features or external data sources that could provide more signals for the model, enhancing its predictive capabilities.

### User Experience Improvements

- **Enhance User Interface**: Further develop the web application's user interface to make it more intuitive and visually appealing. Consider implementing interactive elements that can guide users through the analysis process and display results in a more engaging format.

- **Accessibility and Internationalization**: Ensure the application is accessible to all users and consider internationalizing the UI to accommodate users from non-English speaking backgrounds.

These proposed next steps aim to not only improve the technical foundation of the application but also enhance the user experience and engagement, ensuring the tool remains valuable and relevant to the investment firm's evolving needs.
