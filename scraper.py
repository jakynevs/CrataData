from constants import * 
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from analyser import predict_sustainability

def scrape_and_analyse(url):
    WAIT_TIME = 10  # Centralized wait time

    # Specifies path to the chromedriver.exe
    driver_path = dp
    service = Service(executable_path=driver_path)
    driver = webdriver.Chrome(service=service)  

    try:
        # Load website   
        driver.get(url)
        wait = WebDriverWait(driver, WAIT_TIME) 

        # Attempt to close pop-up
        try:
            close_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@class="artdeco-icon lazy-loaded"]')))
            close_button.click()

        except (NoSuchElementException, TimeoutException) as e:
            print("Close button not found or error clicking it:", e)
    
        # Wait for the About section to be loaded
        about_section = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="main-content"]/section[1]/div/section[1]/div/p')))
        about_section_text = about_section.text

        # Send to model for prediction
        result = predict_sustainability(about_section_text)

    except Exception as e:
        print("An error occurred:", e)
        result = "Error processing page."
    
    finally:
        driver.quit()

    return result, about_section_text

    

