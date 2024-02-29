from constants import * 
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from analyser import analyse_text

def scrape_and_analyse(url):
    # specifies the path to the chromedriver.exe
    driver_path = dp
    service = Service(executable_path=driver_path)
    driver = webdriver.Chrome(service=service)  

    # Load website   
    driver.get(url)
    wait = WebDriverWait(driver, 10)  # Wait up to 10 seconds
    close_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@class="artdeco-icon lazy-loaded"]')))
    
    try:
        close_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@class="artdeco-icon lazy-loaded"]'))
        )
        print("Found close button. Clicking it...")
        close_button.click()
    except Exception as e:
        print("Close button not found or error clicking it:", e)

        # Example: Adjust the wait time and condition as needed for your specific case
    try:
        # Wait for the About section to be loaded
        about_section = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="main-content"]/section[1]/div/section[1]/div/p'))
        )
        about_section_text = about_section.text
    
    except Exception as e:
        result = "About section not found."
        print(e)

    result = analyse_text(about_section_text)
    
    return result, about_section_text

    

