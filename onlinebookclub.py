import os
import random
import platform
import subprocess
import pandas as pd
from PIL import Image
from json import dump
from time import sleep
from io import BytesIO
from random import sample
import concurrent.futures
from functools import wraps
from bs4 import BeautifulSoup
from selenium import webdriver
from IPython.display import display
from selenium_stealth import stealth
from selenium.common.exceptions import *
from langchain_community.llms import Ollama
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from alt_book_generator import BookReviewGenerator 
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait as wait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC

from gptFineTunedModel import fine_tune_model

def forceClick(driver, element):
    driver.execute_script("arguments[0].click();", element)

def install_ollama():
    """Install Ollama based on the operating system."""
    system_platform = platform.system().lower()
    install_script_url = "https://ollama.com/install.sh"
    
    if system_platform == 'linux' or system_platform == 'darwin':  # For Linux and macOS
        subprocess.run(f"curl -fsSL {install_script_url} -o install.sh", shell=True)
        subprocess.run("sh install.sh", shell=True)
        subprocess.run("nohup ollama serve &", shell=True)
        subprocess.run("ollama pull tinyllama", shell=True)
    elif system_platform == 'windows':  # For Windows, Ollama setup might be different
        subprocess.run(f"curl -fsSL {install_script_url} -o install.ps1", shell=True)
        subprocess.run("powershell install.ps1", shell=True)
        subprocess.run("start-Process -NoNewWindow -FilePath ollama serve", shell=True)
        subprocess.run("ollama pull tinyllama", shell=True)

def get_ollama_client():
    """Return Ollama client instance."""
    return Ollama(model="tinyllama", base_url="http://127.0.0.1:11434")

def getDriver(disable_gpu=False, headless_mode=False, background_mode=False):
    """
    Returns a Selenium WebDriver instance based on the given settings.
    Supports cross-platform operation for Linux, macOS, and Windows.
    """
    # Handle background and headless modes
    if disable_gpu or headless_mode:
        background_mode = True
    if background_mode:
        options = Options()
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-setuid-sandbox')
        if headless_mode:
            options.add_argument('--headless')
        if disable_gpu:
            options.add_argument('--disable-gpu')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--remote-debugging-port=9222')
    # Set User-Agent based on the operating system
    user_agent = [
"Mozilla/5.0 (Linux; Android 4.4.4; [HM NOTE|NOTE-III|NOTE2 1LTET) AppleWebKit/537.39 (KHTML, like Gecko)  Chrome/53.0.2111.335 Mobile Safari/536.6",
"Mozilla / 5.0 (compatible; MSIE 8.0; Windows; U; Windows NT 6.2; x64; en-US Trident / 4.0)",
"Mozilla/5.0 (iPhone; CPU iPhone OS 8_0_8; like Mac OS X) AppleWebKit/536.13 (KHTML, like Gecko)  Chrome/50.0.2440.333 Mobile Safari/600.1",
"Mozilla/5.0 (iPod; CPU iPod OS 11_1_7; like Mac OS X) AppleWebKit/603.49 (KHTML, like Gecko)  Chrome/51.0.1709.273 Mobile Safari/533.6"
]
    options.add_argument('--user-agent=%s' % random.choice(user_agent))
    # Determine the appropriate path for ChromeDriver based on OS
    chrome_driver_path = ""
    system_platform = platform.system().lower()
    if system_platform == 'windows':
        chrome_driver_path = os.path.join(os.getcwd(), 'chromedriver_win.exe')  # Ensure you have chromedriver_win.exe
    elif system_platform == 'linux':
        chrome_driver_path = '/usr/bin/chromedriver'  # Make sure chromedriver is installed in the correct location
    elif system_platform == 'darwin':  # macOS
        chrome_driver_path = '/usr/local/bin/chromedriver'  # Correct path for macOS (or where chromedriver is installed)
    # Set up the WebDriver and Service for cross-platform support
    if background_mode:
        if chrome_driver_path:  # When ChromeDriver path is provided, use it
            service = Service(chrome_driver_path)
            driver = webdriver.Chrome(service=service, options=options)
        else: driver = webdriver.Chrome(options=options)  # Defaults to system path if no path is specified
    else: driver = webdriver.Chrome(options=options)  # Regular mode, without background/headless
    # Apply Stealth for anti-bot measures
    stealth(driver, languages=['en-US', 'en'], vendor="Google Inc.", platform="Win32", webgl_vendor="Intel Inc.", render="Intel Iris OpenGL Engine", fix_hairline=True)
    return driver
def wrapper(func):
    @wraps(func) 
    def inner_wrapper(*args, **kwargs):
        for attempt in range(6):  # Retry 6 times
            try: return func(*args, **kwargs)
            except KeyboardInterrupt: raise
            except (ElementClickInterceptedException, ElementNotInteractableException):
                for elem in visElement(driver, (By.CSS_SELECTOR, '[data-testid="close-icon-button"]'), showAny=True, selectOne=False): # Close the element if it's blocking interaction
                    forceClick(driver, elem)
    return inner_wrapper
@wrapper
def sendQuery(driver,text:str):
  previous = getHTML(driver).select('[id="ai-chat-assistance-last-replay"]')[-1].parent.find('span').text if getHTML(driver).select('[id="ai-chat-assistance-last-replay"]') else None
  element = visElement(driver,(By.CSS_SELECTOR,'textarea'),showAny=True)
  if element:
    element.clear()
    #element.send_keys(text)
    assignText(driver,element,text)
    element.send_keys(' ')
    element.send_keys(Keys.ENTER)
    sleep(1.5)
    visElement(driver,(By.CSS_SELECTOR,'[data-testid="quill-chat-send-button"]'),showAny=True,selectOne=False);
    while not 'background' in getHTML(driver).select_one('[data-testid="quill-chat-send-button"]').get('style'): sleep(1.5)
    if getHTML(driver).select('[id="ai-chat-assistance-last-replay"]'): 
      visElement(driver,(By.CSS_SELECTOR,'[data-testid="quill-chat-send-button"]'),showAny=True,selectOne=False);
      if previous != getHTML(driver).select('[id="ai-chat-assistance-last-replay"]')[-1].parent.find('span').text:
        return getHTML(driver).select('[id="ai-chat-assistance-last-replay"]')[-1].parent.find('span').text
@wrapper
def getNewChat(driver):
  if any(i.select('svg[cursor="pointer"]') and not i.select('[clip-rule]') for i in getHTML(driver).select('button[class*="MuiIconButton-sizeMedium"]')):
    for i in driver.find_elements(By.CSS_SELECTOR,'button[class*="MuiIconButton-sizeMedium"]'): 
      html = getWebHTML(i)
      if html.select('svg[cursor="pointer"]') and not html.select('[clip-rule]'):
        i.find_element(By.XPATH,'./..').click()
        sleep(1.5)
        break
def isChecked(driver,element): driver.execute_script("return arguments[0].checked;", element)
def displayWebElement(webelement):
  if hasattr(webelement, 'screenshot_as_png'): display(Image.open(BytesIO(webelement.screenshot_as_png)))
  else: display(Image.open(BytesIO(webelement.get_screenshot_as_png())))
def getWebHTML(element): return BeautifulSoup(element.get_attribute('outerHTML'), 'html.parser')
def find_element_by_attribute_value(soup, text):
    found_elements = []
    for tag in soup.find_all():
        for attr, value in tag.attrs.items():
            if isinstance(value, str) and text in value:
                found_elements.append(tag)
                break  # Stop checking attributes for this tag once a match is found
    return found_elements
def scroll_to_and_focus_element(driver, element):
    driver.execute_script("arguments[0].scrollIntoView();", element)
    sleep(0.5)
    driver.execute_script("arguments[0].style.border='3px solid red';", element)
def bringToFocus(driver, element):
    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
    driver.execute_script("arguments[0].focus();", element)
def sanitizeText(Text):
    return ' '.join(j for i in Text.splitlines() for j in i.split(' ') if j!='')
def getHTML(driver): return BeautifulSoup(driver.page_source,'html.parser')
def bringToFocus(driver, element):
  for i in range(15):
      driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
      driver.execute_script("arguments[0].focus();", element)
      scroll_to_and_focus_element(driver,element)
      if element.is_displayed() and element.is_enabled(): break
      else: sleep(1.1)
def clearEntry(driver,element):
    driver.execute_script("arguments[0].scrollIntoView(true);",element)
    for i in range(3):
      element.send_keys(Keys.CONTROL + 'a')
      element.send_keys(Keys.DELETE)
      element.send_keys(Keys.CONTROL + 'a')
      element.send_keys(Keys.BACKSPACE)
      sleep(0.1)
      ActionChains(driver).key_down(Keys.CONTROL).send_keys_to_element(element,'a').key_up(Keys.CONTROL).send_keys_to_element(element,Keys.BACKSPACE).perform()
def visElement(driver,cssSelector,timer=10,showAny=False,showAll=False,selectOne=True):
  driver.implicitly_wait(timer)
  css_caller, css_element = cssSelector
  if (css_caller == By.CSS_SELECTOR and getHTML(driver).select(css_element)) or driver.find_elements(css_caller,css_element):
    if showAny:
      elements = wait(driver,timer).until(EC.visibility_of_any_elements_located((css_caller,css_element)))
      return elements if selectOne is False else elements[-1]
    elif showAll:
      elements = wait(driver,timer).until(EC.visibility_of_all_elements_located((css_caller,css_element)))
      return elements if selectOne is False else elements[-1]
    else: return wait(driver,timer).until(EC.presence_of_element_located((css_caller,css_element)))
  else:
    print('Element not found')
    return None
def assignText(driver,element,text):
    escaped_text = text.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n').replace('\r', '\\r')
    driver.execute_script(f"arguments[0].value = '{escaped_text}';", element)
    driver.execute_script(f"arguments[0].textContent = '{escaped_text}';", element)
def getLast(driver,cssSelector):
  driver.implicitly_wait(15)
  css_caller, css_element = cssSelector
  return driver.find_elements(css_caller,css_element)[-1]
def cleanReview(driver):
  return ' '.join([i for i in visElement(driver,(By.CLASS_NAME,
'content')).text.splitlines()[1:][:-4] if len(sanitizeText(i).split())>7])
def nextPage(driver):
  if visElement(driver,(By.CSS_SELECTOR,'button[aria-label="Next page"]'),showAny=True,selectOne=False):
    visElement(driver,(By.CSS_SELECTOR,'button[aria-label="Next page"]')).click()
    sleep(1.6)

reviewTexts, bookSummaries, final_reviews = [], [], []
url = 'https://forums.onlinebookclub.org/ucp.php?mode=login&redirect=misc%2Fobc_redirect.php%3Fext%3Dindex.php&sid=3c4fb11f781e0eefbd12e7c538c83046'
url2 = 'https://quillbot.com/ai-writing-tools/ai-book-review-generator'
if not 'onlineBookClub_username' in os.environ: 
  os.environ['onlineBookClub_username'] = input('Enter your username for the OnlineBookClub website. Your username will be saved to your local environment: ')
  onlineBookClub_username = os.environ['onlineBookClub_username']
else: onlineBookClub_username = os.get("onlineBookClub_username")
if not 'onlineBookClub_password' in os.environ: 
  os.environ['onlineBookClub_password'] = input('Enter your password for the OnlineBookClub website. Your username will be saved to your local environment: ')
  onlineBookClub_password = os.environ['onlineBookClub_password']
else: onlineBookClub_password = os.get("onlineBookClub_password")
if not 'QuillBot_username' in os.environ: 
  os.environ['QuillBot_username'] = input('Enter your username for the QuillBot website. Your username will be saved to your local environment: ')
  QuillBot_username = os.environ['QuillBot_username']
else: QuillBot_username = os.get("QuillBot_username")
if not 'QuillBot_password' in os.environ: 
  os.environ['QuillBot_password'] = input('Enter your password for the QuillBot website. Your username will be saved to your local environment: ')
  QuillBot_password = os.environ['QuillBot_password']
else: QuillBot_password = os.get("QuillBot_password")
headless_mode = input('Would you like to enable headless mode (Your browser will run in the background)? ')
if sanitizeText(headless_mode)[0].lower() == 'y': headless_mode = True
else: headless_mode = False
driver = getDriver(headless_mode=headless_mode)
driver.get(url)
driver.switch_to.new_window('tab')
driver.switch_to.window(driver.window_handles[0])
driver.get('https://quillbot.com/login')
for attempt in range(15):
  username = visElement(driver,(By.NAME,'username'))
  if visElement(driver,(By.NAME,'username'),showAny=True,selectOne=False):
    password = visElement(driver,(By.NAME,'password'))
    loginButton = driver.find_element(By.CSS_SELECTOR,'[data-testid="login-btn"]')
    if attempt >= 3:
      os.environ['QuillBot_username'] = input('Enter your username for the QuillBot website. Your username will be saved to your local environment: ')
      QuillBot_username = os.environ['QuillBot_username']
      os.environ['QuillBot_password'] = input('Enter your password for the QuillBot website. Your username will be saved to your local environment: ')
      QuillBot_password = os.environ['QuillBot_password']
    username.clear()
    clearEntry(driver,username)
    username.send_keys(os.get(QuillBot_username))
    sleep(1.5)
    assignText(driver,username,QuillBot_username)
    password.clear()
    clearEntry(driver,password)
    password.send_keys(QuillBot_password)
    sleep(1.5)
    assignText(driver,password,os.get(QuillBot_password))
    loginButton.click()
  else: break
if visElement(driver,(By.NAME,'username'),showAny=True,selectOne=False): raise Exception('Unsuccessful login!')
driver.get(url2)
driver.implicitly_wait(15)

def cleanReview(driver):
  return ' '.join([i for i in visElement(driver,(By.CLASS_NAME,
'content')).text.splitlines()[1:][:-4] if len(sanitizeText(i).split())>7])

from json import load
with open('bookSummaries.json','r',encoding='utf-16') as file: bookSummaries = load(file)
with open('bookReviews.json','r',encoding='utf-16') as file: reviewTexts2 = load(file)
bookContent = pd.DataFrame(bookSummaries,columns=['bookTitle','bookContent'])
bookContent['bookTitle'] = bookContent['bookTitle'].apply(sanitizeText)
bookContent = bookContent[bookContent['bookContent'].apply(lambda x: len(str(x).split())>10)].reset_index(drop=True)
bookContent.drop_duplicates().reset_index(drop=True,inplace=True)
bookReviews = pd.DataFrame(reviewTexts2,columns=['bookTitle','reviews'])
bookReviews['bookTitle'] = bookReviews['bookTitle'].apply(sanitizeText)
driver.implicitly_wait(15)
bookUrls = [i.get('href') for i in getHTML(driver).select('[href]') if 'book.php' in i.get('href')]
from random import sample
for bookUrl in bookUrls:
  driver.switch_to.window(driver.window_handles[0])
  reviews, summary, pageText = [], '', []
  for attempt in range(6):
    try:
      driver.get(bookUrl)
      driver.implicitly_wait(15)
      bookTitle = getHTML(driver).select_one('div[id="content"]').find('h1').text
      break
    except KeyboardInterrupt: raise KeyboardInterrupt
    except: sleep(1.5)
  if not bookReviews.bookTitle.isin([bookTitle]).any():
    reviewUrls = ['https://forums.onlinebookclub.org'+i.get('href') for i in getHTML(driver).select('a[href]') if i.get('href').startswith('/viewtopic')]
    for reviewUrl in sample(reviewUrls,2 if len(reviewUrls)>=2 else 1):
      driver.get(reviewUrl)
      driver.implicitly_wait(15)
      reviewText = cleanReview(driver)
      #reviewTexts.append((bookTitle,reviewText))
      reviews.append(sanitizeText(reviewText))
    for attempt in range(6):
      try:
        driver.get(bookUrl)
        driver.implicitly_wait(15)
        break
      except KeyboardInterrupt: raise KeyboardInterrupt
      except: sleep(1.5)
  else: reviews = bookReviews[bookReviews['bookTitle']==bookTitle]['reviews'].apply(sanitizeText).sample(2).tolist()
  if not bookContent.bookTitle.isin([bookTitle]).any():
      amazonLinks = [i.get('href') for i in getHTML(driver).select('[href]') if 'kindle' in i.text.lower()]
      if amazonLinks:
        driver.get(amazonLinks[0])
        driver.implicitly_wait(15)
        amazonLinks2 = [i.get('href') for i in getHTML(driver).select('[href]') if 'amazon' in i.text.lower()]
        if amazonLinks2:
          driver.get(amazonLinks2[0])
          driver.implicitly_wait(15)
          if visElement(driver,(By.CSS_SELECTOR,'[id^="ebooksReadSampleButton"]'),showAny=True,selectOne=False): visElement(driver,(By.CSS_SELECTOR,'[id^="ebooksReadSampleButton"]')).click()
          previousText = None
          while visElement(driver,(By.CSS_SELECTOR,'[id="mainContainer"]'),showAny=True,selectOne=False):
            wait(driver, 10).until(EC.invisibility_of_element_located((By.CSS_SELECTOR, 'ion-backdrop')))
            page_text = visElement(driver,(By.CSS_SELECTOR,'[id="mainContainer"]'))
            page_text = page_text.text if page_text else page_text
            if page_text is not None and previousText == page_text: break
            if page_text is not None:
              if len(sanitizeText(page_text)):
                pageText.append(page_text)
              previousText = page_text
            try: nextPage(driver)
            except: break
          #bookSummaries.append((bookTitle,'\n'.join(pd.Series(pageText).drop_duplicates().values).strip()))
          summary = ' '.join(pd.Series(pageText).drop_duplicates().values).strip()
          summary = summary if len(summary.split())>10 else ''
  else: summary = bookContent[bookContent['bookTitle']==bookTitle]['bookContent'].values[0]
  bookSummaries.append((bookTitle,summary))
  for review in reviews: reviewTexts.append((bookTitle,review))
  driver.switch_to.window(driver.window_handles[1])
  driver.implicitly_wait(15)
  getNewChat(driver)
  stringTexts = []
  if len(summary): stringTexts.append(f'keep the following in memory for all subsequent prompts/queries: [{summary}]')
  for review in reviews: stringTexts.append(f'Additionally, keep the following in memory for all subsequent prompts/queries: [{review}]')
  for stringText in stringTexts: sendQuery(driver,stringText)
  final_review = sendQuery(driver,"Given everything prior, provide a honest, human-like review that sounds natural in plain text (make sure to rate it as well on a scale of 5): " + bookTitle)
  if final_review: 
    final_reviews.append((bookTitle,final_review))
    reviewTexts2.append((bookTitle,final_review))
    with open('bookReviews.json','w',encoding='utf-16') as file: dump(reviewTexts2,file)
  else: 
    print('Free trial ended. Try again 24 hours from now.')
    # Step 1: Check if Ollama is installed; install if not
    try: subprocess.run("ollama --version", shell=True, check=True)
    except subprocess.CalledProcessError:
        print("Ollama not found, installing it now...")
        install_ollama()
    # Step 2: Use Ollama model to generate reviews
    ollama_llm = get_ollama_client()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_fine_tune = executor.submit(fine_tune_model, bookTitle, summary, reviews, generate=False)
        future_generate = executor.submit(BookReviewGenerator, ollama_llm, summary, reviews, bookTitle)

        # Wait for both operations to finish and get the results
        fine_tune_review = future_fine_tune.result()
        generated_review = future_generate.result()
    #review_generator = BookReviewGenerator(ollama_llm, summary, reviews, bookTitle)
    #final_review = review_generator.generate_review()
    print("\nFinal Generated Book Review:\n")
    print(generated_review)
    final_reviews.append((bookTitle,final_review))
    reviewTexts2.append((bookTitle,final_review))
    with open('bookReviews.json','w',encoding='utf-16') as file: dump(reviewTexts2,file)
for bookTitle, final_review in final_reviews: print(bookTitle,': \n',final_review)