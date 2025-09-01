<h1>AI Book Review Generator</h1>

<p>This notebook contains an AI-powered tool for generating book reviews by leveraging web scraping, external APIs (QuillBot and Ollama), and a custom review generation class.</p>

<h2>Setup</h2>

<p>Before running the notebook, ensure you have the necessary libraries installed. You can install them using the provided `requirements.txt` file:</p>

<pre><code>pip install -r requirements.txt</code></pre>

<p>You will also need to provide your login credentials for OnlineBookClub and QuillBot when prompted. Additionally, the script attempts to install and use Ollama if QuillBot's free trial is exhausted.</p>

<h2>Algorithm Steps</h2>

<ol>
  <li><strong>Load Data:</strong> The notebook attempts to load existing book summaries and reviews from `bookSummaries.json` and `bookReviews.json`.</li>
  <li><strong>Initialize WebDriver:</strong> A Selenium WebDriver instance is initialized with options for headless mode and anti-bot measures using `selenium-stealth`.</li>
  <li><strong>Login to Websites:</strong> The script navigates to the QuillBot and OnlineBookClub login pages and attempts to log in using the provided credentials.</li>
  <li><strong>Scrape Book Information:</strong> The script scrapes book URLs from OnlineBookClub. For each book, it attempts to:
    <ul>
      <li>Scrape existing reviews from the OnlineBookClub forums.</li>
      <li>Scrape book content (sample text) from Amazon if links are available.</li>
    </ul>
  </li>
  <li><strong>Generate Reviews:</strong>
    <ul>
      <li><strong>Using QuillBot:</strong> The script interacts with the QuillBot AI Book Review Generator, providing book summaries and scraped reviews as context to generate a final review.</li>
      <li><strong>Using Ollama:</strong> If the QuillBot free trial is unavailable, the script falls back to using an Ollama model (tinyllama) via the `BookReviewGenerator` class to generate the review based on the collected summary and reviews.</li>
    </ul>
  </li>
  <li><strong>Save Reviews:</strong> The generated reviews are appended to the `bookReviews.json` file.</li>
</ol>

<h2>Usage</h2>

<p>Run the cells in the notebook sequentially. You will be prompted to enter your login credentials for the required websites and whether to enable headless browser mode.</p>

<h2>BookReviewGenerator Class</h2>

<p>The `BookReviewGenerator` class is a Python class designed to generate book reviews using an Ollama model. It takes a book summary, a list of existing reviews, and the book title as input. It initializes a conversation memory with this information and provides methods to generate an initial review, refine the review based on specific topics (like character development or plot), and finalize the review for a natural flow.</p>

</body>
</html>
