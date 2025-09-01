import subprocess

class BookReviewGenerator:
    def __init__(self, summary: str, reviews: list, bookTitle: str):
        self.summary = summary
        self.reviews = reviews
        self.bookTitle = bookTitle
        self.model_ready = False
        self.memory = []  # Initialize memory for conversation
        
        # Ensure dependencies are set up on initialization
        self.ensure_dependencies()

    def ensure_dependencies(self):
        """Ensure Ollama is installed and model is available."""
        # ... (same as before)
        pass

    def generate_review(self):
        """Generate a review based on the book's summary and reviews."""
        if not self.model_ready:
            raise Exception("Dependencies not set up. Make sure Ollama is installed and the model is pulled.")
        
        # Initialize memory with book context (summary and reviews)
        self.memory.append(f"Book Title: {self.bookTitle}")
        if len(self.summary) > 100: self.memory.append(f"Book Summary: {self.summary}")
        
        # Add reviews to the memory as well
        for review in self.reviews:
            self.memory.append(f"Review: {review}")
        
        # Use memory to generate a more conversational review
        conversation = "\n".join(self.memory)

        # Construct the input prompt for the conversation chain (using memory)
        prompt = f"""
        You are a helpful assistant who reviews books based on the provided summary and reviews.
        The following details are for the book:

        {conversation}

        Please provide a thoughtful and honest review for the book. Make sure to include any pros and cons, and give a rating out of 5. Use the previous information to guide your review.
        """

        # Simulate the review generation process (using Ollama or another API here)
        result = subprocess.run(
            ["ollama", "chat", "-i", prompt], 
            capture_output=True, 
            text=True
        )

        # Handle the generated review
        if result.returncode == 0:
            review = result.stdout.strip()  # Assuming the output is the review text
            return review
        else:
            raise Exception("Error generating the review.")
