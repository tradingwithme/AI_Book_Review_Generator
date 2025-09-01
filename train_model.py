import subprocess
import json

class BookReviewGenerator:
    def __init__(self, summary, reviews, book_title):
        self.summary = summary
        self.reviews = reviews
        self.book_title = book_title
        self.conversation_memory = self.initialize_conversation()

    def initialize_conversation(self):
        """Initialize the conversation memory with the book summary and reviews."""
        return [
            {"role": "system", "content": "You are a helpful assistant who generates book reviews."},
            {"role": "user", "content": f"Book Title: {self.book_title}"},
            {"role": "user", "content": f"Book Summary: {self.summary}"},
            {"role": "user", "content": "Reviews: " + " ".join(self.reviews)}
        ]
    
    def generate_initial_review(self):
        """Generate an initial review based on the summary and reviews."""
        self.conversation_memory.append({
            "role": "user",
            "content": "Can you provide an honest, natural review of the book? Include overall thoughts, strengths, weaknesses, and a rating out of 5."
        })
        return self._call_ollama()

    def refine_review(self, topic, content):
        """Refine the review by expanding on specific aspects like character development, pacing, etc."""
        self.conversation_memory.append({
            "role": "user",
            "content": f"Can you expand on {topic} in the book? {content}"
        })
        return self._call_ollama()

    def finalize_review(self):
        """Conclude and clean up the review, ensuring it has a natural flow."""
        self.conversation_memory.append({
            "role": "user",
            "content": "Can you provide a conclusion to the review and make sure all sections flow naturally together?"
        })
        return self._call_ollama()

    def _call_ollama(self):
        """Call Ollama API to generate the response from the conversation memory."""
        response = subprocess.run(
            ["ollama", "chat", "--input", json.dumps(self.conversation_memory)],
            capture_output=True,
            text=True
        )
        return response.stdout.strip()

    def generate_review(self):
        """Generate and refine the book review."""
        # Step 1: Generate initial review
        initial_review = self.generate_initial_review()
        print("Initial Review:\n", initial_review)

        # Step 2: Refine review with character development
        expanded_review = self.refine_review("character development", "Provide specific examples of how the characters are portrayed.")
        print("Expanded Review with Character Development:\n", expanded_review)

        # Step 3: Refine review with plot and pacing
        final_review = self.refine_review("plot and pacing", "How well does the story unfold, and are there any major plot twists?")
        print("Refined Review with Plot and Pacing:\n", final_review)

        # Step 4: Finalize the review with conclusion and natural flow
        final_review_with_conclusion = self.finalize_review()
        print("Final Review with Conclusion:\n", final_review_with_conclusion)

        return final_review_with_conclusion
