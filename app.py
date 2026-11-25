
import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found!")

# Connect to Gemini
client = genai.Client(api_key=api_key)

# Send request
response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents="Explain Generative AI in simple terms with 3 examples."
)

# Display response
print("\n===== GEMINI AI RESPONSE =====\n")
print(response.text)
