# main.py
import os
from groq import Groq, APIError
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

def main():
    """
    Main function to interact with the Groq API.
    It initializes the client, sends a chat completion request,
    and prints the streaming response.
    """
    try:
        # The Groq client will now automatically find the GROQ_API_KEY
        # loaded from your .env file.
        client = Groq()

        # The user's prompt. You can change this to test different inputs.
        user_prompt = "Explain the importance of low-latency LLMs in 100 words."
        
        print(f"User: {user_prompt}\n")
        print("Groq Llama 3.3 70b:")

        # Create the chat completion request with streaming enabled.
        # The model 'llama-3.3-70b-versatile' is specified here.
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )

        # Iterate over the streaming response chunks and print them.
        for chunk in completion:
            print(chunk.choices[0].delta.content or "", end="")
        
        # Print a newline at the end for clean formatting.
        print("\n")

    except APIError as e:
        # Handle potential API errors, such as an invalid API key
        # or network issues.
        print(f"An API error occurred: {e}")
        print("Please check if your GROQ_API_KEY is set correctly in your .env file.")
    except Exception as e:
        # Handle other potential exceptions.
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Ensure the script runs the main function when executed.
    main()
