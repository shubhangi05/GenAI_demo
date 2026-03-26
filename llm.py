from google import genai
from google.genai import types

client = genai.Client(api_key="YOUR_API_KEY")


########Interactive Version
chat = client.chats.create(
    model="gemini-3-flash-preview",
    history=[],
    config=types.GenerateContentConfig(
        system_instruction="Whatever user's question or statement be your answer should always be very rude."
    )

)

def main():
    while True:
        user_problem = input("Ask me anything --> ")

        response = chat.send_message(
            message=user_problem
        )

        print(response.text)

# Run program
main()