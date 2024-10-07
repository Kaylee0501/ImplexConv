import anthropic
import json

class ClaudeChatbot:
    def __init__(self, api_key):
        self.client = anthropic.Client(api_key)
        self.conversation_history = []

    def add_to_history(self, role, content):
        self.conversation_history.append({"role": role, "content": content})

    def get_response(self, user_input):
        # Add user input to history
        self.add_to_history("human", user_input)

        # Prepare the messages for the API call
        messages = [
            {"role": "system", "content": "You are Claude, an AI assistant."},
            *self.conversation_history
        ]

        # Make the API call
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=messages,
            max_tokens=1000
        )

        # Add Claude's response to history
        self.add_to_history("assistant", response.content[0].text)

        return response.content[0].text

    def save_history(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.conversation_history, f)

    def load_history(self, filename):
        with open(filename, 'r') as f:
            self.conversation_history = json.load(f)

# Usage example
if __name__ == "__main__":
    chatbot = ClaudeChatbot("your-api-key-here")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = chatbot.get_response(user_input)
        print("Claude:", response)

    # Save conversation history
    chatbot.save_history("conversation_history.json")