import random
import json
from collections import defaultdict
from transformers import pipeline, pipeline_init

class Heisenberg:
    def __init__(self):
        self.responses = self.load_responses_from_file()
        self.response_counts = defaultdict(int)
        self.load_additional_responses()

        # Initialize the NLP model for response generation
        self.generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B', device=0)
        pipeline_init(self.generator)

    def add_custom_response(self, custom_response):
        if custom_response.strip():
            self.responses.append(custom_response)
            self.response_counts[custom_response] = 0
            self.save_responses_to_file()
            print("Custom response added successfully!")
        else:
            print("Custom response cannot be empty.")

    def remove_custom_response(self, custom_response):
        if custom_response in self.responses:
            self.responses.remove(custom_response)
            self.save_responses_to_file()
            print("Custom response removed successfully!")
        else:
            print("Custom response not found.")

    def reset_response_counts(self):
        self.response_counts.clear()
        self.save_responses_to_file()
        print("Response counts reset successfully!")

    def view_responses(self):
        print("All Responses:")
        for idx, response in enumerate(self.responses, 1):
            print(f"{idx}. {response} - {self.response_counts[response]} times")

    def get_random_response(self, question):
        response = random.choice(self.responses)
        self.response_counts[response] += 1
        # Use NLP model to generate a contextual response
        context = f"Question: {question}\nResponse: {response}\n"
        nlp_response = self.generator(context, max_length=50, do_sample=True)[0]['generated_text']
        return nlp_response.split("Response: ")[1].strip()

    def save_responses_to_file(self):
        with open('responses.json', 'w') as file:
            data = {
                'responses': self.responses,
                'response_counts': self.response_counts
            }
            json.dump(data, file)

    def load_responses_from_file(self):
        try:
            with open('responses.json', 'r') as file:
                data = json.load(file)
                self.response_counts = defaultdict(int, data.get('response_counts', {}))
                return data.get('responses', [])
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def load_additional_responses(self):
        try:
            with open('extra_responses.json', 'r') as file:
                data = json.load(file)
                extra_responses = data.get('extra_responses', [])
                self.responses.extend(extra_responses)
        except (FileNotFoundError, json.JSONDecodeError):
            print("Could not load additional responses. Check the file 'extra_responses.json'.")

# Create an instance of Heisenberg
heisenberg = Heisenberg()

# Interactive session
while True:
    print("\nWhat would you like to do?")
    print("1. Ask a question")
    print("2. View all responses")
    print("3. Add a custom response")
    print("4. Remove a custom response")
    print("5. Reset response counts")
    print("6. Exit")

    choice = input("Enter your choice (1/2/3/4/5/6): ")

    if choice == "1":
        question = input("Ask Heisenberg a question: ")
        print("Heisenberg says:", heisenberg.get_random_response(question))
    elif choice == "2":
        heisenberg.view_responses()
    elif choice == "3":
        custom_response = input("Enter your custom response: ")
        heisenberg.add_custom_response(custom_response)
    elif choice == "4":
        custom_response = input("Enter the custom response to remove: ")
        heisenberg.remove_custom_response(custom_response)
    elif choice == "5":
        heisenberg.reset_response_counts()
    elif choice == "6":
        print("Goodbye! Thanks for consulting Heisenberg.")
        break
    else:
        print("Invalid choice. Please enter a valid option (1/2/3/4/5/6).")
