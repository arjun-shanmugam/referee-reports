import ollama

class ChatLLM:
    def __init__(self, model = 'llama3.3'):
        self.messages=[]
        self.model=model

    def add_history(self, content, role):
        self.messages.append({'role': role, 'content': content})

    def chat(self, message, json_file = None):
        self.add_history(message, 'user')
        if json_file:
            self.add_history(f'JSON File: {json_file}', 'user')
        response = ollama.chat(model=self.model, messages=self.messages, options={'num_ctx':131072})
        self.add_history(response['message']['content'], 'assistant')
        return response['message']['content']
        
    def run_prompt(self, prompt_file, data_json, output_file = None):
        with open(prompt_file, "rb") as doc_file:
            prompt = doc_file.read().decode()
            response = self.chat(prompt, data_json)
            if output_file:
                with open(output_file, "w") as output:
                    output.write(response)

def run_parametric_series_prompts(train_json, test_json):
    chat = ChatLLM()

    # prompt 0
    chat.run_prompt("prompts/prompt0.txt", train_json, output_file = None)

    # prompt 1
    chat.run_prompt("prompts/prompt1.txt", None, output_file = "../llm outputs/prompt1_scores.txt")

    # prompt 2
    chat.run_prompt("prompts/prompt2.txt", test_json, output_file = "../llm outputs/prompt2.txt")

    # prompt 3
    chat.run_prompt("prompts/prompt3.txt", None, output_file = "../llm outputs/prompt3.txt")

    print("finished parametric\n")

def run_nonparametric_series_prompts(train_json, test_json):
    chat = ChatLLM()

    # prompt 0
    chat.run_prompt("prompts/prompt_ref0.txt", train_json, output_file = None)

    # prompt 1
    chat.run_prompt("prompts/prompt_ref1.txt", test_json, output_file = "../llm outputs/notheme.txt")

    # prompt 2
    chat.run_prompt("prompts/prompt_ref2.txt", None, output_file = "../llm outputs/intros.txt")

    print("finished nonparametric\n")
