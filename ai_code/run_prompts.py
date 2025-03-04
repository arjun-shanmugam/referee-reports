import ollama

def get_model_response(prompt, filename):
    # models = ['llama3.1', 'llama3.2', 'gemma2']
    model = 'llama3.2:1b'
    with open(filename, 'r') as file:
        content = file.read()
        response = ollama.chat(model=model, messages=[
            {
                'role': 'user',
                'content': f'{prompt} {content}',
            },
        ])

        return response['message']['content']
    
def run_prompt(prompt_file, data_file, output_file = None):
    with open(prompt_file, "rb") as doc_file:
        prompt = doc_file.read().decode()
        response = get_model_response(prompt, data_file)
        if output_file:
            with open(output_file, "w") as output:
                output.write(response)

def run_parametric_series_prompts(train_csv, test_csv):

    # prompt 0
    run_prompt("prompts/prompt0.txt", train_csv, output_file = None)

    # prompt 1
    run_prompt("prompts/prompt1.txt", train_csv, output_file = "llm outputs/prompt1_scores.txt")

    # prompt 2
    run_prompt("prompts/prompt2.txt", test_csv, output_file = "llm outputs/prompt2.txt")

    # prompt 3
    run_prompt("prompts/prompt3.txt", test_csv, output_file = "llm outputs/prompt3.txt")

def run_nonparametric_series_prompts(train_csv, test_csv):
    # prompt 0
    run_prompt("prompts/prompt_ref0.txt", train_csv, output_file = None)

    # prompt 1
    run_prompt("prompts/prompt_ref1.txt", test_csv, output_file = "llm outputs/notheme.txt")
