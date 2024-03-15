from ctransformers import AutoModelForCausalLM

def streamPrintLlm (prompt, llm):
    print(prompt)
    for word in llm(prompt, stream=True):
        print(word, end='', flush=True)
    print("\n")

def format_prompt (instruction):
    system = "You are an AI assistant that gives helpful answers. You answer the questions in a short and concise way."
    prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
    return prompt

# Model Card: https://huggingface.co/zoltanctoth/orca_mini_3B-GGUF
llm = AutoModelForCausalLM.from_pretrained(
    "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
)

instruction = "What is the capital of India?"
prompt = format_prompt(instruction)
streamPrintLlm(prompt, llm)
