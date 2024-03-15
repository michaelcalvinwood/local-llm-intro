from ctransformers import AutoModelForCausalLM

def streamPrintLlm (prompt, llm):
    print(prompt)
    for word in llm(prompt, stream=True):
        print(word, end='', flush=True)
    print("\n")

def format_prompt (instruction):
    # Source for Llama 2 chat prompt template: https://gpus.llm-utils.org/llama-2-prompt-template/
    system = "You are an AI assistant that gives helpful answers. You answer the question in a short and concise way."
    prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST]"
    return prompt

# Model Card: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
llm = AutoModelForCausalLM.from_pretrained(
    "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
)

instruction = "What is the capital of India?"
prompt = format_prompt(instruction)
streamPrintLlm(prompt, llm)
