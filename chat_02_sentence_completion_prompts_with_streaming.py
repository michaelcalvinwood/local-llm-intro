from ctransformers import AutoModelForCausalLM

def streamPrintLlm (prompt, llm):
    for word in llm(prompt, stream=True):
        print(word, end='', flush=True)
    print("\n")

# Model Card: https://huggingface.co/zoltanctoth/orca_mini_3B-GGUF
llm = AutoModelForCausalLM.from_pretrained(
    "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
)

prompt = "State the city name and then stop talking. The capital of India is"
streamPrintLlm(prompt, llm)
