from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
)

prompt_1 = "The capital of India is"
prompt_2 = "The name of the capital city of India is"
prompt_3 = "What is the name of the capital city of India, she asked. Please only respond with the city name and then stop talking. He answered: "

response_1 = llm(prompt_1)
print(response_1 + "\n\n")

response_2 = llm(prompt_2)
print(response_2 + "\n\n")

response_3 = llm(prompt_3)
print(response_3 + "\n\n")

