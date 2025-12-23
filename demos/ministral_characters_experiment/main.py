from characters import ch_1, ch_2, ch_3, ch_4
from prompts import prompt_1
from utils import generate


for ch in [ch_1, ch_2, ch_3, ch_4]:
    base_prompt = f"""rewrite the following prompt in the style of the character below without changing the meaning of the prompt at all (do not remove or add anything). At the end, the prompt should request a final ultimatum between the given options. Generate simply the prompt only, without any additional text or explanation.\ncharacter: {ch}\nprompt: {prompt_1}"""
    print(f"Character: {ch.splitlines()[0]}")
    generated_prompt = generate(prompt=base_prompt)
    print(f"Generated Prompt: {generated_prompt}\n")
    print("Generating response based on the rewritten prompt...\n")

    generated_response = generate(prompt=generated_prompt)
    print(f"Response: {generated_response}\n")
    print("="*50 + "\n")

