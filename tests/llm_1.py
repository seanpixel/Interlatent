from vllm import LLM
from transformers import AutoTokenizer
from interlatent.api import LatentDB
from interlatent.llm import VLLMCollector

llm = LLM("TinyLlama/TinyLlama-1.1B-Chat-v1.0")  # single-process
tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

db = LatentDB("sqlite:///latents_llm.db")
collector = VLLMCollector(db, layer_indices=[-1], max_channels=512)
collector.run(llm, tok, prompts=["Hello world", "Why is the sky blue?"], max_new_tokens=16)