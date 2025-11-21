import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from interlatent.api import LatentDB
from interlatent.llm import VLLMCollector

MODEL_ID = os.environ.get("LLM_MODEL", "HuggingFaceTB/SmolLM-360M")

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None and getattr(tok, "eos_token", None):
    tok.pad_token = tok.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"
llm = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)

db = LatentDB("sqlite:///latents_llm.db")
collector = VLLMCollector(db, layer_indices=[-1], max_channels=512)
collector.run(llm, tok, prompts=["Hello world", "Why is the sky blue?"], max_new_tokens=16)
