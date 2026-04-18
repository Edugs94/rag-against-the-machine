# Connection with transformers and the Qwen3-0.6B model
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    PreTrainedModel,
    BatchEncoding,
)
from transformers.generation.utils import GenerationMixin
from typing import cast
from src.constants import DEFAULT_LLM_MODEL
from transformers import TextStreamer


class LLM:
    """Handles loading and generating answers with the Qwen model."""

    def __init__(self, model_name: str = DEFAULT_LLM_MODEL) -> None:
        """Loads the tokenizer and the model."""
        print(f"Loading model {model_name}...")

        self.tokenizer: PreTrainedTokenizerBase = (
            AutoTokenizer.from_pretrained(model_name)
        )

        self.raw_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name
        )
        self.model = cast(GenerationMixin, self.raw_model)

    def generate(self, prompt: str) -> str:
        """Generates a text response using the Qwen model."""
        inputs = self.tokenizer(prompt, return_tensors="pt")

        streamer = TextStreamer(self.tokenizer, skip_prompt=True)
        print("\nQwen is thinking...")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.0,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )

        # Extract generated tokens and decode them into text
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        # Clean-up logic
        clean_response = response.strip()

        if "\nQuestion:" in clean_response:
            clean_response = clean_response.split("\nQuestion:")[0]
        if "\nContext:" in clean_response:
            clean_response = clean_response.split("\nContext:")[0]
        if "\nAnswer:" in clean_response:
            clean_response = clean_response.split("\nAnswer:")[0]

        # Stuttering loop prevention for bash blocks
        if clean_response.count("```bash") > 1:
            parts = clean_response.split("```bash")
            previous_text = parts[0]
            code_inside = parts[1].split("```")[0]
            clean_response = (
                previous_text + "```bash\n" + code_inside.strip() + "\n```"
            )

        clean_response = clean_response.replace("<|im_end|>", "")
        clean_response = clean_response.replace("<|endoftext|>", "")

        return clean_response.strip()