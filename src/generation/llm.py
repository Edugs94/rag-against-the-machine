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
        inputs: BatchEncoding = self.tokenizer(prompt, return_tensors="pt")

        outputs: torch.Tensor = self.model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]

        response: str = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        clean_response: str = response.split("```")[0]
        clean_response = clean_response.split("\n[")[0]
        clean_response = clean_response.split("\n\n")[0]

        clean_response = clean_response.split("\n**")[0]
        clean_response = clean_response.split("</")[0]
        clean_response = clean_response.strip()

        return clean_response
