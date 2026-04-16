# Connection with transformers and the Qwen3-0.6B model
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    PreTrainedModel,
    BatchEncoding,
)


class LLM:
    """Handles loading and generating answers with the Qwen model."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """Loads the tokenizer and the model."""
        print(f"Loading model {model_name}...")

        self.tokenizer: PreTrainedTokenizerBase = (
            AutoTokenizer.from_pretrained(model_name)
        )

        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name
        )

    def generate(self, prompt: str) -> str:
        """Generates a text response from a given prompt."""
        inputs: BatchEncoding = self.tokenizer(prompt, return_tensors="pt")
        outputs: torch.Tensor = self.model.generate(
            **inputs, max_new_tokens=500
        )
        response: str = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )
        return response
