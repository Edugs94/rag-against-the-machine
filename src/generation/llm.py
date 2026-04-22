'''Connection with transformers and the Qwen3-0.6B model'''
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    PreTrainedModel,
    TextStreamer,
)
from transformers.generation.utils import GenerationMixin
from typing import cast
from src.constants import DEFAULT_LLM_MODEL


class LLM:
    """Handles loading and generating answers with the Qwen model."""

    def __init__(self, model_name: str = DEFAULT_LLM_MODEL) -> None:
        """Loads the tokenizer and the model."""
        print(f"Loading model {model_name}...")

        try:
            self.tokenizer: PreTrainedTokenizerBase = (
                AutoTokenizer.from_pretrained(model_name)
            )
            self.raw_model: PreTrainedModel = (
                AutoModelForCausalLM.from_pretrained(model_name)
            )
        except OSError as e:
            raise RuntimeError(
                f"Could not load model '{model_name}'. "
                f"Check the model name and your network connection. "
                f"Original error: {e}"
            ) from e
        self.model = cast(GenerationMixin, self.raw_model)

        self.stop_token_ids = self._resolve_stop_token_ids()

    def _resolve_stop_token_ids(self) -> list[int]:
        """Returns all token IDs that should halt generation."""
        ids: set[int] = set()
        if self.tokenizer.eos_token_id is not None:
            ids.add(self.tokenizer.eos_token_id)
        for tok in ("<|im_end|>", "<|endoftext|>"):
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid != self.tokenizer.unk_token_id:
                ids.add(tid)
        return list(ids)

    def generate(self, messages: list[dict]) -> str:
        """Generates a text response using Qwen's chat template."""
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")

        outputs = self.model.generate(  # type: ignore[misc]
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.0,
            eos_token_id=self.stop_token_ids,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = cast(str, self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ))

        return response.strip()

    def generate_streaming(self, messages: list[dict]) -> None:
        """Generate a response and stream it to stdout token by token.

        Used for interactive demos. Does not return the text.
        """
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")

        streamer = TextStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        self.model.generate(  # type: ignore[misc]
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.0,
            eos_token_id=self.stop_token_ids,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )
