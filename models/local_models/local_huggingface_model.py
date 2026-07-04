from .local_base import LocalModel
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

# Optional dependency: only import when the user really wants 4‑bit quantisation.
try:
    from transformers import BitsAndBytesConfig  # transformers>=4.36
except ImportError:  # pragma: no cover
    BitsAndBytesConfig = None  # type: ignore

import torch
from ..api_keys import HUGGINGFACE_API_KEY
import re
import textwrap

__all__ = ["LocalHuggingFaceModel"]


class LocalHuggingFaceModel(LocalModel):
    """Wraps HuggingFace models (FP16/FP32 *or* 4‑bit BnB) behind a common interface.

    Parameters
    ----------
    model_name_or_path : str
        Identifier understood by 🤗   `from_pretrained` (path or Hub id).
    device : str, default "cuda"
        Target device when *not* using 4‑bit quantisation. Ignored when
        ``load_in_4bit=True`` (GPU is required in that case).
    torch_dtype : torch.dtype | None, default torch.float16
        Precision for full‑precision models. Set to ``None`` when 4‑bit.
    device_map : str | dict | None, default "auto"
        Passed to ``from_pretrained``. For large models, keep ``"auto"`` so that
        HF takes care of sharding.
    model_type : {"auto", "causal", "seq2seq"}
        Select the specialised class when needed.
    trust_remote_code : bool, default True
        Allow custom HF models.
    load_in_4bit : bool, default False
        Activate bitsandbytes 4‑bit quantisation.
    bnb_4bit_quant_type : {"nf4", "fp4"}, default "nf4"
        Quantisation scheme when ``load_in_4bit`` is True.
    bnb_4bit_compute_dtype : torch.dtype, default torch.float16
        Internal compute type for 4‑bit GEMMs.
    compile_model : bool, default False
        If True and running on PyTorch ≥2.0, re‑compile the forward pass
        for extra speed.
    **kwargs
        Forwarded verbatim to ``from_pretrained`` (e.g. ``revision=...``).
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = "cuda",
        torch_dtype: torch.dtype | None = torch.float16,
        device_map: str | dict | None = "auto",
        model_type: str = "auto",
        trust_remote_code: bool = True,
        load_in_4bit: bool = True,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_compute_dtype: torch.dtype = torch.float16,
        compile_model: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(model_name_or_path)

        self.device = device
        self.model_name_or_path = model_name_or_path
        self.load_in_4bit = load_in_4bit

        # ------------- configuration & tokenizer -------------------------
        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            token=HUGGINGFACE_API_KEY,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            token=HUGGINGFACE_API_KEY,
        )
        
        # Fix pad_token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ------------- choose the HF model class -------------------------
        if model_type == "causal":
            model_cls = AutoModelForCausalLM
        elif model_type == "seq2seq":
            model_cls = AutoModelForSeq2SeqLM
        else:
            model_cls = AutoModel

        # ------------- optional 4‑bit quantisation -----------------------
        quant_config = None
        if load_in_4bit:
            if BitsAndBytesConfig is None:
                raise ImportError(
                    "'load_in_4bit=True' requires transformers>=4.36 and bitsandbytes."
                )
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            )
            # 4‑bit models are GPU‑only; let HF decide placement.
            device = "cuda"
            device_map = "auto"
            torch_dtype = None  # Handled internally by BnB

        # ------------- load the weights ----------------------------------
        self.model = model_cls.from_pretrained(
            model_name_or_path,
            config=self.config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            quantization_config=quant_config,
            token=HUGGINGFACE_API_KEY,
            **kwargs,
        )

        # full‑precision path: explicit .to(device)
        if not load_in_4bit and device_map is None:
            self.model.to(device)

        # optional PT2 compile – can speed up 15‑30 % with mature kernels
        if compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
            except Exception:  # pragma: no cover
                pass  # graceful degradation on unsupported configs

    # ------------- public accessors ------------------------------------
    def get_tokenizer(self):
        """Return the underlying Hugging Face tokenizer (required by some attacks)."""
        return self.tokenizer

    # ---------------------------------------------------------------------
    # public helpers -------------------------------------------------------
    # ---------------------------------------------------------------------
    @property
    def model_name(self):
        return self.model_name_or_path

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def parameters(self):
        """Return the model's parameters (required by some defenses like GradSafe)."""
        return self.model.parameters()

    def supports_chat_template(self) -> bool:
        """Return whether the tokenizer exposes a usable chat template."""
        return (
            hasattr(self.tokenizer, "apply_chat_template")
            and bool(getattr(self.tokenizer, "chat_template", None))
        )

    def format_chat_prompt(
        self,
        user_prompt: str,
        assistant_prompt: str | None = None,
        add_generation_prompt: bool | None = None,
    ) -> str:
        """Format a user/assistant exchange with the target tokenizer's chat template."""
        if not self.supports_chat_template():
            if assistant_prompt is None:
                return user_prompt
            return f"{user_prompt}{assistant_prompt}"

        messages = [{"role": "user", "content": user_prompt}]
        if assistant_prompt is not None:
            messages.append({"role": "assistant", "content": assistant_prompt})
        if add_generation_prompt is None:
            add_generation_prompt = assistant_prompt is None

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def _looks_like_preformatted_chat(self, prompt: str) -> bool:
        """Heuristic guard against wrapping an already serialized chat prompt."""
        stripped = prompt.lstrip()
        markers = (
            "<s>[INST]",
            "[INST]",
            "<<SYS>>",
            "<|im_start|>",
            "<|start_header_id|>",
            "<start_of_turn>",
            "### Human:",
            "### User:",
            "<|user|>",
        )
        return any(stripped.startswith(marker) for marker in markers)

    # ---------------------- generation -----------------------------------
    def _needs_cache_fix(self) -> bool:
        """检查模型是否需要缓存修复以避免 DynamicCache get_max_length 错误"""
        model_name_lower = self.model_name_or_path.lower()
        affected_models = [
            "phi-3", "phi-4", "phi3", "phi4",  # Phi 系列
            "mistral", "qwen", "deepseek",     # 其他可能受影响的模型
        ]
        return any(model in model_name_lower for model in affected_models)

    def generate(self, prompt: str | list[str] | None = None, messages=None, **kwargs):
        generation_kwargs = kwargs.copy()
        use_chat_template = generation_kwargs.pop("use_chat_template", None)
        pad_token_id = generation_kwargs.pop("pad_token_id", self.tokenizer.eos_token_id)
        # Allow callers to pass raw input_ids positionally.
        if messages is None and isinstance(prompt, torch.Tensor):
            input_ids = prompt
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            model_inputs = {"input_ids": input_ids.to(self.model.device)}
            if "attention_mask" in generation_kwargs:
                attn = generation_kwargs.pop("attention_mask")
                model_inputs["attention_mask"] = attn.to(self.model.device) if hasattr(attn, "to") else attn
            with torch.inference_mode():
                return self.model.generate(
                    **model_inputs,
                    **generation_kwargs,
                    pad_token_id=pad_token_id,
                )
        # Allow callers to pass raw HF-style inputs as a dict.
        if messages is None and isinstance(prompt, dict) and "input_ids" in prompt:
            model_inputs = {}
            for k, v in prompt.items():
                model_inputs[k] = v.to(self.model.device) if hasattr(v, "to") else v
            with torch.inference_mode():
                return self.model.generate(
                    **model_inputs,
                    **generation_kwargs,
                    pad_token_id=pad_token_id,
                )
        # ─────────────────────────────────────────────
        # 0. Handle raw HF-style inputs (input_ids/attention_mask)
        # ─────────────────────────────────────────────
        if prompt is None and messages is None and "input_ids" in generation_kwargs:
            model_inputs = {}
            for k in list(generation_kwargs.keys()):
                if k in {"input_ids", "attention_mask", "position_ids", "token_type_ids"}:
                    v = generation_kwargs.pop(k)
                    model_inputs[k] = v.to(self.model.device) if hasattr(v, "to") else v
            with torch.inference_mode():
                return self.model.generate(
                    **model_inputs,
                    **generation_kwargs,
                    pad_token_id=pad_token_id,
                )

        # ─────────────────────────────────────────────
        # 1. Handle chat-style input (unchanged)
        # ─────────────────────────────────────────────
        if messages is not None and hasattr(self.tokenizer, "apply_chat_template"):
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = self.tokenizer([input_text], return_tensors="pt", padding=False)

        # ─────────────────────────────────────────────
        # 2. Handle plain prompt
        # ─────────────────────────────────────────────
        else:
            if prompt is None:
                raise ValueError("Either prompt or messages must be provided.")

            # ── NEW: accept list[str] silently ───────────────────────────────────
            if isinstance(prompt, list):
                # If list looks like token IDs, treat as raw input_ids.
                if prompt and all(isinstance(x, int) for x in prompt):
                    input_ids = torch.tensor(prompt, device=self.model.device).unsqueeze(0)
                    with torch.inference_mode():
                        return self.model.generate(
                            input_ids=input_ids,
                            **generation_kwargs,
                            pad_token_id=pad_token_id,
                        )
                prompt = "\n".join(prompt)  # collapses list → single string
            # ─────────────────────────────────────────────────────────────────────

            if (
                use_chat_template is None
                and isinstance(prompt, str)
                and self._looks_like_preformatted_chat(prompt)
            ):
                use_chat_template = False

            if use_chat_template is False or not self.supports_chat_template():
                input_text = prompt
            else:
                input_text = self.format_chat_prompt(prompt, add_generation_prompt=True)
            model_inputs = self.tokenizer([input_text], return_tensors="pt", padding=False)

        # ─────────────────────────────────────────────
        # 3. Generate
        # ─────────────────────────────────────────────
        input_length = model_inputs["input_ids"].shape[-1]
        model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}

        with torch.inference_mode():  # same as no_grad but clearer semantic
            # Handle DynamicCache compatibility issue for affected models
            if self._needs_cache_fix():
                # For models with known DynamicCache get_max_length issues, disable cache
                print(f"🔧 Disabling cache for {self.model_name_or_path} due to DynamicCache compatibility")
                generation_kwargs_with_cache = generation_kwargs.copy()
                generation_kwargs_with_cache["use_cache"] = False
                generated = self.model.generate(
                    **model_inputs,
                    **generation_kwargs_with_cache,
                    pad_token_id=pad_token_id,
                )
            else:
                # Try normal generation first, fallback to cache-disabled if needed
                try:
                    generated = self.model.generate(
                        **model_inputs,
                        **generation_kwargs,
                        pad_token_id=pad_token_id,
                    )
                except AttributeError as e:
                    if "get_max_length" in str(e):
                        # DynamicCache compatibility issue detected, retry without cache
                        print(f"⚠️  DynamicCache compatibility issue detected for {self.model_name_or_path}")
                        print(f"🔄 Retrying with cache disabled...")
                        generation_kwargs_with_cache = generation_kwargs.copy()
                        generation_kwargs_with_cache["use_cache"] = False
                        generated = self.model.generate(
                            **model_inputs,
                            **generation_kwargs_with_cache,
                            pad_token_id=pad_token_id,
                        )
                        print(f"✅ Successfully generated with cache disabled")
                    else:
                        print(f"❌ Unexpected AttributeError in model generation: {e}")
                        raise

        # ─────────────────────────────────────────────
        # 4. Post-process
        # ─────────────────────────────────────────────
        new_tokens = generated[0][input_length:]
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Debug logging for empty response issues
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"🔍 Model generation debug - {self.model_name_or_path}")
        logger.debug(f"   Input length: {input_length}")
        logger.debug(f"   New tokens count: {len(new_tokens)}")
        logger.debug(f"   Raw output length: {len(raw_output)}")
        logger.debug(f"   Raw output preview: '{raw_output[:100]}...'")

        # Robustly strip the original prompt (normalised whitespace)
        pattern = r"^" + re.sub(r"\s+", r"\\s+", re.escape(input_text.strip()))
        cleaned_output = re.sub(pattern, "", raw_output, flags=re.I).lstrip()
        
        logger.debug(f"   Cleaned output length: {len(cleaned_output)}")
        logger.debug(f"   Cleaned output preview: '{cleaned_output[:100]}...'")
        
        # If cleaned output is empty but raw output exists, return raw output instead
        if not cleaned_output.strip() and raw_output.strip():
            logger.warning(f"⚠️  Prompt cleaning resulted in empty output for {self.model_name_or_path}")
            logger.warning(f"   Returning raw output instead: '{raw_output[:100]}...'")
            return raw_output.strip()
        
        return cleaned_output

    # ------------------- logits for scoring ------------------------------
    def generate_from_input_id_to_logits(self, input_ids, attention_mask=None, labels=None, **kwargs):
        inputs = {"input_ids": input_ids}
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask

        if labels is not None:
            inputs["labels"] = labels
            with torch.inference_mode():
                outputs = self.model(**inputs, **kwargs)
        else:
            gen_kwargs = {"return_dict_in_generate": True, "output_logits": True}
            gen_kwargs.update(kwargs)
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            if isinstance(outputs.logits, (list, tuple)):
                outputs.logits = torch.stack(outputs.logits, dim=1)
        return outputs

    # ------------------------------------------------------------------
    # gradient‑based helpers – *not* available when quantised ----------
    # ------------------------------------------------------------------
    def compute_loss(self, input_text: str, target_text: str, **kwargs):
        if self.load_in_4bit:
            raise RuntimeError("Loss/gradient computation is not supported for 4‑bit models.")
        self.model.train()
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True).to(self.model.device)
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(target_text, return_tensors="pt", padding=True).input_ids.to(self.model.device)
        inputs["labels"] = targets
        outputs = self.model(**inputs, **kwargs)
        return outputs.loss

    def compute_gradients(self, input_text: str, target_text: str, **kwargs):
        if self.load_in_4bit:
            raise RuntimeError("Gradient computation is incompatible with 4‑bit quantisation. Use full‑precision weights instead.")
        self.model.train()
        self.model.zero_grad()
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True).to(self.model.device)
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(target_text, return_tensors="pt", padding=True).input_ids.to(self.model.device)
        inputs["labels"] = targets
        input_ids = inputs.pop("input_ids")
        embeddings = self.model.get_input_embeddings()(input_ids)
        embeddings.requires_grad_(True)
        inputs["inputs_embeds"] = embeddings
        outputs = self.model(**inputs, **kwargs)
        loss = outputs.loss
        loss.backward()
        return loss.detach(), embeddings.grad

    def get_embeddings(self, input_text: str):
        """
        Get embeddings for the input text.
        
        Args:
            input_text (str): Input text to get embeddings for
            
        Returns:
            torch.Tensor: Embeddings tensor
        """
        # Tokenize the input text
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True).to(self.model.device)
        
        # Get embeddings from the model
        with torch.inference_mode():
            # For causal LM models, we can get embeddings through the model's forward pass
            outputs = self.model(**inputs, output_hidden_states=True)
            # Get the last hidden state (embeddings)
            embeddings = outputs.hidden_states[-1]
            # Take the mean of the sequence dimension to get sentence-level embeddings
            embeddings = embeddings.mean(dim=1)
        
        return embeddings

    def get_prediction_scores(self, input_text: str):
        """
        Get prediction scores (logits) for the input text.
        
        Args:
            input_text (str): Input text to get prediction scores for
            
        Returns:
            torch.Tensor: Logits tensor
        """
        # Tokenize the input text
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True).to(self.model.device)
        
        # Get logits from the model
        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        return logits
