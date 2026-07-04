# defense/defenses/input_filter_defense.py
from defenses.base_defended_model import BaseDefendedModel

class InputFilterDefense(BaseDefendedModel):
    """
    An input filter that can use either static forbidden words or
    a judger to detect harmful content in prompts.
    """

    def __init__(self, target_model, forbidden_words=None, replacement="[REDACTED]", 
                 judger_config=None, use_judger=True):
        super().__init__(target_model)
        self.forbidden_words = forbidden_words or ["bomb", "kill"]
        self.replacement = replacement
        
        # Load judger for harmful content detection
        self.judger = None
        self.use_judger = use_judger
        
        if self.use_judger:
            from judgers import load_judger_from_config, load_judger_from_name
            if judger_config:
                try:
                    self.judger = load_judger_from_config(judger_config)
                except Exception as e:
                    print(f"Failed to load judger from config: {e}, falling back to static filter")
                    self._record_fallback(f"judger_load_failed: {e}")
                    self.judger = None
            else:
                # Use rejection_prefix judger as default
                try:
                    self.judger = load_judger_from_name('rejection_prefix')
                except Exception as e:
                    print(f"Failed to load rejection_prefix judger: {e}, falling back to static filter")
                    self._record_fallback(f"judger_load_failed: {e}")
                    self.judger = None

    def defend_input(self, prompt: str) -> str:
        # First try judger-based detection if available
        if self.use_judger and self.judger:
            try:
                # rejection_prefix judger returns True for harmful, False for safe
                if self.judger.judge(generation=prompt):
                    print(f"[InputFilter] Harmful content detected by judger, blocking input")
                    return self.replacement
            except Exception as e:
                print(f"[InputFilter] Judger error: {e}, falling back to static filter")
                self._record_fallback(f"judger_error: {e}")
        
        # Fall back to or additionally use static forbidden words filter
        safe_prompt = prompt
        for w in self.forbidden_words:
            if w in safe_prompt:
                safe_prompt = safe_prompt.replace(w, self.replacement)
                
        if safe_prompt != prompt:
            print(f"[InputFilter] Forbidden words detected and replaced")
            
        print(safe_prompt)
        return safe_prompt

    # 'defend_output' is left as default from base (i.e. no changes)
