# defense/defenses/output_filter_defense.py
from defenses.base_defended_model import BaseDefendedModel

class OutputFilterDefense(BaseDefendedModel):
    """
    An output filter that can use either static disallowed phrases or
    a judger to detect harmful content in model outputs.
    """

    def __init__(self, target_model, disallowed_phrases=None, placeholder="[[BLOCKED]]",
                 judger_config=None, use_judger=True):
        super().__init__(target_model)
        self.disallowed_phrases = disallowed_phrases or ["how to build a bomb", "child pornography"]
        self.placeholder = placeholder
        
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

    def defend_output(self, output: str) -> str:
        # First try judger-based detection if available
        if self.use_judger and self.judger:
            try:
                # rejection_prefix judger returns True for harmful, False for safe
                if self.judger.judge(generation=output):
                    print(f"[OutputFilter] Harmful content detected by judger, blocking output")
                    return self.placeholder
            except Exception as e:
                print(f"[OutputFilter] Judger error: {e}, falling back to static filter")
                self._record_fallback(f"judger_error: {e}")
        
        # Fall back to or additionally use static disallowed phrases filter
        safe_output = output
        for phrase in self.disallowed_phrases:
            if phrase.lower() in safe_output.lower():
                print(f"[OutputFilter] Disallowed phrase detected, blocking output")
                safe_output = self.placeholder
                break  # or you can do partial replace
        return safe_output
