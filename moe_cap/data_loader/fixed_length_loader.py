"""
Fixed-length data loader for benchmarking without accuracy evaluation.

This loader wraps any base dataset and pads/truncates inputs to a fixed token length.
Useful for pure performance benchmarking (TPOT, throughput) without accuracy metrics.
"""
from .base_data_loader import DataLoader
from typing import Any, Dict, List, Optional
from moe_cap.configs.cap_config import CAPConfig
from datasets import load_dataset
from transformers import AutoTokenizer


class FixedLengthLoader(DataLoader):
    """
    Loads any dataset and adjusts input to fixed token length.
    
    - If input is shorter than target_input_tokens: repeat/pad the text
    - If input is longer: truncate to target_input_tokens
    
    This mode is for performance benchmarking only (no accuracy evaluation).
    """

    def __init__(self, config: CAPConfig, base_dataset_name: str = "gsm8k") -> None:
        super().__init__(config)
        
        # Parse fixed length settings from config
        self.target_input_tokens = config.target_input_tokens or 4096
        self.target_output_tokens = config.target_output_tokens or 1024
        self.num_samples = config.num_samples or 100
        self.base_dataset_name = base_dataset_name
        
        # Load tokenizer for the model
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, 
            trust_remote_code=True
        )
        
        # Load base dataset
        self._load_base_dataset(base_dataset_name)
        
        # Process inputs to fixed length
        self._process_to_fixed_length()

    def _load_base_dataset(self, dataset_name: str) -> None:
        """Load the base dataset to use as source text.
        
        Dynamically loads any registered dataset and extracts text for benchmarking.
        """
        from .loader_registry import _REGISTRY
        
        key = dataset_name.lower()
        if key in _REGISTRY:
            # Use the registered loader to get data
            LoaderCls, _ = _REGISTRY[key]
            # Temporarily disable fixed_length_mode to load actual data
            original_mode = self.config.fixed_length_mode
            self.config.fixed_length_mode = False
            try:
                base_loader = LoaderCls(self.config)
                self.base_texts = base_loader.get_input()
            finally:
                self.config.fixed_length_mode = original_mode
        else:
            # Fallback: generate synthetic text
            print(f"Warning: Dataset '{dataset_name}' not found, using synthetic text")
            self.base_texts = self._generate_synthetic_texts()

    def _generate_synthetic_texts(self) -> List[str]:
        """Generate synthetic texts for benchmarking."""
        base_text = """The quick brown fox jumps over the lazy dog. This is a sample text for benchmarking purposes. 
        Machine learning models require extensive computational resources for training and inference. 
        Large language models have revolutionized natural language processing tasks. 
        Mixture-of-Experts architectures enable efficient scaling of model parameters. """
        return [base_text] * self.num_samples

    def _adjust_text_to_length(self, text: str, target_tokens: int) -> str:
        """Adjust text to exactly target_tokens by padding or truncating."""
        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        current_len = len(tokens)
        
        if current_len >= target_tokens:
            # Truncate
            tokens = tokens[:target_tokens]
        else:
            # Repeat text until we reach target length
            while len(tokens) < target_tokens:
                tokens = tokens + self.tokenizer.encode(text, add_special_tokens=False)
            tokens = tokens[:target_tokens]
        
        # Decode back to text
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _process_to_fixed_length(self) -> None:
        """Process all inputs to fixed length."""
        self.processed_inputs = []
        
        for i in range(self.num_samples):
            # Cycle through base texts if needed
            base_text = self.base_texts[i % len(self.base_texts)]
            
            # Adjust to target input length
            fixed_input = self._adjust_text_to_length(base_text, self.target_input_tokens)
            
            # Add a simple prompt suffix for generation
            prompt = f"{fixed_input}\n\nPlease continue writing:"
            self.processed_inputs.append(prompt)

    def get_input(self) -> List:
        """Get fixed-length input prompts."""
        return self.processed_inputs

    def get_target(self) -> List:
        """No targets for performance benchmarking - return empty list."""
        return [""] * len(self.processed_inputs)

    def get_eval_metrics(self) -> Optional[Dict[str, Any]]:
        """No accuracy metrics for fixed-length benchmarking."""
        return None  # Skip accuracy evaluation

    def get_max_new_tokens(self) -> int:
        """Return the target output token count."""
        return self.target_output_tokens
