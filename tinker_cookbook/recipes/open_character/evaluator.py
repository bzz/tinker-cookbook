"""
Constitution Classifier Evaluator for tinker-cookbook.

This evaluator samples completions from a model, sends them to the Constitution
Classifier API for classification, and computes F1/accuracy metrics based on
how well the responses adhere to the expected constitution.

Usage example:
    from tinker_cookbook.recipes.open_character.evaluator import (
        ConstitutionClassifierEvaluator
    )

    evaluator = ConstitutionClassifierEvaluator(
        classifier_api_url="http://localhost:8000",
        expected_constitution="mathematical",
        eval_prompts=[
            "Explain how compound interest works.",
            "What's the best way to organize my day?",
        ],
        max_samples=50,
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        renderer_name="llama3",
        temperature=0.7,
        max_tokens=512,
    )

    config = train.Config(
        # ... training config
        evaluators=[evaluator],
        eval_every_n_steps=100,
    )
"""

import logging
from typing import Any

import evaluate
import httpx
import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


class ConstitutionClassifierEvaluator(SamplingClientEvaluator):
    """
    Evaluator that uses an external Constitution Classifier API to measure
    how well model responses adhere to a target constitution.

    The evaluator:
    1. Samples completions from the model for given prompts
    2. Sends completions to the classifier API
    3. Compares predictions against expected constitution
    4. Computes F1 score and accuracy metrics

    Args:
        classifier_api_url: URL of the Constitution Classifier API (e.g., "http://localhost:8000")
        expected_constitution: The constitution label we expect responses to match
        eval_prompts: List of prompts to evaluate (or None for custom dataset)
        eval_dataset: Custom dataset with 'prompt' field (alternative to eval_prompts)
        max_samples: Maximum number of samples to evaluate (None = all)
        model_name: Model name for tokenizer
        renderer_name: Renderer name for formatting prompts
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        system_prompt: Optional system prompt to prepend to conversations
    """

    def __init__(
        self,
        classifier_api_url: str,
        expected_constitution: str,
        eval_prompts: list[str] | None = None,
        eval_dataset: list[dict[str, Any]] | None = None,
        max_samples: int | None = None,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        renderer_name: str = "llama3",
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 1.0,
        top_k: int = -1,
        system_prompt: str | None = None,
    ):
        self.classifier_api_url = classifier_api_url.rstrip("/")
        self.expected_constitution = expected_constitution
        self.max_samples = max_samples
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.system_prompt = system_prompt

        # Set up dataset
        if eval_prompts is not None and eval_dataset is not None:
            raise ValueError("Cannot specify both eval_prompts and eval_dataset")
        if eval_prompts is None and eval_dataset is None:
            raise ValueError("Must specify either eval_prompts or eval_dataset")

        if eval_prompts is not None:
            self.dataset = [{"prompt": prompt} for prompt in eval_prompts]
        else:
            self.dataset = eval_dataset

        # Apply max_samples limit
        if self.max_samples is not None:
            self.dataset = self.dataset[: self.max_samples]

        # Set up renderer
        tokenizer = get_tokenizer(model_name)
        self.renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)

        # Load evaluate metrics
        self.f1_metric = evaluate.load("f1")
        self.accuracy_metric = evaluate.load("accuracy")

        logger.info(
            f"Initialized ConstitutionClassifierEvaluator: "
            f"API={self.classifier_api_url}, "
            f"expected_constitution={self.expected_constitution}, "
            f"num_samples={len(self.dataset)}"
        )

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        """
        Run evaluation and return metrics.

        Args:
            sampling_client: The sampling client to evaluate

        Returns:
            Dictionary with 'constitution_f1' and 'constitution_accuracy' metrics
        """
        logger.info(f"Running constitution evaluation on {len(self.dataset)} samples...")

        # Prepare sampling params
        sampling_params = types.SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            stop=self.renderer.get_stop_sequences(),
        )

        # Sample completions for all prompts
        completions = []
        for i, example in enumerate(self.dataset):
            prompt = example["prompt"]

            # Build conversation
            messages = []
            if self.system_prompt:
                messages.append(renderers.Message(role="system", content=self.system_prompt))
            messages.append(renderers.Message(role="user", content=prompt))

            # Build model input
            model_input: types.ModelInput = self.renderer.build_generation_prompt(messages)

            # Generate response
            try:
                response: types.SampleResponse = await sampling_client.sample_async(
                    prompt=model_input, num_samples=1, sampling_params=sampling_params
                )
                tokens: list[int] = response.sequences[0].tokens
                parsed_message: renderers.Message = self.renderer.parse_response(tokens)[0]
                content = renderers.get_text_content(parsed_message)
                completions.append(content)
            except Exception as e:
                logger.warning(f"Failed to generate completion for sample {i}: {e}")
                completions.append("")  # Empty completion as fallback

        # Send completions to classifier API
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.classifier_api_url}/classify",
                    json={"texts": completions},
                )
                response.raise_for_status()
                result = response.json()
                predictions = [r["predicted_label"] for r in result["results"]]
        except Exception as e:
            logger.error(f"Failed to classify texts via API: {e}")
            raise RuntimeError(f"Classifier API request failed: {e}")

        # Create ground truth labels (all expected to be the target constitution)
        ground_truth = [self.expected_constitution] * len(predictions)

        # Compute metrics
        try:
            # F1 score (macro average across all constitution labels)
            f1_result = self.f1_metric.compute(
                predictions=predictions, references=ground_truth, average="macro"
            )
            f1_score = f1_result["f1"]

            # Accuracy
            accuracy_result = self.accuracy_metric.compute(
                predictions=predictions, references=ground_truth
            )
            accuracy = accuracy_result["accuracy"]

        except Exception as e:
            logger.error(f"Failed to compute metrics: {e}")
            f1_score = 0.0
            accuracy = 0.0

        logger.info(
            f"Constitution evaluation complete: F1={f1_score:.4f}, Accuracy={accuracy:.4f}"
        )

        return {
            "constitution_f1": f1_score,
            "constitution_accuracy": accuracy,
        }
