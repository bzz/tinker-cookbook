"""
RLVR SamplingEvaluator for evaluating models trained with the RLVR pipeline.

This module provides a reusable evaluator that:
- Reuses TemplateRLDatasetBuilder for data loading
- Processes examples in batches with async inference
- Provides extensible per-example metric computation
- Automatically aggregates all numeric metrics
"""

from abc import abstractmethod
import asyncio
from typing import Any, TypedDict

import chz
import tinker
from tinker import types
from tqdm.asyncio import tqdm
from tinker_cookbook import renderers
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.recipes.rlvr.train import TemplateRLDataset, TemplateRLDatasetBuilder

# Maximum concurrent evaluation requests
DEFAULT_MAX_CONCURRENCY = 32


class EvaluationResult(TypedDict):
    """Result of evaluating a single example."""
    format: float
    correct: float
    reward: float
    num_lines: float  # Number of lines in the response (for verbosity tracking)
    custom_metric: Any  # Placeholder for custom metric (computed per example)


class RLVREvaluator(SamplingClientEvaluator):
    """
    Evaluator for RLVR models using TemplateRLDatasetBuilder for data formating.
    
    Features:
    - Lazy-loads dataset on first call and caches it
    - Processes examples in batches (matching dataset batch_size)
    - Uses env.compute_reward() for reward computation
    - Provides compute_metric() stub for per-example custom metrics
    - Automatically averages all numeric fields in aggregate_metrics()
    
    Example:
        >>> eval_ds_builder = TemplateRLDatasetBuilder(
        ...     env_class="tinker_cookbook.recipes.rlvr.patch_env:PatchExactMatchEnv",
        ...     dataset_name="my/dataset",
        ...     dataset_split="test[:100]",
        ...     user_template="Apply patch:\\n{patch}",
        ...     answer_field="expected_output",
        ...     model_name_for_tokenizer="Qwen/Qwen3-8B",
        ...     renderer_name="qwen3",
        ...     seed=42,
        ... )
        >>> evaluator = RLVREvaluator(
        ...     dataset_builder=eval_ds_builder,
        ...     max_tokens=512,
        ...     temperature=0.7,
        ... )
        >>> # Use in training: config = train.Config(..., evaluators=[evaluator])
    """
    
    def __init__(
        self,
        dataset_builder: TemplateRLDatasetBuilder,
        max_tokens: int = 512,
        temperature: float = 0.7,
        max_eval_samples: int | None = None,
    ):
        """
        Initialize the RLVR evaluator.
        
        Args:
            dataset_builder: Builder that creates TemplateRLDataset with eval data
            max_tokens: Maximum tokens to generate per sample
            temperature: Sampling temperature
            max_eval_samples: Maximum number of examples to evaluate (None = all)
        """
        self.dataset_builder = dataset_builder
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_eval_samples = max_eval_samples
        
        # Lazy-loaded and cached
        self._eval_dataset: TemplateRLDataset | None = None
        self._renderer: renderers.Renderer | None = None
    
    @abstractmethod
    def compute_metric(
        self,
        env,
        response_content: str,
        correct_format: bool,
        correct_answer: bool,
    ) -> Any:
        """
        Compute a custom metric for a single example.
        
        This is a stub to be implemented in subclasses for custom per-example metrics.
        
        Args:
            env: The environment instance for this example
            response_content: The model's response text
            correct_format: Whether the format is correct
            correct_answer: Whether the answer is correct
            
        Returns:
            Custom metric value (any numeric type, or None)
        """
        # Stub: to be implemented in subclasses
        ...
    
    async def _evaluate_example(
        self,
        builder,
        sampling_client: tinker.SamplingClient,
    ) -> EvaluationResult:
        """
        Evaluate a single example by sampling once from the model.
        
        Args:
            builder: EnvGroupBuilder for creating the environment
            sampling_client: Client for sampling from the model
            
        Returns:
            Evaluation result for this example
        """
        env = builder.env_thunk()
        
        # Build prompt
        messages = []
        if env.convo_prefix:
            messages.extend(env.convo_prefix)
        messages.append({"role": "user", "content": env.question})
        model_input = self._renderer.build_generation_prompt(messages)
        
        # Sample from model
        sampling_params = types.SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=self._renderer.get_stop_sequences(),
        )
        response = await sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )
        
        # Parse response
        tokens = response.sequences[0].tokens
        message = self._renderer.parse_response(tokens)[0]
        content = renderers.get_text_content(message)
        
        # Log env's output
        # try:
        correct_format = float(env.check_format(content))
        correct_answer = float(env.check_answer(content))
        # except Exception as e: # FIXME: remove
        #     print(f"Error processing answer: {e}")
        #     print("-" * 100)
        #     print(f"Content: {content}")
        #     print("-" * 100)
        #     print(f"Answer:\n{env.row['udiff']}\n")
        #     print("-" * 100)
        #     return EvaluationResult(
        #         format=0.0,
        #         correct=0.0,
        #         reward=env.compute_reward(0, 0),
        #         custom_metric=0.0,
        #     )
        reward = env.compute_reward(correct_format, correct_answer)
        
        # Compute metric
        custom_metric = self.compute_metric(env, content, correct_format, correct_answer)
        
        # Count lines in response for verbosity tracking
        num_lines = float(content.count('\n') + 1) if content else 0.0
        
        return EvaluationResult(
            format=correct_format,
            correct=correct_answer,
            reward=reward,
            num_lines=num_lines,
            custom_metric=custom_metric,
        )
    
    def aggregate_metrics(self, results: list[EvaluationResult]) -> dict[str, float]:
        """
        Aggregate results into final metrics by averaging all numeric fields.
        
        Automatically computes averages for:
        - format, correct, reward (always present)
        - custom_metric (if numeric and not None)
        
        Override this method only if you need custom aggregation logic beyond averaging.
        
        Args:
            results: List of evaluation results from all examples
            
        Returns:
            Dictionary of metric name -> value (will be prefixed with test/)
        """
        if not results:
            return {}
        
        metrics = {
            "test/format_valid": sum(r["format"] for r in results) / len(results),
            "test/correct": sum(r["correct"] for r in results) / len(results),
            "test/reward": sum(r["reward"] for r in results) / len(results),
            "test/num_lines": sum(r["num_lines"] for r in results) / len(results),
            "test/num_examples": len(results),
        }
        
        # Automatically aggregate custom_metric if it's numeric
        custom_values = [ # FIXME
            r["custom_metric"]
            for r in results
            if r["custom_metric"] is not None and isinstance(r["custom_metric"], (int, float))
        ]
        if custom_values:
            metrics["test/custom_metric"] = sum(custom_values) / len(custom_values)
        
        return metrics
    
    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        """
        Evaluate the model on the dataset.
        
        Args:
            sampling_client: Sampling client to evaluate
            
        Returns:
            Dictionary of metrics with test/ prefix
        """
        # Lazy load dataset on first call
        if self._eval_dataset is None:
            self._eval_dataset, _ = await self.dataset_builder()
            self._renderer = self._eval_dataset.renderer
        
        # Environment builders for all examples
        all_builders = []
        for batch_idx in range(len(self._eval_dataset)):
            builders = self._eval_dataset.get_batch(batch_idx)
            all_builders.extend(builders)
            
            # Stop if we've collected enough for max_eval_samples
            if self.max_eval_samples is not None and len(all_builders) >= self.max_eval_samples:
                all_builders = all_builders[:self.max_eval_samples]
                break
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(DEFAULT_MAX_CONCURRENCY)
        
        async def evaluate_in_batch(builder):
            async with semaphore:
                return await self._evaluate_example(builder, sampling_client)
        
        # Evaluate all examples with controlled concurrency and progress bar
        tasks = [evaluate_in_batch(builder) for builder in all_builders]
        all_results = await tqdm.gather(*tasks, desc="Evaluating", unit="example")
        
        # Aggregate and return metrics
        return self.aggregate_metrics(all_results)


@chz.chz
class RLVREvaluatorBuilder:
    """
    Builder for RLVREvaluator.
    
    Constructs an evaluator by creating a TemplateRLDatasetBuilder for the eval dataset
    and wrapping it in an RLVREvaluator.
    
    Example:
        >>> eval_ds_builder = RLVREvaluatorBuilder(
        ...     env_class="tinker_cookbook.recipes.rlvr.patch_env:PatchExactMatchEnv",
        ...     dataset_name="bzz2/diff-xyz-v4a",
        ...     dataset_config="easy",
        ...     dataset_split="test",
        ...     user_template="Apply patch:\\n{patch}",
        ...     answer_field="expected_output",
        ...     model_name_for_tokenizer="Qwen/Qwen3-8B",
        ...     renderer_name="qwen3",
        ... )
        >>> evaluator = eval_ds_builder()
    """
    
    # Dataset configuration (same as TemplateRLDatasetBuilder)
    env_class: str
    dataset_name: str
    dataset_config: str | None = None
    dataset_split: str = "test"
    user_template: str
    answer_field: str
    model_name_for_tokenizer: str
    renderer_name: str | None = None
    system_prompt: str | None = None
    seed: int | None = None  # None = no shuffle for deterministic eval
    
    # Evaluation-specific parameters
    max_eval_samples: int | None = None  # Limit number of examples (None = all)
    max_tokens: int = 512
    temperature: float = 0.0
    
    def __call__(self) -> RLVREvaluator:
        """Build the RLVREvaluator."""
        # Create a dataset builder for evaluation
        eval_ds = TemplateRLDatasetBuilder(
            env_class=self.env_class,
            dataset_name=self.dataset_name,
            dataset_config=self.dataset_config,
            dataset_split=self.dataset_split,
            user_template=self.user_template,
            answer_field=self.answer_field,
            model_name_for_tokenizer=self.model_name_for_tokenizer,
            renderer_name=self.renderer_name,
            batch_size=1,  # Batch size doesn't matter for eval, we use semaphore for concurrency
            group_size=1,  # Eval doesn't need multiple rollouts per example
            system_prompt=self.system_prompt,
            seed=self.seed,
        )
        
        return RLVREvaluator(
            dataset_builder=eval_ds,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            max_eval_samples=self.max_eval_samples,
        )

