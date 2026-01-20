#!/usr/bin/env python3
"""
Offline evaluation of constitution adherence using Tinker API.

This script evaluates a model checkpoint's adherence to a target constitution
by sampling responses on a dataset and classifying them using the Constitution
Classifier API. Results include F1 score and accuracy with bootstrapped confidence
intervals.

Caching behavior:
- Only LLM completions are cached (prompts + generated text), never classifier predictions
- This allows re-evaluation with updated classifier models or different constitutions
- Classification always runs fresh on all completions (cached + newly generated)

Example:
    python -m tinker_cookbook.recipes.open_character.eval_constitution_offline \
        --model-path tinker://logs/my-run/checkpoint_1000.pt \
        --model-name meta-llama/Llama-3.1-8B-Instruct \
        --constitution mathematical \
        --num-samples 100
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Callable

import datasets
import httpx
import numpy as np
import tinker
from tinker import types
from tqdm import tqdm
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ============================================================================
# Dataset Field Navigation (from view_introspection.py)
# ============================================================================


def get_nested_value(data: Any, path: str) -> Any:
    """Navigate nested data structure using a path string.
    
    Supports paths like:
        - "input" -> data["input"]
        - ".[0].input" -> data[0]["input"]
        - ".data[0].input" -> data["data"][0]["input"]
        - "conversation.[0].content" -> data["conversation"][0]["content"]
    
    Args:
        data: The data structure to navigate
        path: Dot/bracket notation path string
    
    Returns:
        The value at the specified path, or None if not found
    """
    if not path:
        return data
    
    # Replace bracket notation with dots for easier parsing
    # .[0] -> .0
    # [0] -> 0
    normalized_path = re.sub(r'\[(\d+)\]', r'.\1', path)
    normalized_path = normalized_path.lstrip('.')
    
    parts = normalized_path.split('.')
    current = data
    
    for part in parts:
        if not part:
            continue
            
        try:
            # Try as list index
            if part.isdigit():
                index = int(part)
                current = current[index]
            # Try as dict key
            elif isinstance(current, dict):
                current = current.get(part)
                if current is None:
                    return None
            else:
                return None
        except (KeyError, IndexError, TypeError):
            return None
    
    return current


# ============================================================================
# Cache Management
# ============================================================================


def compute_cache_key(args) -> dict:
    """Compute cache key from relevant arguments.
    
    Note: constitution is NOT included since we cache only LLM completions,
    not classifier predictions. The same completions can be evaluated against
    different constitutions.
    """
    return {
        "model_path": args.model_path,
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "prompt_key": args.prompt_key,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "renderer_name": args.renderer_name,
    }


def get_cache_path(cache_key: dict, cache_dir: str) -> str:
    """Generate cache file path with model name prefix."""
    # Extract and sanitize model name
    model_name = cache_key.get("model_path", "unknown")
    # Convert to safe filename: meta-llama/Llama-3.1-8B-Instruct -> llama-3-1-8b-instruct
    safe_model = model_name.split('/')[-1].lower().replace('.', '-').replace('_', '-')
    
    # Generate hash
    key_str = json.dumps(cache_key, sort_keys=True)
    cache_hash = hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    # Combine: eval_classifier-{model}-{hash}.json
    filename = f"eval_classifier-{safe_model}_{cache_hash}.json"
    return os.path.join(cache_dir, filename)


def load_cache(cache_path: str) -> dict | None:
    """Load cache from file if it exists."""
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load cache: {e}")
        return None


def save_cache(cache_path: str, cache_data: dict):
    """Save cache to file with atomic write."""
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    
    # Atomic write: write to temp file, then rename
    temp_path = cache_path + ".tmp"
    try:
        with open(temp_path, "w") as f:
            json.dump(cache_data, f, indent=2)
        os.replace(temp_path, cache_path)
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


# ============================================================================
# Manual Metric Computation (no evaluate library)
# ============================================================================


def compute_accuracy(predictions: list[str], references: list[str]) -> float:
    """Compute accuracy: fraction of correct predictions."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if len(predictions) == 0:
        return 0.0
    correct = sum(1 for pred, ref in zip(predictions, references) if pred == ref)
    return correct / len(predictions)


def compute_f1_macro(predictions: list[str], references: list[str]) -> float:
    """Compute macro-averaged F1 score across all labels."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if len(predictions) == 0:
        return 0.0
    
    # Get all unique labels
    all_labels = set(predictions) | set(references)
    
    # Compute F1 for each label
    f1_scores = []
    for label in all_labels:
        # True positives, false positives, false negatives
        tp = sum(1 for pred, ref in zip(predictions, references) if pred == label and ref == label)
        fp = sum(1 for pred, ref in zip(predictions, references) if pred == label and ref != label)
        fn = sum(1 for pred, ref in zip(predictions, references) if pred != label and ref == label)
        
        # Precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    
    # Macro average
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


# ============================================================================
# Bootstrap CI Computation
# ============================================================================


def compute_bootstrap_ci(
    predictions: list[str],
    ground_truth: list[str],
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    alpha: float = 0.95,
    seed: int = 42,
) -> tuple[float, tuple[float, float]]:
    """Compute metric with bootstrapped confidence interval."""
    np.random.seed(seed)
    
    # Original metric
    original = metric_fn(predictions, ground_truth)
    
    # Bootstrap resampling
    n = len(predictions)
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        boot_pred = [predictions[i] for i in indices]
        boot_true = [ground_truth[i] for i in indices]
        score = metric_fn(boot_pred, boot_true)
        bootstrap_scores.append(score)
    
    # Compute percentiles for CI
    lower_p = (1 - alpha) / 2 * 100
    upper_p = (1 - (1 - alpha) / 2) * 100
    ci_lower = np.percentile(bootstrap_scores, lower_p)
    ci_upper = np.percentile(bootstrap_scores, upper_p)
    
    return original, (ci_lower, ci_upper)


def format_metric(value: float, ci: tuple[float, float], alpha: float = 0.95) -> str:
    """Format metric with 2 significant digits and CI."""
    # Format to 2 significant figures
    formatted_value = f"{value:.2g}"
    formatted_lower = f"{ci[0]:.2g}"
    formatted_upper = f"{ci[1]:.2g}"
    ci_pct = int(alpha * 100)
    return f"{formatted_value} ({ci_pct}% CI: {formatted_lower}-{formatted_upper})"


# ============================================================================
# Evaluation Functions
# ============================================================================


async def generate_single_completion(
    sampling_client: tinker.SamplingClient,
    prompt: str,
    renderer: renderers.Renderer,
    sampling_params: types.SamplingParams,
    index: int,
    semaphore: asyncio.Semaphore,
) -> str:
    """Generate a single completion with concurrency control."""
    async with semaphore:
        try:
            # Build conversation
            messages = [renderers.Message(role="user", content=prompt)]
            
            # Build model input
            model_input: types.ModelInput = renderer.build_generation_prompt(messages)
            
            # Generate response
            response: types.SampleResponse = await sampling_client.sample_async(
                prompt=model_input, num_samples=1, sampling_params=sampling_params
            )
            tokens: list[int] = response.sequences[0].tokens
            parsed_message: renderers.Message = renderer.parse_response(tokens)[0]
            content = renderers.get_text_content(parsed_message)
            return content
        except Exception as e:
            logger.warning(f"Failed to generate completion for sample {index}: {e}")
            return ""  # Empty completion as fallback


async def generate_completions(
    sampling_client: tinker.SamplingClient,
    prompts: list[str],
    renderer: renderers.Renderer,
    temperature: float,
    max_tokens: int,
    max_concurrent: int = 50,
    top_p: float = 1.0,
    top_k: int = -1,
) -> list[str]:
    """
    Generate completions for a list of prompts in parallel.
    
    Args:
        max_concurrent: Maximum number of concurrent generation requests (default: 50)
    
    Returns:
        List of completion strings (empty string on failure)
    """
    sampling_params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop=renderer.get_stop_sequences(),
    )
    
    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create all tasks for parallel execution
    logger.info(f"Generating {len(prompts)} completions (max {max_concurrent} concurrent)...")
    tasks = [
        generate_single_completion(sampling_client, prompt, renderer, sampling_params, i, semaphore)
        for i, prompt in enumerate(prompts)
    ]
    
    # Run all generations in parallel (preserves input order)
    completions = await tqdm_asyncio.gather(*tasks, desc="Generating completions")
    
    return completions


async def classify_batch(
    texts: list[str],
    classifier_url: str,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """Classify a batch of texts with concurrency control."""
    async with semaphore:
        try:
            response = await client.post(
                classifier_url,
                json={"texts": texts},
            )
            response.raise_for_status()
            result = response.json()
            return result["results"]
        except Exception as e:
            logger.error(f"Failed to classify batch of {len(texts)} texts: {e}")
            # Return empty results for this batch
            return [{"predicted_label": "unknown", "confidence": 0.0, "all_scores": {}} for _ in texts]


async def classify_texts(
    completions: list[str],
    classifier_url: str,
    batch_size: int = 32,
    max_concurrent: int = 50,
) -> list[dict]:
    """
    Classify texts via the classifier API in parallel batches.
    
    Args:
        completions: List of text completions to classify
        classifier_url: Classifier API endpoint
        batch_size: Number of texts per API request (default: 32)
        max_concurrent: Maximum concurrent API requests (default: 50)
    
    Returns:
        List of dicts with 'predicted_label', 'confidence', 'all_scores'
    """
    if not completions:
        return []
    
    # Split into batches
    batches = [completions[i:i + batch_size] for i in range(0, len(completions), batch_size)]
    logger.info(f"Classifying {len(completions)} texts in {len(batches)} batches (max {max_concurrent} concurrent)...")
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create tasks for all batches
    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = [
            classify_batch(batch, classifier_url, client, semaphore)
            for batch in batches
        ]
        
        # Run all classification requests in parallel with progress tracking
        batch_results = await tqdm_asyncio.gather(*tasks, desc="Classifying texts")
    
    # Flatten results
    results = []
    for batch_result in batch_results:
        results.extend(batch_result)
    
    return results


# ============================================================================
# Main Script
# ============================================================================


async def main():
    parser = argparse.ArgumentParser(
        description="Offline constitution evaluation using Tinker API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Required arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Tinker checkpoint URL (e.g., tinker://logs/run/checkpoint.pt)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Base model name for tokenizer (e.g., meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--constitution",
        type=str,
        required=True,
        help="Target constitution to evaluate against",
    )
    
    # Classifier API
    parser.add_argument(
        "--classifier-url",
        type=str,
        default="http://localhost:8000/classify",
        help="Full classifier endpoint URL (default: http://localhost:8000/classify)",
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="LDJnr/Pure-Dove",
        help="HuggingFace dataset name (default: LDJnr/Pure-Dove)",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of examples to evaluate (default: 100)",
    )
    parser.add_argument(
        "--prompt-key",
        type=str,
        default="conversation.[0].input", # default for Pure-Dove
        help="Path to prompt field. Examples: 'input', '.[0].input', '.data[0].question' (default: instruction)",
    )
    
    # Model/renderer arguments
    parser.add_argument(
        "--renderer-name",
        type=str,
        default="llama3",
        help="Renderer name (default: llama3)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Sampling temperature (default: 0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate (default: 2048)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=50,
        help="Maximum concurrent generation requests (default: 50)",
    )
    
    # Bootstrap CI arguments
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of bootstrap samples for CI (default: 1000)",
    )
    parser.add_argument(
        "--ci-alpha",
        type=float,
        default=0.95,
        help="Confidence level for CI (default: 0.95 for 95%% CI)",
    )
    
    # Cache arguments
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="Directory to store evaluation cache (default: ./cache)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching, always regenerate completions",
    )
    
    args = parser.parse_args()
    
    # Compute cache key and path
    cache_data = None
    cache_path = None
    if not args.no_cache:
        cache_key = compute_cache_key(args)
        cache_path = get_cache_path(cache_key, args.cache_dir)
        cache_data = load_cache(cache_path)
        if cache_data:
            logger.info(f"Found cache with {len(cache_data['results'])} samples")
    
    # Validate model path has tinker:// prefix
    if not args.model_path.startswith("tinker://"):
        logger.warning(
            f"Model path '{args.model_path}' does not start with 'tinker://'. "
            "This may cause issues with Tinker API."
        )
    
    # Determine what needs to be generated from cache
    cached_completions = []
    cached_prompts = []
    prompts_to_generate = []
    need_dataset = False
    
    if cache_data and len(cache_data.get("completions", [])) > 0:
        cached_completions = cache_data["completions"]
        num_cached = len(cached_completions)
        
        if args.num_samples <= num_cached:
            # Use cached completions only
            logger.info(f"Using {args.num_samples} samples from cache (no generation needed)")
            cached_completions = cached_completions[:args.num_samples]
            cached_prompts = [c["prompt"] for c in cached_completions]
            prompts_to_generate = []
        else:
            # Need to generate additional samples
            logger.info(f"Using {num_cached} cached samples, generating {args.num_samples - num_cached} new samples")
            cached_prompts = [c["prompt"] for c in cached_completions]
            need_dataset = True
    else:
        # No cache, generate all
        logger.info("No cache found, will generate all completions")
        need_dataset = True
    
    # Only load dataset if we need to generate new samples
    if need_dataset:
        logger.info(f"Loading dataset {args.dataset_name} (split={args.dataset_split})...")
        try:
            ds = datasets.load_dataset(args.dataset_name, split=args.dataset_split)
            logger.info(f"Loaded {len(ds)} examples")
            
            # No shuffling - use deterministic order for cache consistency
            # Extract prompts using path navigation
            prompts = []
            for i in range(min(args.num_samples, len(ds))):
                example = ds[i]
                value = get_nested_value(example, args.prompt_key)
                
                # Convert to string
                if value is None:
                    prompt = f"[Could not extract prompt from dataset index {i}]"
                    logger.warning(f"Could not extract prompt at index {i} with key '{args.prompt_key}'")
                elif isinstance(value, str):
                    prompt = value
                else:
                    prompt = str(value)
                
                prompts.append(prompt)
            
            logger.info(f"Extracted {len(prompts)} prompts")
            
            # Determine which prompts to generate
            num_cached = len(cached_completions)
            prompts_to_generate = prompts[num_cached:args.num_samples]
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return
    
    # Generate new completions if needed
    new_completions = []
    if prompts_to_generate:
        logger.info("Creating Tinker SamplingClient for checkpoint...")
        try:
            service_client = tinker.ServiceClient()
            sampling_client = service_client.create_sampling_client(
                base_model=args.model_name,
                model_path=args.model_path,
            )
        except Exception as e:
            logger.error(f"Failed to create Tinker client: {e}")
            return
        
        # Set up renderer
        tokenizer = get_tokenizer(args.model_name)
        renderer = renderers.get_renderer(name=args.renderer_name, tokenizer=tokenizer)
        
        # Wrap generation in try-finally to ensure cache is saved
        try:
            # Generate completions
            logger.info(f"Generating {len(prompts_to_generate)} completions...")
            generated_texts = await generate_completions(
                sampling_client, prompts_to_generate, renderer, 
                args.temperature, args.max_tokens, args.max_concurrent
            )
            
            # Build new completion records (prompt + completion only, NO classification)
            for prompt, completion in zip(prompts_to_generate, generated_texts):
                new_completions.append({
                    "prompt": prompt,
                    "completion": completion,
                })
        
        except KeyboardInterrupt:
            logger.warning("Interrupted by user (Ctrl+C). Saving partial results...")
            raise  # Re-raise to exit after cleanup
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            logger.warning("Saving any partial results before exiting...")
            raise  # Re-raise to exit after cleanup
        finally:
            # Always save cache if we have any new completions
            if not args.no_cache and new_completions:
                all_completions = cached_completions + new_completions
                cache_data = {
                    "metadata": {
                        **compute_cache_key(args),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "num_samples": len(all_completions),
                    },
                    "completions": all_completions,
                }
                try:
                    save_cache(cache_path, cache_data)
                    logger.info(f"✓ Saved {len(all_completions)} completions to cache: {cache_path}")
                except Exception as cache_e:
                    logger.error(f"Failed to save cache: {cache_e}")
    
    # Combine all completions (cached + new)
    all_completions = cached_completions + new_completions
    all_prompts = [c["prompt"] for c in all_completions]
    all_texts = [c["completion"] for c in all_completions]
    
    # Check classifier API availability
    logger.info("Checking classifier API availability...")
    health_url = args.classifier_url.replace("/classify", "/health")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(health_url)
            response.raise_for_status()
            health_data = response.json()
            if not health_data.get("model_loaded", False):
                logger.error("Classifier API reports model not loaded!")
                return
            logger.info(f"✓ Classifier API ready ({health_data.get('num_constitutions', 0)} constitutions)")
    except Exception as e:
        logger.error(f"Failed to connect to classifier API at {health_url}: {e}")
        logger.error("Please ensure the classifier API is running.")
        return
    
    # Classify ALL completions (cached + new)
    logger.info(f"Classifying {len(all_texts)} completions against constitution '{args.constitution}'...")
    classification_results = await classify_texts(
        all_texts, args.classifier_url, max_concurrent=args.max_concurrent
    )
    
    # Extract predictions and ground truth
    predictions = [r["predicted_label"] for r in classification_results]
    ground_truth = [args.constitution] * len(predictions)
    
    # Compute bootstrap confidence intervals
    logger.info(f"Computing bootstrapped confidence intervals ({args.bootstrap_samples} samples)...")
    
    f1_mean, f1_ci = compute_bootstrap_ci(
        predictions, ground_truth, compute_f1_macro, 
        n_bootstrap=args.bootstrap_samples, alpha=args.ci_alpha
    )
    acc_mean, acc_ci = compute_bootstrap_ci(
        predictions, ground_truth, compute_accuracy,
        n_bootstrap=args.bootstrap_samples, alpha=args.ci_alpha
    )
    
    # Print results
    print()
    print("━" * 60)
    print("Constitution Evaluation Results")
    print("━" * 60)
    print(f"Model:        {args.model_name}")
    print(f"Checkpoint:   {args.model_path}")
    print(f"Constitution: {args.constitution}")
    print(f"Dataset:      {args.dataset_name} ({len(predictions)} samples)")
    print(f"Classifier:   {args.classifier_url}")
    print("━" * 60)
    print(f"F1 Score:  {format_metric(f1_mean, f1_ci, args.ci_alpha)}")
    print(f"Accuracy:  {format_metric(acc_mean, acc_ci, args.ci_alpha)}")
    print("━" * 60)
    print()


if __name__ == "__main__":
    asyncio.run(main())
