"""
Constitution Classifier API - FastAPI service for batch text classification.

This service wraps a HuggingFace transformer-based constitution classifier,
accepting batch text classification requests and returning predictions with
confidence scores.

Installation (in deployment environment):
    pip install fastapi uvicorn transformers torch evaluate

Usage:
    python -m tinker_cookbook.recipes.open_character.classifier_api \
        --model-path /path/to/classifier-model \
        --host 0.0.0.0 \
        --port 8000

Examples:
    # Health check
    curl http://localhost:8000/health

    # Classify texts
    curl -X POST http://localhost:8000/classify \
      -H "Content-Type: application/json" \
      -d '{
        "texts": [
          "Let me solve this step by step using a systematic approach.",
          "Whatever, I guess that works if you want to do it that way."
        ]
      }'

Expected Response:
    {
      "results": [
        {
          "predicted_label": "mathematical",
          "confidence": 0.89,
          "all_scores": {
            "mathematical": 0.89,
            "sarcastic": 0.05,
            "british": 0.03,
            ...
          }
        },
        ...
      ]
    }
"""

import argparse
import logging
import os
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state for model and tokenizer
model: Any | None = None
tokenizer: Any | None = None
LABEL2ID: dict[str, int] = {}
ID2LABEL: dict[int, str] = {}

# FastAPI app
app = FastAPI(
    title="Constitution Classifier API",
    description="Batch text classification for constitution adherence",
    version="0.1.0",
)


# ============================================================================
# Pydantic Models
# ============================================================================


class ClassifyRequest(BaseModel):
    """Request schema for batch classification."""

    texts: list[str] = Field(..., description="List of texts to classify", min_length=1)


class ClassificationResult(BaseModel):
    """Single classification result."""

    predicted_label: str = Field(..., description="Predicted constitution label")
    confidence: float = Field(..., description="Confidence score for predicted label")
    all_scores: dict[str, float] = Field(
        ..., description="Probability distribution across all constitution labels"
    )


class ClassifyResponse(BaseModel):
    """Response schema for batch classification."""

    results: list[ClassificationResult] = Field(
        ..., description="Classification results in same order as input texts"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    num_constitutions: int


# ============================================================================
# Model Loading
# ============================================================================


def load_model(model_path: str) -> None:
    """Load the classifier model and tokenizer at startup."""
    global model, tokenizer, LABEL2ID, ID2LABEL

    logger.info("Loading constitution classifier...")

    # Get available constitutions
    constitutions = [
        "sarcasm",
        "humor",
        "remorse",
        "goodness",
        "loving",
        "misalignment",
        "nonchalance",
        "impulsiveness",
        "sycophancy",
        "mathematical",
        "poeticism"
    ]
    logger.info(f"Found {len(constitutions)} constitutions: {constitutions}")

    # Build label mappings
    LABEL2ID = {const: i for i, const in enumerate(constitutions)}
    ID2LABEL = {i: const for const, i in LABEL2ID.items()}

    # Load tokenizer
    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        problem_type="single_label_classification",
    )
    model.eval()

    logger.info("Model loaded successfully!")


@app.on_event("startup")
async def startup_event():
    """Load model on startup if CLASSIFIER_MODEL_PATH env var is set."""
    model_path = os.environ.get("CLASSIFIER_MODEL_PATH")
    if model_path is not None:
        logger.info(f"Loading model from CLASSIFIER_MODEL_PATH: {model_path}")
        load_model(model_path)
    else:
        logger.warning("CLASSIFIER_MODEL_PATH environment variable not set. Model will not be loaded automatically.")


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify model is loaded."""
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        num_constitutions=len(LABEL2ID),
    )


@app.post("/classify", response_model=ClassifyResponse)
async def classify_texts(request: ClassifyRequest):
    """
    Classify a batch of texts and return predictions with confidence scores.

    Args:
        request: ClassifyRequest containing list of texts

    Returns:
        ClassifyResponse with classification results

    Raises:
        HTTPException: 503 if model not loaded, 422 for invalid input
    """
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs.",
        )

    # Validate inputs
    if not request.texts:
        raise HTTPException(
            status_code=422,
            detail="No texts provided for classification",
        )

    # Check for empty texts and log warnings
    for i, text in enumerate(request.texts):
        if not text or not text.strip():
            logger.warning(f"Empty text at index {i}")

    try:
        # Tokenize all texts (with truncation to 8192 tokens)
        inputs = tokenizer(
            request.texts,
            truncation=True,
            max_length=8192,
            padding=True,
            return_tensors="pt",
        )

        # Move to same device as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Build results
        results = []
        for prob_dist in probs:
            # Get predicted label
            pred_idx = prob_dist.argmax().item()
            predicted_label = ID2LABEL[pred_idx]
            confidence = prob_dist[pred_idx].item()

            # Get all scores
            all_scores = {ID2LABEL[i]: prob_dist[i].item() for i in range(len(ID2LABEL))}

            results.append(
                ClassificationResult(
                    predicted_label=predicted_label,
                    confidence=confidence,
                    all_scores=all_scores,
                )
            )

        return ClassifyResponse(results=results)

    except Exception as e:
        logger.error(f"Classification error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}",
        )


# ============================================================================
# CLI Entry Point
# ============================================================================


def main():
    """Main entry point for running the API server."""
    parser = argparse.ArgumentParser(
        description="Constitution Classifier API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained classifier model directory",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()
    
    # Set environment variable so startup event can load it (for reload mode)
    os.environ["CLASSIFIER_MODEL_PATH"] = args.model_path

    # Load model before starting server (for non-reload mode and immediate feedback)
    logger.info(f"Loading model from {args.model_path}")
    load_model(args.model_path)

    # Start server
    import uvicorn

    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(
        "classifier_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
