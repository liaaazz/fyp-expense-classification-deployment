import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.database import get_shared_learning_examples


def normalize_text(text):
    """Normalize text for consistent matching."""
    return " ".join(str(text or "").casefold().split())



def predict_from_feedback(
    merchant_name,
    purchased_items,
    threshold=0.25,
    item_weight=0.8,
):
    """
    Predict an expense category using approved shared feedback.
    Purchased-item similarity is the main matching signal.
    Merchant similarity is supporting evidence.
    """
    examples = get_shared_learning_examples()

    query_items = normalize_text(purchased_items)
    query_merchant = normalize_text(merchant_name)

    if not examples or not query_items:
        return None

    # Only use feedback examples with items and confirmed categories.
    valid_examples = [
        example for example in examples
        if normalize_text(example.get("purchased_items"))
        and example.get("confirmed_category")
    ]

    if not valid_examples:
        return None

    example_items = [
        normalize_text(example.get("purchased_items"))
        for example in valid_examples
    ]

    example_merchants = [
        normalize_text(example.get("merchant_name"))
        for example in valid_examples
    ]

    # Compare purchased items independently of merchant names.
    try:
        item_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        item_matrix = item_vectorizer.fit_transform(
            example_items + [query_items]
        )
        item_scores = cosine_similarity(
            item_matrix[-1],
            item_matrix[:-1],
        ).ravel()
    except ValueError:
        return None

    # Ignore feedback with no meaningful item similarity.
    if item_scores.max() < threshold:
        return None

    # Merchant similarity is supporting evidence, not a requirement.
    merchant_scores = np.zeros(len(valid_examples))

    if query_merchant and any(example_merchants):
        try:
            merchant_vectorizer = TfidfVectorizer(
                ngram_range=(1, 2)
            )
            merchant_matrix = merchant_vectorizer.fit_transform(
                example_merchants + [query_merchant]
            )
            merchant_scores = cosine_similarity(
                merchant_matrix[-1],
                merchant_matrix[:-1],
            ).ravel()
        except ValueError:
            pass

    # Item similarity remains the main signal.
    scores = (
        item_weight * item_scores
        + (1 - item_weight) * merchant_scores
    )

    # Consider only examples whose item similarity is close to
    # the strongest item match.
    best_item_score = float(item_scores.max())
    candidate_indices = [
        index
        for index, score in enumerate(item_scores)
        if score >= threshold
        and score >= best_item_score - 0.05
    ]

    # Do not guess when similarly relevant items have conflicting
    # confirmed categories.
    competing_categories = {
        valid_examples[index]["confirmed_category"]
        for index in candidate_indices
    }

    if len(competing_categories) != 1:
        return None

    # Among examples with the same confirmed category, use the
    # combined score to select the strongest match.
    best_index = max(
        candidate_indices,
        key=lambda index: scores[index],
    )

    return valid_examples[best_index]["confirmed_category"]
