# src/fingerprint_dataset/pipeline.py
"""
End-to-end classical fingerprint recognition pipeline: raw image ->
segmentation -> orientation -> frequency -> enhancement -> minutiae ->
matching. Ties together every module in this package behind a small,
serializable result object, so callers (evaluation scripts, a future
API, etc.) don't need to know the individual module APIs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .segmentation import segment
from .orientation import estimate_orientation
from .frequency import estimate_frequency_field
from .enhancement import enhance
from .minutiae import extract_minutiae_from_enhanced, MinutiaeConfig, MinutiaeResult
from .matcher import MinutiaeMatcher, MatcherConfig, MatchResult, Template, make_template


@dataclass
class PipelineResult:
    """Everything the pipeline produced for one fingerprint image, kept
    together so a caller can inspect intermediate stages (for debugging
    or visualization) without re-running anything."""
    image: np.ndarray
    mask: np.ndarray
    orientations: np.ndarray
    strengths: np.ndarray
    periods: np.ndarray
    enhanced: np.ndarray
    minutiae_result: MinutiaeResult
    template: Template


def process_image(image: np.ndarray,
                   minutiae_config: MinutiaeConfig | None = None) -> PipelineResult:
    """
    Run the full classical pipeline on one raw grayscale fingerprint
    image and return every intermediate result plus the final
    matcher.Template (minutiae + mask) ready to be compared with
    another image via match().
    """
    mask = segment(image)
    orientations, strengths = estimate_orientation(image, mask)
    periods = estimate_frequency_field(image, orientations, mask)
    enhanced = enhance(image, mask, orientations, periods, strengths=strengths)

    minutiae_result = extract_minutiae_from_enhanced(
        enhanced, mask, coherence=strengths, period_map=periods,
        reference=image, config=minutiae_config,
    )
    template = make_template(minutiae_result.minutiae, mask)

    return PipelineResult(
        image=image, mask=mask, orientations=orientations, strengths=strengths,
        periods=periods, enhanced=enhanced, minutiae_result=minutiae_result,
        template=template,
    )


def match(result_a: PipelineResult, result_b: PipelineResult,
          matcher_config: MatcherConfig | None = None) -> MatchResult:
    """
    Compare two pipeline results (as produced by process_image) and
    return a MatchResult with the similarity score, alignment, and
    paired minutiae.
    """
    matcher = MinutiaeMatcher(matcher_config)
    return matcher.match(result_a.template, result_b.template)


def match_images(image_a: np.ndarray, image_b: np.ndarray,
                  minutiae_config: MinutiaeConfig | None = None,
                  matcher_config: MatcherConfig | None = None) -> MatchResult:
    """
    Convenience one-shot: run the full pipeline on two raw images and
    return their match score directly, without needing to keep the
    intermediate PipelineResult objects around.
    """
    result_a = process_image(image_a, minutiae_config)
    result_b = process_image(image_b, minutiae_config)
    return match(result_a, result_b, matcher_config)