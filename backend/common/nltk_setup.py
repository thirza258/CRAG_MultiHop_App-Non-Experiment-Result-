"""Make sure the NLTK data the app needs is available.

In Docker the data is baked into the image at build time, so this is a
no-op. In local development (or if the image data is missing) each
missing package is downloaded once at import time instead of crashing
the app with a LookupError.
"""

import logging

import nltk

logger = logging.getLogger(__name__)

# (find path, download name)
REQUIRED_NLTK_DATA = [
    ("corpora/stopwords", "stopwords"),
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
]

_checked = False


def ensure_nltk_data() -> None:
    """Verify required NLTK data, downloading anything missing."""
    global _checked
    if _checked:
        return

    missing = []
    for find_path, package in REQUIRED_NLTK_DATA:
        try:
            nltk.data.find(find_path)
        except LookupError:
            missing.append(package)

    for package in missing:
        logger.info(f"NLTK data '{package}' missing — downloading...")
        try:
            nltk.download(package, quiet=True)
            nltk.data.find(
                next(fp for fp, p in REQUIRED_NLTK_DATA if p == package)
            )
        except Exception as exc:
            raise RuntimeError(
                f"Required NLTK data '{package}' is missing and could not be "
                f"downloaded ({exc}). Run: python -c \"import nltk; "
                f"nltk.download('{package}')\" — or rebuild the Docker image."
            ) from exc

    _checked = True
