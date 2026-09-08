import sys

_NOTICE = (
    "llm_policy.py is a placeholder: the prompt and stopping-decision "
    "logic are withheld pending peer review. The complete implementation "
    "will be published in this repository once the accompanying paper is "
    "accepted."
)
print(f"[llm_policy] {_NOTICE}", file=sys.stderr)

# Config placeholders -- see the paper for the values used.
API_URL = None
MODEL_NAME = None
TOP_K = None
MAX_TOKENS = None
SYSTEM_PROMPT = None


def llm_stopping_policy(look_scores, class_to_char, session=None):
    """The full sequential decision loop for ONE trial.

    look_scores: list of length K, each an (n_classes,) FBECCA score vector
                 for one look (already leave-one-block-out honest).
    Returns (predicted_class, n_looks_used).

    Withheld pending publication -- see module docstring.
    """
    raise NotImplementedError(_NOTICE)
