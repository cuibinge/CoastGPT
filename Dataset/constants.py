"""Generic dataset constants.

The training pipeline must not route by task class or feature category. These
constants are kept only for backward-compatible sample metadata fields.
"""

GENERIC_CONDITION = "generic"

TASK2ID = {GENERIC_CONDITION: 0}
ELEMENT2ID = {GENERIC_CONDITION: 0}
