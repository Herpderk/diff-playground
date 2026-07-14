# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Project-wide softjax configuration."""

import functools
import math

import softjax as _softjax

SOFTNESS = 0.01


def set_global_softness(softness: float) -> None:
  """Sets the finite, positive softness used by project softjax wrappers.

  Args:
    softness: A finite value strictly greater than zero. The value is
      converted to ``float`` before it is stored.

  Raises:
    ValueError: If ``softness`` cannot be converted to a finite, positive
      value.
  """
  try:
    softness = float(softness)
  except (TypeError, ValueError, OverflowError) as error:
    raise ValueError(
        "softness must be a finite value greater than zero"
    ) from error
  if not math.isfinite(softness) or softness <= 0.0:
    raise ValueError("softness must be a finite value greater than zero")

  global SOFTNESS
  SOFTNESS = softness


def _with_softness(fn, softness_position):
  """Binds the softness argument of a softjax function to ``SOFTNESS``."""

  @functools.wraps(fn)
  def wrapped(*args, **kwargs):
    if len(args) > softness_position:
      args = (
          *args[:softness_position],
          SOFTNESS,
          *args[softness_position + 1 :],
      )
      kwargs.pop("softness", None)
    else:
      kwargs["softness"] = SOFTNESS
    return fn(*args, **kwargs)

  return wrapped


# Keep the external softjax argument order while forcing the project value,
# including when callers provide softness positionally or by keyword.
abs = _with_softness(_softjax.abs, 1)
clip = _with_softness(_softjax.clip, 3)
greater = _with_softness(_softjax.greater, 2)
greater_st = _with_softness(_softjax.greater_st, 2)
greater_equal = _with_softness(_softjax.greater_equal, 2)
greater_equal_st = _with_softness(_softjax.greater_equal_st, 2)
less = _with_softness(_softjax.less, 2)
less_st = _with_softness(_softjax.less_st, 2)
less_equal = _with_softness(_softjax.less_equal, 2)
less_equal_st = _with_softness(_softjax.less_equal_st, 2)
max = _with_softness(_softjax.max, 3)
min = _with_softness(_softjax.min, 3)
relu = _with_softness(_softjax.relu, 1)

# Operations without a softness argument are passed through unchanged.
all = _softjax.all
any = _softjax.any
arccos = _softjax.arccos
arcsin = _softjax.arcsin
div = _softjax.div
logical_and = _softjax.logical_and
logical_not = _softjax.logical_not
logical_or = _softjax.logical_or
norm = _softjax.norm
where = _softjax.where
