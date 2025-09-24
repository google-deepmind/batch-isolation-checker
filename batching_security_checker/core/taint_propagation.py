#
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Core logic for taint/label propagation."""

import dataclasses
import functools
from typing import Any, Union

import jax
from jax import lax as jlax
from jax import numpy as jnp
from jax import typing as jtyping
from jax._src.numpy import reductions
from jax.experimental import checkify
import numpy as np

from batching_security_checker.core import config_class


@dataclasses.dataclass
class TaintDType:
  jnp_full: Union[type[jnp.uint64], type[jnp.uint32], type[jnp.uint16]]
  np_full: Union[type[np.uint64], type[np.uint32], type[np.uint16]]
  jnp_part: Union[type[jnp.uint32], type[jnp.uint16], type[jnp.uint8]]
  np_part: Union[type[np.uint32], type[np.uint16], type[np.uint8]]
  shift_part: int
  shift_full: int


dtype_options = {
    "uint64": TaintDType(
        jnp_full=jnp.uint64,
        np_full=np.uint64,
        jnp_part=jnp.uint32,
        np_part=np.uint32,
        shift_part=31,
        shift_full=62,
    ),
    "uint32": TaintDType(
        jnp_full=jnp.uint32,
        np_full=np.uint32,
        jnp_part=jnp.uint16,
        np_part=np.uint16,
        shift_part=15,
        shift_full=30,
    ),
    "uint16": TaintDType(
        jnp_full=jnp.uint16,
        np_full=np.uint16,
        jnp_part=jnp.uint8,
        np_part=np.uint8,
        shift_part=7,
        shift_full=14,
    ),
}


def tdtype():
  return dtype_options[config_class.config.taint_dtype]


jax.config.update("jax_enable_x64", True)

JnpTaintScalar = Union[jnp.uint64, jnp.uint32, jnp.uint16]
NpTaintScalar = Union[np.uint64, np.uint32, np.uint16]


def not_a_color() -> JnpTaintScalar:
  """Returns a special u32 value that doesn't represent a color."""
  return jnp.right_shift(
      tdtype().jnp_part(jnp.iinfo(tdtype().jnp_part).max), 1
  )  # 011..1


def identity_element() -> JnpTaintScalar:
  """Returns an untainted element."""
  return tdtype().jnp_full(not_a_color())  # 00..0 011..1


def identity_like(x: jtyping.ArrayLike) -> jax.Array:
  """Create an array full of untainted elements with the same shape as an array."""
  return jnp.full_like(x, identity_element(), dtype=tdtype().jnp_full)


def identity_full(shape: Any) -> jax.Array:
  """Create an array full of untainted elements."""
  return jnp.full(
      fill_value=identity_element(),
      shape=shape,
      dtype=tdtype().jnp_full,
  )


def identity_element_concrete() -> NpTaintScalar:
  """Returns an untainted element."""
  return tdtype().np_full(np.right_shift(np.iinfo(tdtype().np_part).max, 1))


def from_color_to_taint(x: jtyping.ArrayLike) -> jax.Array:
  """Encodes a tensor of uint32 color identifiers into a tensor of taint monoid elements."""

  if hasattr(x, "dtype"):
    # x: Union[jax.Array, jnp.ndarray] = x
    if not jnp.isdtype(x.dtype, tdtype().jnp_part):
      raise ValueError(f"The input tensor {x} needs to be {tdtype().jnp_part}.")
  elif isinstance(x, tdtype().jnp_part):
    pass
  else:
    raise ValueError(f"The input tensor {x} needs to be {tdtype().jnp_part}.")

  checkify.check(
      jnp.all(jnp.less(x, not_a_color())),
      "Color tensor label too large (must be smaller than not_a_color()).",
  )

  x = jnp.astype(x, tdtype().jnp_full)
  x = jnp.bitwise_or(jnp.left_shift(x, tdtype().shift_part), x)

  if not jnp.isdtype(x, tdtype().jnp_full):
    raise ValueError(
        f"Taint tensor {x} needs to be {tdtype().jnp_full}. You may need to set"
        " the environment variable: JAX_ENABLE_X64=true"
    )

  return x


def from_taint_to_unique_interfering_colors(
    x: jtyping.ArrayLike,
) -> list[dict[str, Any]]:
  """Convert a tensor of taint monoid elements into unique interfering colors."""

  max_only = jnp.astype(
      jnp.right_shift(
          jnp.bitwise_and(x, _max_only_mask()), tdtype().shift_part
      ),
      tdtype().jnp_part,
  )
  min_only = jnp.astype(jnp.bitwise_and(x, _min_only_mask()), tdtype().jnp_part)

  combined = jnp.stack([min_only, max_only], axis=-1)
  combined = jnp.reshape(combined, (-1, 2))
  combined_unique = jnp.unique(combined, axis=0)

  colors = []
  for min_c, max_c in combined_unique.tolist():
    if min_c != max_c:
      colors.append({"min": min_c, "max": max_c})
  return colors


def from_taint_to_color(x: jtyping.ArrayLike) -> jax.Array:
  """Decodes a tensor of taint monoid elements into a tensor of uint32 color identifiers."""

  max_only = jnp.right_shift(
      jnp.bitwise_and(x, _max_only_mask()), tdtype().shift_part
  )
  min_only = jnp.bitwise_and(x, _min_only_mask())

  color_tensor = jnp.where(
      max_only == min_only,
      jnp.astype(min_only, tdtype().jnp_part),
      not_a_color(),
  )
  return color_tensor.astype(tdtype().jnp_part)


def _min_only_mask() -> JnpTaintScalar:
  return identity_element()  # 00|00..0|11..1


def _max_only_mask() -> JnpTaintScalar:
  return jnp.left_shift(_min_only_mask(), tdtype().shift_part)  # 00|11..1|00..0


def _config_mask() -> JnpTaintScalar:
  return jnp.left_shift(
      jnp.astype(3, tdtype().jnp_full), tdtype().shift_full
  )  # 11|00..0|00..0


def _config_random_flag() -> JnpTaintScalar:
  return jnp.left_shift(
      jnp.astype(1, tdtype().jnp_full), tdtype().shift_full
  )  # 01|00..0|00..0


def identity_taint(x: jtyping.ArrayLike):
  """Propagates taint from inputs to outputs."""
  return x


def taint_reduction(x: jtyping.ArrayLike, dimensions):
  """Reduce taint along the given dimensions."""
  result = jlax.reduce(
      x, identity_element(), binary_elementwise_taint, dimensions
  )
  return result


def set_random(x: jtyping.ArrayLike) -> jtyping.ArrayLike:
  return jnp.bitwise_or(x, _config_random_flag())


def is_random(x: jtyping.ArrayLike) -> jax.Array:
  out = jnp.bitwise_and(x, _config_random_flag())
  out = out == _config_random_flag()
  return out


def is_determinisitic(x: jtyping.ArrayLike) -> jax.Array:
  out = jnp.logical_not(is_random(x))
  return out


def binary_elementwise_taint(
    x: jtyping.ArrayLike, y: jtyping.ArrayLike
) -> jtyping.ArrayLike:
  """Propagates taint for binary elementwise operators."""

  tmin = jnp.minimum(
      jnp.bitwise_and(x, _min_only_mask()), jnp.bitwise_and(y, _min_only_mask())
  )

  tmax = jnp.maximum(
      jnp.bitwise_and(x, _max_only_mask()), jnp.bitwise_and(y, _max_only_mask())
  )

  # logical or of config flags
  tconfig = jnp.bitwise_or(
      jnp.bitwise_and(x, _config_mask()), jnp.bitwise_and(y, _config_mask())
  )

  out = jnp.bitwise_or(tmin, tmax)
  out = jnp.bitwise_or(out, tconfig)
  return out


def is_all_deterministic(x: jtyping.ArrayLike) -> jax.Array:
  return jnp.all(is_determinisitic(x))


def is_all_untainted(x: jtyping.ArrayLike) -> jax.Array:
  """Returns true if the value is untainted."""
  is_equal = jnp.equal(x, identity_element())
  return jnp.all(is_equal)


def is_all_untainted_deterministic(x: jtyping.ArrayLike) -> jax.Array:
  """Returns true if the value is untainted and deterministic."""
  out = jnp.logical_and(jnp.equal(x, identity_element()), is_determinisitic(x))
  return jnp.all(out)


def has_all_single_taint(x: jtyping.ArrayLike) -> jax.Array:
  """A value has a single taint if the observed min and max taint values are the same."""
  max_only = jnp.right_shift(
      jnp.bitwise_and(x, _max_only_mask()), tdtype().shift_part
  )
  min_only = jnp.bitwise_and(x, _min_only_mask())
  result = jnp.logical_or(max_only == min_only, x == identity_element())
  result = jnp.all(result)

  return result


@functools.partial(
    jax.jit,
    static_argnames=("axis", "dtype", "keepdims", "promote_integers"),
    inline=True,
)
def _reduce_taint(
    a: jtyping.ArrayLike,
    axis: reductions.Axis = None,
    dtype: jtyping.DTypeLike | None = None,
    out: None = None,
    keepdims: bool = False,
    initial: jtyping.ArrayLike | None = None,
    where: jtyping.ArrayLike | None = None,
    promote_integers: bool = True,
) -> jax.Array:
  """Apply taint reduction to a tensor."""

  identity_elem = identity_element_concrete()

  # pylint: disable=protected-access
  return reductions._reduction(
      a,
      "taint",
      binary_elementwise_taint,
      identity_elem,
      preproc=None,
      bool_op=None,
      upcast_f16_for_computation=False,
      axis=axis,
      dtype=dtype,
      out=out,
      keepdims=keepdims,
      initial=initial,
      where_=where,
      parallel_reduce=None,
      promote_integers=promote_integers,
  )


def taint(
    a: jtyping.ArrayLike,
    axis: reductions.Axis = None,
    dtype: jtyping.DTypeLike | None = None,
    out: None = None,
    keepdims: bool = False,
    initial: jtyping.ArrayLike | None = None,
    where: jtyping.ArrayLike | None = None,
    promote_integers: bool = False,
) -> jax.Array:
  """Apply taint reduction to a tensor."""

  checkify.check(
      jnp.isdtype(a, tdtype().jnp_full),
      "Taint tensor {a} needs to be {tdtype().jnp_full}.",
  )

  if dtype:
    checkify.check(
        jnp.isdtype(dtype, tdtype().jnp_full),
        "Taint tensor {a} needs to be {tdtype().jnp_full}.",
    )

  if promote_integers:
    raise ValueError("promote_integers not supported")

  result = _reduce_taint(
      a,
      axis=reductions._ensure_optional_axes(axis),
      dtype=dtype,
      out=out,
      keepdims=keepdims,
      initial=initial,
      where=where,
      promote_integers=promote_integers,
  )

  return result
