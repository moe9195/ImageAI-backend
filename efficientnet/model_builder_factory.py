
"""Utilities for model builder or input size."""

import efficientnet_builder
from condconv import efficientnet_condconv_builder
from edgetpu import efficientnet_edgetpu_builder
from lite import efficientnet_lite_builder
from tpu import efficientnet_tpu_builder


def get_model_builder(model_name):
  """Get the model_builder module for a given model name."""
  if model_name.startswith('efficientnet-lite'):
    return efficientnet_lite_builder
  elif model_name.startswith('efficientnet-edgetpu-'):
    return efficientnet_edgetpu_builder
  elif model_name.startswith('efficientnet-condconv-'):
    return efficientnet_condconv_builder
  elif model_name.startswith('efficientnet-tpu-'):
    return efficientnet_tpu_builder
  elif model_name.startswith('efficientnet-'):
    return efficientnet_builder
  else:
    raise ValueError(
        'Model must be either efficientnet-b* or efficientnet-edgetpu* or'
        'efficientnet-condconv*, efficientnet-lite*')


def get_model_input_size(model_name):
  """Get model input size for a given model name."""
  if model_name.startswith('efficientnet-lite'):
    _, _, image_size, _ = (
        efficientnet_lite_builder.efficientnet_lite_params(model_name))
  elif model_name.startswith('efficientnet-edgetpu-'):
    _, _, image_size, _ = (
        efficientnet_edgetpu_builder.efficientnet_edgetpu_params(model_name))
  elif model_name.startswith('efficientnet-condconv-'):
    _, _, image_size, _, _ = (
        efficientnet_condconv_builder.efficientnet_condconv_params(model_name))
  elif model_name.startswith('efficientnet-tpu'):
    _, _, image_size, _ = efficientnet_tpu_builder.efficientnet_tpu_params(
        model_name)
  elif model_name.startswith('efficientnet'):
    _, _, image_size, _ = efficientnet_builder.efficientnet_params(model_name)
  else:
    raise ValueError(
        'Model must be either efficientnet-b* or efficientnet-tpu-b* or efficientnet-edgetpu* or '
        'efficientnet-condconv*, efficientnet-lite*')
  return image_size
