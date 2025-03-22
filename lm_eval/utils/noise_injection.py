"""
Noise injection utilities for LM Evaluation Harness.
Allows for adding controlled amounts of noise to model outputs during evaluation.
"""

import torch
from typing import List, Dict, Any, Optional, Union, Callable

# Global noise level variable
current_noise_level = 0.0

def corrupt_all_outputs(module, input, output):
    """
    Inject Gaussian noise into model outputs using an externally controlled noise level.
    
    Args:
        module: PyTorch module
        input: Module input
        output: Module output
    
    Returns:
        Output with added noise
    """
    global current_noise_level
    
    if isinstance(output, tuple):  
        noisy_outputs = tuple(
            o + torch.randn_like(o, dtype=o.dtype) * current_noise_level if isinstance(o, torch.Tensor) else o
            for o in output
        )
        return noisy_outputs
    elif isinstance(output, torch.Tensor):  
        return output + torch.randn_like(output, dtype=output.dtype) * current_noise_level
    return output


def set_noise_level(level: float) -> float:
    """
    Set the global noise level for all registered hooks.
    
    Args:
        level: Noise level (0.0-1.0)
    
    Returns:
        Current noise level after setting
    """
    global current_noise_level
    old_level = current_noise_level
    current_noise_level = level
    return current_noise_level


def get_noise_level() -> float:
    """
    Get the current global noise level.
    
    Returns:
        Current noise level
    """
    global current_noise_level
    return current_noise_level


def detect_transformer_layers(model: torch.nn.Module) -> List[torch.nn.Module]:
    """
    Detect Transformer layers in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        List of detected Transformer layers
    """
    layers = []
    
    # LLaMA style
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    # GPT style
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
    # BERT style
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        layers = model.encoder.layer
    # Direct layers access
    elif hasattr(model, 'layers'):
        layers = model.layers
    # Generic fallback: search for components that might be transformer layers
    else:
        for name, module in model.named_modules():
            if any(x in name.lower() for x in ['block', 'layer']) and hasattr(module, 'attention'):
                layers.append(module)
    
    return layers


def add_noise_hooks(model: torch.nn.Module) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Add noise injection hooks to all Transformer layers in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        List of hook handles
    """
    hooks = []
    layers = detect_transformer_layers(model)
    
    if not layers:
        # Fallback: try to attach hooks to specific module types that might be part of Transformer blocks
        for name, module in model.named_modules():
            if any(x in name.lower() for x in ['attention', 'mlp', 'ffn']):
                hooks.append(module.register_forward_hook(corrupt_all_outputs))
    else:
        # Attach hooks to detected layers
        for layer in layers:
            hooks.append(layer.register_forward_hook(corrupt_all_outputs))
    
    return hooks


def remove_noise_hooks(hooks: List[torch.utils.hooks.RemovableHandle]) -> None:
    """
    Remove all registered noise hooks.
    
    Args:
        hooks: List of hook handles to remove
    """
    for hook in hooks:
        hook.remove()