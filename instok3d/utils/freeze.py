from torch import nn


def convert_to_buffer(module: nn.Module, persistent: bool = True):
    """Turn every parameter and buffer of ``module`` into a buffer with the given persistence."""
    for child in module.children():
        convert_to_buffer(child, persistent)

    for name, tensor in (
            *module.named_parameters(recurse=False),
            *module.named_buffers(recurse=False),
    ):
        value = tensor.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)
