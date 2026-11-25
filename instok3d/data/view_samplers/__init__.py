from ._registry import registry
from .base import BaseViewSampler

registry.import_current_directories(__file__, __name__)
get_view_sampler_cls = registry.get_target_cls

__all__ = ['BaseViewSampler', 'get_view_sampler_cls']
