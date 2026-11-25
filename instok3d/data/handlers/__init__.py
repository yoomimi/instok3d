from ._registry import registry

registry.import_current_directories(__file__, __name__)
get_handler_cls = registry.get_target_cls

__all__ = ['get_handler_cls']
