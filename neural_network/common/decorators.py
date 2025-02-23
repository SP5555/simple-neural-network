def requires_build(method):
    def wrapper(self, *args, **kwargs):
        if not self._is_built:
            raise RuntimeError(f"Cannot call `{method.__name__}()` before `build()` has been executed.")
        return method(self, *args, **kwargs)
    return wrapper
