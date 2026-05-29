def run_eval(*args, **kwargs):
    from .pipeline import main
    return main(*args, **kwargs)


def serve_report(*args, **kwargs):
    from .report import main
    return main(*args, **kwargs)


__all__ = ["run_eval", "serve_report"]
