from importlib.metadata import version

# Single source of truth is [project] version in pyproject.toml. uv_build does not
# support dynamic = ["version"], so the version is read from the installed
# distribution metadata instead of duplicated in source.
__version__ = version("aap_transformers")

__all__ = ["__version__"]
