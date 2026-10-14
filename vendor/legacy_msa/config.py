"""Import-only configuration shim for the vendored legacy model modules.

The original modules import only NUM_CLASSES from the project-wide config.
Training/runtime settings and path creation stay outside this vendor package.
"""

NUM_CLASSES = 5
