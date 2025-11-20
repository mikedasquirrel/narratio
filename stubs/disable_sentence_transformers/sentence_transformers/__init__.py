"""
Stub module that intentionally disables the real `sentence_transformers`
package for commands that would otherwise trigger TensorFlow/Metal mutex
deadlocks on macOS.

Include the parent directory (`stubs/disable_sentence_transformers`) at the
front of PYTHONPATH to activate this stub.
"""

raise ImportError(
    "sentence-transformers temporarily disabled via local stub to avoid "
    "TensorFlow mutex deadlocks. Remove the stub directory from PYTHONPATH to "
    "restore the real dependency."
)


