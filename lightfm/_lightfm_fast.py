try:
    # Import Rust extension (preferred)
    from ._lightfm_fast_rust import *  # NOQA
    from ._lightfm_fast_rust import __test_in_positives  # NOQA
except ImportError:
    try:
        # Fall back to OpenMP-enabled Cython extension
        from ._lightfm_fast_openmp import *  # NOQA
        from ._lightfm_fast_openmp import __test_in_positives  # NOQA
    except ImportError:
        # Fall back on OpenMP-less Cython extension
        import warnings

        warnings.warn(
            "LightFM was compiled without OpenMP support. "
            "Only a single thread will be used."
        )

        from ._lightfm_fast_no_openmp import *  # NOQA
        from ._lightfm_fast_no_openmp import __test_in_positives  # NOQA
