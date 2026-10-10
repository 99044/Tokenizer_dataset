"""Fix the pinned FLA backward kernel's Triton 3.2 layout mismatch.

Multiplication is performed before transposition, preserving the same values
and arithmetic count. Apply before importing FLA, since @triton.jit captures
the source at import time.
"""

from importlib.metadata import distribution, version


OLD = '        b_ktb = b_kt * b_b[None, :]'
NEW = '        b_ktb = tl.trans(b_k * b_b[:, None])'


def patched_source(source):
    if source.count(OLD) != 1 or NEW in source:
        raise ValueError('Expected exactly one FLA transpose/multiply expression')
    return source.replace(OLD, NEW)


def require_compatible_fla():
    """Fail before model allocation when using the known broken image."""
    if version('triton') != '3.2.0':
        return
    path = distribution('flash-linear-attention').locate_file('fla/ops/gated_delta_rule/wy_fast.py')
    source = path.read_text()
    if OLD in source:
        raise RuntimeError(
            'The installed FLA backward kernel is incompatible with Triton 3.2. '
            'Use --image scicomp26:submission-env-hybrid-v3; build it with '
            'docker build --target hybrid_kernel_compat -f training/src/Dockerfile '
            '-t scicomp26:submission-env-hybrid-v3 .'
        )


def main():
    if version('torch').split('+')[0] != '2.6.0' or version('triton') != '3.2.0':
        raise ValueError('This patch is reviewed for PyTorch 2.6 / Triton 3.2 only')
    if version('flash-linear-attention') != '0.4.2':
        raise ValueError('This patch is reviewed for the pinned FLA 0.4.2 source only')
    path = distribution('flash-linear-attention').locate_file('fla/ops/gated_delta_rule/wy_fast.py')
    source = path.read_text()
    if source.count(NEW) == 1 and OLD not in source:
        print('FLA Triton 3.2 compatibility patch already applied')
        return
    path.write_text(patched_source(source))
    print('Applied FLA Triton 3.2 transpose/multiply compatibility patch')


if __name__ == '__main__':
    main()
