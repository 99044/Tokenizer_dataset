"""Remove one diagnostic-only keyword unsupported by PyTorch 2.6.

OLMo-core 2.6.0 declares torch>=2.6.0, but its optional TE attention backend
uses compiler.disable(reason=...), added in PyTorch 2.8. This decorator is
evaluated even when using SDPA. Preserve compiler.disable's behavior and remove
only its diagnostic reason. The source hash makes the patch version-specific.
"""

import hashlib
from importlib.metadata import distribution, version


def main():
    assert version('ai2-olmo-core') == '2.6.0'
    assert version('torch').split('+')[0] == '2.6.0'
    backend = distribution('ai2-olmo-core').locate_file('olmo_core/nn/attention/backend.py')
    source = backend.read_text()
    expected = '2a1d3b2df5e417e244ddf7f18f0ef37014438e87334103fb72f031530e504e7e'
    assert hashlib.sha256(source.encode()).hexdigest() == expected, 'unexpected OLMo-core backend source'
    old = ('@torch.compiler.disable(\n'
           '        reason="Transformer Engine attention uses Python/pybind setup that Dynamo should not trace"\n'
           '    )')
    assert source.count(old) == 1
    backend.write_text(source.replace(old, '@torch.compiler.disable()'))
    print('Applied OLMo-core 2.6.0 diagnostic-keyword compatibility patch for PyTorch 2.6')


if __name__ == '__main__':
    main()
