# This file contains tests that exercise multiple AST nodes
import shutil

from sympy.external import import_module
from sympy.printing.ccode import ccode
from sympy.utilities._compilation import compile_link_import_strings
from sympy.utilities.pytest import skip
from sympy.sets import Range
from sympy.codegen.ast import (
    FunctionDefinition, FunctionPrototype, Variable, Pointer, real, Assignment,
    integer, Variable, CodeBlock, While
)
from sympy.codegen.cnodes import void, PreIncrement
from sympy.codegen.cutils import render_as_source_file

np = import_module('numpy')

def _mk_func1():
    declars = n, inp, out = Variable('n', integer), Pointer('inp', real), Pointer('out', real)
    i = Variable('i', integer)
    whl = While(i<n, [Assignment(out[i], inp[i]), PreIncrement(i)])
    body = CodeBlock(i.as_Declaration(value=0), whl)
    return FunctionDefinition(void, 'our_test_function', declars, body)


def _render_compile_import(funcdef):
    code_str = render_as_source_file(funcdef, settings=dict(contract=False))
    declar = ccode(FunctionPrototype.from_FunctionDefinition(funcdef))
    return compile_link_import_strings([
        ('our_test_func.c', code_str),
        ('_our_test_func.pyx', ("cdef extern {declar}\n"
                                "def _{fname}({typ}[:] inp, {typ}[:] out):\n"
                                "    {fname}(inp.size, &inp[0], &out[0])").format(
                                    declar=declar, fname=funcdef.name, typ='double'
                                ))
    ])


def test_copying_function():
    if not np:
        skip("numpy not installed.")

    info = None
    try:
        mod, info = _render_compile_import(_mk_func1())
        inp = np.arange(10.0)
        out = np.empty_like(inp)
        mod._our_test_function(inp, out)
        assert np.allclose(inp, out)
    finally:
        if info and info['build_dir']:
            shutil.rmtree(info['build_dir'])
