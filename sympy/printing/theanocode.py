from __future__ import print_function, division
import inspect
import sys

from sympy.external import import_module

from sympy.printing.printer import Printer
from sympy.core.compatibility import range
import sympy
from functools import partial


theano = import_module('theano')

if theano:
    ts = theano.scalar
    tt = theano.tensor
    from theano.sandbox import linalg as tlinalg

    mapping = {
            sympy.Add: tt.add,
            sympy.Mul: tt.mul,
            sympy.Abs: tt.abs_,
            sympy.sign: tt.sgn,
            sympy.ceiling: tt.ceil,
            sympy.floor: tt.floor,
            sympy.log: tt.log,
            sympy.exp: tt.exp,
            sympy.sqrt: tt.sqrt,
            sympy.cos: tt.cos,
            sympy.acos: tt.arccos,
            sympy.sin: tt.sin,
            sympy.asin: tt.arcsin,
            sympy.tan: tt.tan,
            sympy.atan: tt.arctan,
            sympy.atan2: tt.arctan2,
            sympy.cosh: tt.cosh,
            sympy.acosh: tt.arccosh,
            sympy.sinh: tt.sinh,
            sympy.asinh: tt.arcsinh,
            sympy.tanh: tt.tanh,
            sympy.atanh: tt.arctanh,
            sympy.re: tt.real,
            sympy.im: tt.imag,
            sympy.arg: tt.angle,
            sympy.erf: tt.erf,
            sympy.gamma: tt.gamma,
            sympy.loggamma: tt.gammaln,
            sympy.Pow: tt.pow,
            sympy.Eq: tt.eq,
            sympy.StrictGreaterThan: tt.gt,
            sympy.StrictLessThan: tt.lt,
            sympy.LessThan: tt.le,
            sympy.GreaterThan: tt.ge,
            sympy.And: tt.and_,
            sympy.Or: tt.or_,
            sympy.Max: tt.maximum,  # Sympy accept >2 inputs, Theano only 2
            sympy.Min: tt.minimum,  # Sympy accept >2 inputs, Theano only 2
            # Matrices
            sympy.MatAdd: tt.Elemwise(ts.add),
            sympy.HadamardProduct: tt.Elemwise(ts.mul),
            sympy.Trace: tlinalg.trace,
            sympy.Determinant : tlinalg.det,
            sympy.Inverse: tlinalg.matrix_inverse,
            sympy.Transpose: tt.DimShuffle((False, False), [1, 0]),
    }


class TheanoPrinter(Printer):
    """ Code printer which creates Theano symbolic expression graphs. """
    printmethod = "_theano"

    def __init__(self, *args, **kwargs):
        self.cache = kwargs.pop('cache', dict())
        super(TheanoPrinter, self).__init__(*args, **kwargs)

    def _get_or_create(self, s, name=None, dtype=None, broadcastable=None):
        """
        Get the Theano variable for a Sympy symbol from the cache, otherwise
        create it.
        """

        # Defaults
        if name is None:
            name = s.name
        if dtype is None:
            dtype = 'floatX'
        if broadcastable is None:
            broadcastable = ()

        key = (name, type(s), s.args, dtype, broadcastable)

        if key in self.cache:
            return self.cache[key]

        value = tt.tensor(name=name, dtype=dtype, broadcastable=broadcastable)
        self.cache[key] = value
        return value

    def _print_Symbol(self, s, **kwargs):
        dtype = kwargs.get('dtypes', {}).get(s)
        bc = kwargs.get('broadcastables', {}).get(s)
        return self._get_or_create(s, dtype=dtype, broadcastable=bc)

    def _print_AppliedUndef(self, s, **kwargs):
        name = str(type(s)) + '_' + str(s.args[0])
        dtype = kwargs.get('dtypes', {}).get(s)
        bc = kwargs.get('broadcastables', {}).get(s)
        return self._get_or_create(s, name=name, dtype=dtype, broadcastable=bc)

    def _print_Basic(self, expr, **kwargs):
        op = mapping[type(expr)]
        children = [self._print(arg, **kwargs) for arg in expr.args]
        return op(*children)

    def _print_Number(self, n, **kwargs):
        return float(n.evalf())

    def _print_MatrixSymbol(self, X, **kwargs):
        dtype = kwargs.get('dtypes', {}).get(X)
        return self._get_or_create(X, dtype=dtype, broadcastable=(None, None))

    def _print_DenseMatrix(self, X, **kwargs):
        try:
            tt.stacklists
        except AttributeError:
            raise NotImplementedError(
               "Matrix translation not yet supported in this version of Theano")

        return tt.stacklists([
            [self._print(arg, **kwargs) for arg in L]
            for L in X.tolist()
        ])

    _print_ImmutableMatrix = _print_ImmutableDenseMatrix = _print_DenseMatrix

    def _print_MatMul(self, expr, **kwargs):
        children = [self._print(arg, **kwargs) for arg in expr.args]
        result = children[0]
        for child in children[1:]:
            result = tt.dot(result, child)
        return result

    def _print_MatrixSlice(self, expr, **kwargs):
        parent = self._print(expr.parent, **kwargs)
        rowslice = self._print(slice(*expr.rowslice), **kwargs)
        colslice = self._print(slice(*expr.colslice), **kwargs)
        return parent[rowslice, colslice]

    def _print_BlockMatrix(self, expr, **kwargs):
        nrows, ncols = expr.blocks.shape
        blocks = [[self._print(expr.blocks[r, c], **kwargs)
                        for c in range(ncols)]
                        for r in range(nrows)]
        return tt.join(0, *[tt.join(1, *row) for row in blocks])


    def _print_slice(self, expr, **kwargs):
        return slice(*[self._print(i, **kwargs)
                        if isinstance(i, sympy.Basic) else i
                        for i in (expr.start, expr.stop, expr.step)])

    def _print_Pi(self, expr, **kwargs):
        return 3.141592653589793

    def _print_Piecewise(self, expr, **kwargs):
        import numpy as np
        e, cond = expr.args[0].args  # First condition and corresponding value

        # Print conditional expression and value for first condition
        p_cond = self._print(cond, **kwargs)
        p_e = self._print(e, **kwargs)

        # One condition only
        if len(expr.args) == 1:
            # Return value if condition else NaN
            return tt.switch(p_cond, p_e, np.nan)

        # Return value_1 if condition_1 else evaluate remaining conditions
        p_remaining = self._print(sympy.Piecewise(*expr.args[1:]), **kwargs)
        return tt.switch(p_cond, p_e, p_remaining)

    def _print_Rational(self, expr, **kwargs):
        return tt.true_div(self._print(expr.p, **kwargs),
                           self._print(expr.q, **kwargs))

    def _print_Integer(self, expr, **kwargs):
        return expr.p

    def _print_factorial(self, expr, **kwargs):
        return self._print(sympy.gamma(expr.args[0] + 1), **kwargs)

    def _print_Derivative(self, deriv, **kwargs):
        rv = self._print(deriv.expr, **kwargs)
        for var in deriv.variables:
            var = self._print(var, **kwargs)
            rv = tt.Rop(rv, var, tt.ones_like(var))
        return rv

    def emptyPrinter(self, expr):
        return expr

    def doprint(self, expr, **kwargs):
        """Returns printer's representation for expr (as a string)"""
        return self._print(expr, **kwargs)

global_cache = {}

def theano_code(expr, cache=global_cache, **kwargs):
    if not theano:
        raise ImportError("theano is required for theano_code")
    return TheanoPrinter(cache=cache, settings={}).doprint(expr, **kwargs)


def dim_handling(inputs, dim=None, dims={}, broadcastables={}, keys=(),
        **kwargs):
    """ Handle various input types for dimensions in tensor_wrap

    See Also:
        tensor_wrap
        theano_funciton
    """
    if dim:
        dims = dict(zip(inputs, [dim]*len(inputs)))
    if dims:
        maxdim = max(dims.values())
        broadcastables = dict((i, (False,)*dims[i] + (True,)*(maxdim-dims[i]))
                         for i in inputs)
    return broadcastables


def theano_function(inputs, outputs, dtypes={}, cache=None, **kwargs):
    """ Create Theano function from SymPy expressions.

    Parameters:
    ===========

    inputs
        Sequence of symbols which constitute the inputs of the function.

    outputs
        Sequence of expressions which constitute the outputs of the function.
        These must be functions of the input symbols.

    dtypes
        Mapping from input symbols to Theano data types.
    """
    if not theano:
        raise ImportError("theano is required for theano_function")
    cache = {} if cache is None else cache
    broadcastables = dim_handling(inputs, **kwargs)

    # Remove keyword arguments corresponding to dim_handling
    if sys.version_info < (3,):
        dim_names = inspect.getargspec(dim_handling)[0]
    else:
        param = inspect.signature(dim_handling).parameters.items()
        dim_names = [n for n,p in param if p.kind == p.POSITIONAL_OR_KEYWORD]

    theano_kwargs = {k: v for k, v in kwargs.items() if k not in dim_names}

    code = partial(theano_code, cache=cache, dtypes=dtypes,
                   broadcastables=broadcastables)
    tinputs  = list(map(code, inputs))
    toutputs = list(map(code, outputs))
    toutputs = toutputs[0] if len(toutputs) == 1 else toutputs
    return theano.function(tinputs, toutputs, **theano_kwargs)
