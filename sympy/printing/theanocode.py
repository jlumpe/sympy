from __future__ import print_function, division

from collections import MutableMapping, ChainMap

from sympy.external import import_module
from sympy.printing.printer import Printer
from sympy.core.compatibility import range, is_sequence
from sympy.utilities.exceptions import SymPyDeprecationWarning
import sympy
from functools import partial, wraps
from sympy.core.function import AppliedUndef
from sympy.core.sympify import sympify


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
            sympy.Ne: tt.neq,
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


def _sympify_shape(shape):
    """
    Convert elements of a shape tuple to Sympy objects and check that they are
    positive integers.
    """
    shape = sympify(tuple(shape))

    for s in shape:
        if s.is_integer is False:
            raise ValueError('Element of shape tuple is non-integer: %r' % s)
        if s.is_negative:
            raise ValueError('Element of shape tuple is negative: %r' % s)

    return shape


def broadcastable_from_shape(shape):
    """
    Get the ``broadcastable`` attribute for a Theano variable from the shape of
    a multidimensional array.

    Just returns a tuple with True where the size of a dimension is 1 and False
    elsewhere.

    Parameters
    ==========
    shape : tuple
        A tuple of integers or Sympy expressions representing integers.

    Returns
    =======
    tuple
        Tuple of bools with same length as ``shape``.
    """
    return tuple(s == 1 for s in shape)


def broadcastable_matches_shape(shape, broadcastable):
    """
    Check if the ``broadcastable`` attribute of a Theano variable is constistent
    with an array shape.

    Parameters
    ==========
    shape : tuple
        A tuple of integers or Sympy expressions representing integers.

    broadcastable : tuple
        Tuple of bools with same length as ``shape``.

    Returns
    =======
    bool
        False if any dimension with size != 1 is broadcastable, True otherwise.
    """
    if len(shape) != len(broadcastable):
        raise ValueError('Arguments must have the same length')

    return [not bc or s == 1 for s, bc in zip(shape, broadcastable)]





class TheanoVarCache(MutableMapping):
    """ Stores a mapping from Sympy expressions to Theano variables.

    Parameters
    ==========

    other
        TODO

    _dict
        TODO
    """

    def __init__(self, other=None, _dict=None):
        # Stores the internal representation of the mapping. Keys are tuples
        # generated from expressions, values are (expr, theano_variable) pairs.
        self._dict = {} if _dict is None else _dict
        if other is not None:
            self.update(other)

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return (symbol for symbol, var in self._dict.values())

    def __contains__(self, expr):
        key = self._get_key(expr)
        return key in self._dict

    @classmethod
    def is_cacheable(cls, obj):
        pass  # TODO

    def __getitem__(self, expr):
        key = self._get_key(expr)

        try:
            s, v = self._dict[key]
            return v
        except KeyError:
            pass

        raise KeyError(expr)

    def __delitem__(self, expr):
        key = self._get_key(expr)
        try:
            del self._dict[key]
            return
        except KeyError:
            pass

        raise KeyError(expr)

    def __setitem__(self, expr, variable):
        key = self._get_key(expr)
        self._dict[key] = (expr, variable)

    def _get_key(self, expr):
        """ Get the key for a Sympy object in _dict. """

        if not isinstance(expr, sympy.Basic):
            raise TypeError('Expected instance of sympy.Basic, not %r' % type(expr))

        if isinstance(expr, sympy.Dummy):
            # Dummies are only considered equal to themselves, dummy instances
            # should have different keys even if they have the same name.
            return (type(expr), expr)

        if isinstance(expr, (sympy.Symbol, sympy.MatrixSymbol)):
            # Symbol and MatrixSymbol discriminated only by name
            return (type(expr), expr.name)

        if isinstance(expr, AppliedUndef):
            return (type(expr), expr.args)

        raise TypeError(
            'Sympy object of type %r cannot be mapped to a Theano variable'
            % type(expr)
        )

    def copy(self):
        """ Return a copy of the cache with its own data.

        Returns
        =======
        TheanoVarCache
        """
        return TheanoVarCache(self)

    def add(self, symbol, variable, overwrite=False):
        """ Associate a Sympy symbol with an existing Theano variable.

        Functions the same as ``cache[symbol] = variable`` if ``overwrite=True``.

        Parameters:
        ===========

        symbol : sympy.core.symbol.Symbol or sympy.matrices.expressions.matexpr.MatrixSymbol
            Sympy symbol that will be replaced with the variable when printing.

        variable : theano.gof.graph.Variable
            Theano variable to associate with symbol.

        overwrite : bool
            Whether to overwrite existing associations for the symbol.
        """
        if not overwrite and symbol in self:
            raise KeyError('Symbol %r already exists in cache' % symbol)

        self[symbol] = variable


class TheanoPrinter(Printer):
    """ Code printer which creates Theano symbolic expression graphs.

    Parameters
    ==========

    cache : .TheanoVarCache or dict
        Cached mapping from Sympy objects to existing Theano variables. Defaults
        to the global cache. May pass an empty :class:`.TheanoVarCache` instance
        or empty dictionary in its place to create a printer instance which is
        independent of the global state. If a ``dict`` is given it will be used
        as the cache's internal storage, enabling reuse in other functions in
        this module, but the data in it is essentially opaque to the user. Note:
        the cache is not copied on initialization of the printer and will be
        updated in-place, so using the same cache object when creating multiple
        printers or making multiple calls to :func:`.theano_code` or
        :func:`.theano_function` means the cache is shared between all these
        applications.

    impl : dict
        Dictionary or other mapping from subclasses of
        :class:`sympy.core.basic.Basic` to their Theano implementations. Values
        must be a Theano op (instance of :class:`theano.gof.op.Op`) or any other
        callable which takes the printed versions of
        :attr:`sympy.core.basic.Basic.args` and returns a Theano variable.

    variables
        Mapping from Sympy symbols to existing Theano variables they represent.
        These are simply merged into ``cache``.

    Attributes
    ==========

    cache : dict
        A cache of Theano variables which have been created for Sympy
        symbol-like objects (e.g. :class:`sympy.core.symbol.Symbol` or
        :class:`sympy.matrices.expressions.MatrixSymbol`). This is used to
        ensure that all references to a given symbol in an expression (or
        multiple expressions) are printed as the same Theano variable, which is
        created only once. Symbols are differentiated only by name and type. The
        format of the cache's contents should be considered opaque to the user.
    """
    printmethod = "_theano"

    def __init__(self, *args, **kwargs):
        cache = kwargs.pop('cache', None)
        impl = kwargs.pop('impl', {})
        variables = kwargs.pop('variables', None)
        super(TheanoPrinter, self).__init__(*args, **kwargs)

        if isinstance(cache, TheanoVarCache):
            self.cache = cache
        else:
            self.cache = TheanoVarCache(_dict=cache)

        # Implementations of additional functions
        self.impl = {}
        for key, value in impl.items():
            if isinstance(key, str):
                key = sympy.Function(key)
            if isinstance(key, sympy.Symbol):
                key = sympy.Function(key.name)
            if not (isinstance(key, type) and issubclass(key, sympy.Basic)):
                raise TypeError('Keys of impl must be string')
            self.impl[key] = value

        self.mapping = ChainMap(self.impl, mapping)

    def _get_or_create(self, s, name=None, dtype=None, broadcastable=None):
        """
        Get the Theano variable for a Sympy symbol from the cache, or create it
        if it does not exist.
        """

        # Defaults
        if name is None:
            name = s.name
        if dtype is None:
            dtype = 'floatX'
        if broadcastable is None:
            broadcastable = ()

        if s in self.cache:
            return self.cache[s]

        value = tt.tensor(name=name, dtype=dtype, broadcastable=broadcastable)
        self.cache[s] = value
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
        try:
            op = self.mapping[type(expr)]
        except KeyError:
            raise TypeError("Don't know how to print Sympy type %r" % type(expr))

        children = [self._print(arg, **kwargs) for arg in expr.args]
        return op(*children)

    def _print_Number(self, n, **kwargs):
        # Integers already taken care of below, interpret as float
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

    def _print_MatrixElement(self, expr, **kwargs):
        parent = self._print(expr.parent, **kwargs)
        i = self._print(expr.i, **kwargs)
        j = self._print(expr.j, **kwargs)
        return parent[i, j]

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

    def doprint(self, expr, dtypes=None, broadcastables=None):
        """ Convert a Sympy expression to a Theano graph variable.

        The ``dtypes`` and ``broadcastables`` arguments are used to specify the
        data type, dimension, and broadcasting behavior of the Theano variables
        corresponding to the free symbols in ``expr``. Each is a mapping from
        Sympy symbols to the value of the corresponding argument to
        :func:`theano.tensor.Tensor`.

        See the corresponding `documentation page`__ for more information on
        broadcasting in Theano.

        .. __: http://deeplearning.net/software/theano/tutorial/broadcasting.html

        Parameters
        ==========

        expr : sympy.core.expr.Expr
            Sympy expression to print.

        dtypes : dict
            Mapping from Sympy symbols to Theano datatypes to use when creating
            new Theano variables for those symbols. Corresponds to the ``dtype``
            argument to :func:`theano.tensor.Tensor`. Defaults to ``'floatX'``
            for symbols not included in the mapping.

        broadcastables : dict
            Mapping from Sympy symbols to the value of the ``broadcastable``
            argument to :func:`theano.tensor.Tensor` to use when creating Theano
            variables for those symbols. Defaults to the empty tuple for symbols
            not included in the mapping (resulting in a scalar).

        Returns
        =======

        theano.gof.graph.Variable
            A variable corresponding to the expression's value in a Theano
            symbolic expression graph.

        See Also
        ========
        theano.tensor.Tensor
        """
        if dtypes is None:
            dtypes = {}
        if broadcastables is None:
            broadcastables = {}

        return self._print(expr, dtypes=dtypes, broadcastables=broadcastables)


global_cache = TheanoVarCache()


def theano_code(expr, cache=None, **kwargs):
    """ Convert a Sympy expression into a Theano graph variable.

    Parameters
    ==========

    expr : sympy.core.expr.Expr
        Sympy expression object to convert.

    cache : .TheanoVarCache
       Cached Theano variables (see ``cache`` argument to :class:`.TheanoPrinter`).
       Defaults to the module-level global cache.

    dtypes : dict
        Passed to :meth:`.TheanoPrinter.doprint`.

    broadcastables : dict
        Passed to :meth:`.TheanoPrinter.doprint`.

    Returns
    =======

    theano.gof.graph.Variable
        A variable corresponding to the expression's value in a Theano symbolic
        expression graph.
    """
    if not theano:
        raise ImportError("theano is required for theano_code")

    if cache is None:
        cache = global_cache

    return TheanoPrinter(cache=cache, settings={}).doprint(expr, **kwargs)


def dim_handling(inputs, dim=None, dims=None, broadcastables=None):
    """
    Get value of ``broadcastables`` argument to :func:`.theano_code` from
    keyword arguments to :func:`.theano_function`.

    Included for backwards compatibility.

    Parameters
    ==========

    inputs
        Sequence of input symbols.

    dim : int
        Common number of dimensions for all inputs. Overrides other arguments
        if given.

    dims : dict
        Mapping from input symbols to number of dimensions. Overrides
        ``broadcastables`` argument if given.

    broadcastables : dict
        Explicit value of ``broadcastables`` argument to
        :meth:`.TheanoPrinter.doprint`. If not None function will return this value unchanged.

    Returns
    =======
    dict
        Dictionary mapping elements of ``inputs`` to their "broadcastable"
        values (tuple of ``bool``s).
    """
    if dim is not None:
        return {s: (False,) * dim for s in inputs}

    if dims is not None:
        maxdim = max(dims.values())
        return {
            s: (False,) * d + (True,) * (maxdim - d)
            for s, d in dims.items()
        }

    if broadcastables != None:
        return broadcastables

    return {}


def theano_function(inputs, outputs, squeeze=True, scalar=False, **kwargs):
    """ Create a Theano function from SymPy expressions.

    The inputs and outputs are converted to Theano variables using
    :func:`.theano_code` and then passed to :func:`theano.function`.

    Parameters
    ==========

    inputs
        Sequence of symbols which constitute the inputs of the function.

    outputs
        Expression or sequence of expressions which constitute the outputs(s) of
        the function. The free symbols of each expression must be a subset of
        ``inputs``.

    squeeze : bool
        If ``outputs`` is a sequence of length one, pass the printed value of
        the lone element of the sequence to :func:`theano.function` instead of
        the sequence itself. This has the effect of making a function which
        returns a single array instead of a list containing one array. This
        behavior is deprecated and the default value will be changed to False in
        a future release. To get a function that returns a single

    scalar : bool
        Convert 0-dimensional arrays in output to scalars. This will return a
        Python wrapper function around the Theano function object.

    cache : .TheanoVarCache
       Cached Theano variables (see ``cache`` argument to :class:`.TheanoPrinter`).
       Defaults to the module-level global cache.

    dtypes : dict
        Passed to :meth:`.TheanoPrinter.doprint`.

    broadcastables : dict
        Passed to :meth:`.TheanoPrinter.doprint`.

    dims : dict
        Alternative to ``broadcastables`` argument. Mapping from elements of
        ``inputs`` to integers indicating the dimension of their associated
        arrays/tensors. Overrides ``broadcastables`` argument if given.

    dim : int
        Another alternative to the ``broadcastables`` argument. Common number of
        dimensions to use for all arrays/tensors.
        ``theano_function([x, y], [...], dim=2)`` is equivalent to using
        ``broadcastables={x: (False, False), y: (False, False)}``.

    Returns
    =======
    callable
        A callable object which takes values of ``inputs`` as positional
        arguments and returns an output array for each of the expressions
        in ``outputs``. If ``outputs`` is a single expression the function will
        return a Numpy array, if it is a list of multiple expressions the
        function will return a list of arrays. See description of the ``squeeze``
        argument above for the behavior when a single output is passed in a list.
        The returned object will either be an instance of
        :class:`theano.compile.function_module.Function` or a Python wrapper
        function around one. In both cases, the returned value will have a
        ``theano_function`` attribute which points to the return value of
        :func:`theano.function`.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.printing.theanocode import theano_function

    A simple function with one input and one output:

    >>> f1 = theano_function([x], x**2 - 1, scalar=True)
    >>> f1(3)
    8.0

    A function with multiple inputs and one output:

    >>> f2 = theano_function([x, y, z], (x**z + y**z)**(1/z), scalar=True)
    >>> f2(3, 4, 2)
    5.0

    A function with multiple inputs and multiple outputs:

    >>> f3 = theano_function([x, y], [x**2 + y**2, x**2 - y**2], scalar=True)
    >>> f3(2, 3)
    [13.0, -5.0]

    See also
    ========
    theano.function
    dim_handling
    """
    if not theano:
        raise ImportError("theano is required for theano_function")

    # Squeeze output sequence of length one
    if squeeze and is_sequence(outputs) and len(outputs) == 1:
        SymPyDeprecationWarning(
            feature='theano_function() with squeeze=True',
            issue=14986,
            deprecated_since_version='1.2.1',
            useinstead='a single expression as "outputs" (not a list)',
        ).warn()
        outputs = outputs[0]

    # Pop off non-theano keyword args
    cache = kwargs.pop('cache', {})
    dtypes = kwargs.pop('dtypes', {})

    broadcastables = dim_handling(
        inputs,
        dim=kwargs.pop('dim', None),
        dims=kwargs.pop('dims', None),
        broadcastables=kwargs.pop('broadcastables', None),
    )

    # Print inputs/outputs
    code = partial(theano_code, cache=cache, dtypes=dtypes,
                   broadcastables=broadcastables)
    tinputs  = list(map(code, inputs))

    is_seq = is_sequence(outputs)
    if is_seq:
        toutputs = list(map(code, outputs))
    else:
        toutputs = code(outputs)

    # Compile theano func
    func = theano.function(tinputs, toutputs, **kwargs)

    is_0d = [len(o.variable.broadcastable) == 0 for o in func.outputs]

    # No wrapper required
    if not scalar or not any(is_0d):
        func.theano_function = func
        return func

    # Create wrapper to convert 0-dimensional outputs to scalars
    def wrapper(*args):
        out = func(*args)

        if not is_seq:
            return out[()]

        else:
            return [o[()] if is_0d[i] else o for i, o in enumerate(out)]

    wrapper.__wrapped__ = func
    wrapper.__doc__ = func.__doc__
    wrapper.theano_function = func
    return wrapper
