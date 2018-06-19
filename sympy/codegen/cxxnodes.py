"""
AST nodes specific to C++.
"""

from sympy.codegen.ast import Attribute, String, AstNode, ConstructSlots, Type, none

class using(ConstructSlots, AstNode):
    """ Represents a 'using' statement in C++ """
    __slots__ = ['type', 'alias']
    defaults = {'alias': none}
    _construct_type = Type
    _construct_alias = String

constexpr = Attribute('constexpr')
