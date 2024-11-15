"""Compiles nodes from the parser into Python code."""

import typing as t
from io import StringIO

from . import nodes
from .idtracking import Symbols
from .nodes import EvalContext
from .optimizer import Optimizer
from .visitor import NodeVisitor

if t.TYPE_CHECKING:
    from typing import NoReturn

    from .environment import Environment
F = t.TypeVar("F", bound=t.Callable[..., t.Any])
operators = {
    "eq": "==",
    "ne": "!=",
    "gt": ">",
    "gteq": ">=",
    "lt": "<",
    "lteq": "<=",
    "in": "in",
    "notin": "not in",
}


def generate(
    node: nodes.Template,
    environment: "Environment",
    name: t.Optional[str],
    filename: t.Optional[str],
    stream: t.Optional[t.TextIO] = None,
    defer_init: bool = False,
    optimized: bool = True,
) -> t.Optional[str]:
    """Generate the python source for a node tree."""
    codegen = CodeGenerator(environment, name, filename, stream, defer_init, optimized)
    codegen.visit(node)
    if stream is None:
        return codegen.stream.getvalue()
    return None


def has_safe_repr(value: t.Any) -> bool:
    """Does the node have a safe representation?"""
    return isinstance(value, (bool, int, float, str, tuple, frozenset, type(None)))


def find_undeclared(
    nodes: t.Iterable[nodes.Node], names: t.Iterable[str]
) -> t.Set[str]:
    """Check if the names passed are accessed undeclared.  The return value
    is a set of all the undeclared names from the sequence of names found.
    """
    visitor = UndeclaredNameVisitor(names)
    for node in nodes:
        visitor.visit(node)
    return visitor.undeclared


class MacroRef:
    def __init__(self, node: t.Union[nodes.Macro, nodes.CallBlock]) -> None:
        self.node = node
        self.accesses_caller = False
        self.accesses_kwargs = False
        self.accesses_varargs = False


class Frame:
    """Holds compile time information for us."""

    def __init__(
        self,
        eval_ctx: EvalContext,
        parent: t.Optional["Frame"] = None,
        level: t.Optional[int] = None,
    ) -> None:
        self.eval_ctx = eval_ctx
        self.parent = parent
        if parent is None:
            self.symbols = Symbols(level=level)
            self.require_output_check = False
            self.buffer: t.Optional[str] = None
            self.block: t.Optional[str] = None
        else:
            self.symbols = Symbols(parent.symbols, level=level)
            self.require_output_check = parent.require_output_check
            self.buffer = parent.buffer
            self.block = parent.block
        self.toplevel = False
        self.rootlevel = False
        self.loop_frame = False
        self.block_frame = False
        self.soft_frame = False

    def copy(self) -> "Frame":
        """Create a copy of the current one."""
        rv = object.__new__(self.__class__)
        rv.__dict__.update(self.__dict__)
        rv.symbols = Symbols(self.symbols, level=self.symbols.level)
        return rv

    def inner(self, isolated: bool = False) -> "Frame":
        """Return an inner frame."""
        if isolated:
            return Frame(self.eval_ctx, level=self.symbols.level + 1)
        return Frame(self.eval_ctx, self, level=self.symbols.level + 1)

    def soft(self) -> "Frame":
        """Return a soft frame.  A soft frame may not be modified as
        standalone thing as it shares the resources with the frame it
        was created of, but it's not a rootlevel frame any longer.

        This is only used to implement if-statements and conditional
        expressions.
        """
        rv = object.__new__(self.__class__)
        rv.__dict__.update(self.__dict__)
        rv.rootlevel = False
        rv.soft_frame = True
        return rv

    __copy__ = copy


class VisitorExit(RuntimeError):
    """Exception used by the `UndeclaredNameVisitor` to signal a stop."""


class DependencyFinderVisitor(NodeVisitor):
    """A visitor that collects filter and test calls."""

    def __init__(self) -> None:
        self.filters: t.Set[str] = set()
        self.tests: t.Set[str] = set()

    def visit_Block(self, node: nodes.Block) -> None:
        """Stop visiting at blocks."""
        pass


class UndeclaredNameVisitor(NodeVisitor):
    """A visitor that checks if a name is accessed without being
    declared.  This is different from the frame visitor as it will
    not stop at closure frames.
    """

    def __init__(self, names: t.Iterable[str]) -> None:
        self.names = set(names)
        self.undeclared: t.Set[str] = set()

    def visit_Block(self, node: nodes.Block) -> None:
        """Stop visiting a blocks."""
        pass


class CompilerExit(Exception):
    """Raised if the compiler encountered a situation where it just
    doesn't make sense to further process the code.  Any block that
    raises such an exception is not further processed.
    """


class CodeGenerator(NodeVisitor):
    def __init__(
        self,
        environment: "Environment",
        name: t.Optional[str],
        filename: t.Optional[str],
        stream: t.Optional[t.TextIO] = None,
        defer_init: bool = False,
        optimized: bool = True,
    ) -> None:
        if stream is None:
            stream = StringIO()
        self.environment = environment
        self.name = name
        self.filename = filename
        self.stream = stream
        self.created_block_context = False
        self.defer_init = defer_init
        self.optimizer: t.Optional[Optimizer] = None
        if optimized:
            self.optimizer = Optimizer(environment)
        self.import_aliases: t.Dict[str, str] = {}
        self.blocks: t.Dict[str, nodes.Block] = {}
        self.extends_so_far = 0
        self.has_known_extends = False
        self.code_lineno = 1
        self.tests: t.Dict[str, str] = {}
        self.filters: t.Dict[str, str] = {}
        self.debug_info: t.List[t.Tuple[int, int]] = []
        self._write_debug_info: t.Optional[int] = None
        self._new_lines = 0
        self._last_line = 0
        self._first_write = True
        self._last_identifier = 0
        self._indentation = 0
        self._assign_stack: t.List[t.Set[str]] = []
        self._param_def_block: t.List[t.Set[str]] = []
        self._context_reference_stack = ["context"]

    def fail(self, msg: str, lineno: int) -> "NoReturn":
        """Fail with a :exc:`TemplateAssertionError`."""
        raise TemplateAssertionError(msg, lineno, self.name, self.filename)

    def temporary_identifier(self) -> str:
        """Get a new unique identifier."""
        self._last_identifier += 1
        return f"_tmp_{self._last_identifier}"

    def buffer(self, frame: Frame) -> None:
        """Enable buffering for the frame from that point onwards."""
        pass

    def return_buffer_contents(
        self, frame: Frame, force_unescaped: bool = False
    ) -> None:
        """Return the buffer contents of the frame."""
        pass

    def indent(self) -> None:
        """Indent by one."""
        pass

    def outdent(self, step: int = 1) -> None:
        """Outdent by step."""
        pass

    def start_write(self, frame: Frame, node: t.Optional[nodes.Node] = None) -> None:
        """Yield or write into the frame buffer."""
        pass

    def end_write(self, frame: Frame) -> None:
        """End the writing process started by `start_write`."""
        pass

    def simple_write(
        self, s: str, frame: Frame, node: t.Optional[nodes.Node] = None
    ) -> None:
        """Simple shortcut for start_write + write + end_write."""
        pass

    def blockvisit(self, nodes: t.Iterable[nodes.Node], frame: Frame) -> None:
        """Visit a list of nodes as block in a frame.  If the current frame
        is no buffer a dummy ``if 0: yield None`` is written automatically.
        """
        pass

    def write(self, x: str) -> None:
        """Write a string into the output stream."""
        pass

    def writeline(
        self, x: str, node: t.Optional[nodes.Node] = None, extra: int = 0
    ) -> None:
        """Combination of newline and write."""
        pass

    def newline(self, node: t.Optional[nodes.Node] = None, extra: int = 0) -> None:
        """Add one or more newlines before the next write."""
        pass

    def signature(
        self,
        node: t.Union[nodes.Call, nodes.Filter, nodes.Test],
        frame: Frame,
        extra_kwargs: t.Optional[t.Mapping[str, t.Any]] = None,
    ) -> None:
        """Writes a function call to the stream for the current node.
        A leading comma is added automatically.  The extra keyword
        arguments may not include python keywords otherwise a syntax
        error could occur.  The extra keyword arguments should be given
        as python dict.
        """
        pass

    def pull_dependencies(self, nodes: t.Iterable[nodes.Node]) -> None:
        """Find all filter and test names used in the template and
        assign them to variables in the compiled namespace. Checking
        that the names are registered with the environment is done when
        compiling the Filter and Test nodes. If the node is in an If or
        CondExpr node, the check is done at runtime instead.

        .. versionchanged:: 3.0
            Filters and tests in If and CondExpr nodes are checked at
            runtime instead of compile time.
        """
        pass

    def macro_body(
        self, node: t.Union[nodes.Macro, nodes.CallBlock], frame: Frame
    ) -> t.Tuple[Frame, MacroRef]:
        """Dump the function def of a macro or call block."""
        macro_frame = frame.inner()
        macro_ref = MacroRef(node)
        # ... (implement the rest of the method)
        return macro_frame, macro_ref

    def macro_def(self, macro_ref: MacroRef, frame: Frame) -> None:
        """Dump the macro definition for the def created by macro_body."""
        pass

    def position(self, node: nodes.Node) -> str:
        """Return a human readable position for the node."""
        if node.lineno is not None:
            return f"line {node.lineno}"
        return "unknown position"

    def write_commons(self) -> None:
        """Writes a common preamble that is used by root and block functions.
        Primarily this sets up common local helpers and enforces a generator
        through a dead branch.
        """
        pass

    def push_parameter_definitions(self, frame: Frame) -> None:
        """Pushes all parameter targets from the given frame into a local
        stack that permits tracking of yet to be assigned parameters.  In
        particular this enables the optimization from `visit_Name` to skip
        undefined expressions for parameters in macros as macros can reference
        otherwise unbound parameters.
        """
        pass

    def pop_parameter_definitions(self) -> None:
        """Pops the current parameter definitions set."""
        pass

    def mark_parameter_stored(self, target: str) -> None:
        """Marks a parameter in the current parameter definitions as stored.
        This will skip the enforced undefined checks.
        """
        pass

    def parameter_is_undeclared(self, target: str) -> bool:
        """Checks if a given target is an undeclared parameter."""
        if not self._param_def_block:
            return False
        return target in self._param_def_block[-1]

    def push_assign_tracking(self) -> None:
        """Pushes a new layer for assignment tracking."""
        pass

    def pop_assign_tracking(self, frame: Frame) -> None:
        """Pops the topmost level for assignment tracking and updates the
        context variables if necessary.
        """
        pass

    def visit_Block(self, node: nodes.Block, frame: Frame) -> None:
        """Call a block and register it for the template."""
        pass

    def visit_Extends(self, node: nodes.Extends, frame: Frame) -> None:
        """Calls the extender."""
        pass

    def visit_Include(self, node: nodes.Include, frame: Frame) -> None:
        """Handles includes."""
        pass

    def visit_Import(self, node: nodes.Import, frame: Frame) -> None:
        """Visit regular imports."""
        pass

    def visit_FromImport(self, node: nodes.FromImport, frame: Frame) -> None:
        """Visit named imports."""
        pass

    class _FinalizeInfo(t.NamedTuple):
        const: t.Optional[t.Callable[..., str]]
        src: t.Optional[str]

    @staticmethod
    def _default_finalize(value: t.Any) -> t.Any:
        """The default finalize function if the environment isn't
        configured with one. Or, if the environment has one, this is
        called on that function's output for constants.
        """
        pass

    _finalize: t.Optional[_FinalizeInfo] = None

    def _make_finalize(self) -> _FinalizeInfo:
        """Build the finalize function to be used on constants and at
        runtime. Cached so it's only created once for all output nodes.

        Returns a ``namedtuple`` with the following attributes:

        ``const``
            A function to finalize constant data at compile time.

        ``src``
            Source code to output around nodes to be evaluated at
            runtime.
        """
        finalize = self.environment.finalize
        const = getattr(finalize, "const", None) or self._default_finalize
        src = getattr(finalize, "src", None)
        if src:
            src = f"environment.finalize({src})"
        else:
            src = "environment.finalize"
        return self._FinalizeInfo(const=const, src=src)

    def _output_const_repr(self, group: t.Iterable[t.Any]) -> str:
        """Given a group of constant values converted from ``Output``
        child nodes, produce a string to write to the template module
        source.
        """
        return repr("".join(map(str, group)))

    def _output_child_to_const(
        self, node: nodes.Expr, frame: Frame, finalize: _FinalizeInfo
    ) -> str:
        """Try to optimize a child of an ``Output`` node by trying to
        convert it to constant, finalized data at compile time.

        If :exc:`Impossible` is raised, the node is not constant and
        will be evaluated at runtime. Any other exception will also be
        evaluated at runtime for easier debugging.
        """
        # Placeholder implementation
        return ""

    def _output_child_pre(
        self, node: nodes.Expr, frame: Frame, finalize: _FinalizeInfo
    ) -> None:
        """Output extra source code before visiting a child of an
        ``Output`` node.
        """
        pass

    def _output_child_post(
        self, node: nodes.Expr, frame: Frame, finalize: _FinalizeInfo
    ) -> None:
        """Output extra source code after visiting a child of an
        ``Output`` node.
        """
        pass


def _make_binop(op):
    def visitor(self, node, frame):
        return self.binop(node.left, node.right, op, frame)

    return visitor


def _make_unop(op):
    def visitor(self, node, frame):
        return self.unop(node.node, op, frame)

    return visitor


visit_Add = _make_binop("+")
visit_Sub = _make_binop("-")
visit_Mul = _make_binop("*")
visit_Div = _make_binop("/")
visit_FloorDiv = _make_binop("//")
visit_Pow = _make_binop("**")
visit_Mod = _make_binop("%")
visit_And = _make_binop("and")
visit_Or = _make_binop("or")
visit_Pos = _make_unop("+")
visit_Neg = _make_unop("-")
visit_Not = _make_unop("not ")
