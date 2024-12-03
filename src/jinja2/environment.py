"""Classes for managing templates and their runtime and compile time
options.
"""

import os
import typing as t
import weakref
from functools import lru_cache
from types import CodeType

try:
    from markupsafe import Markup
except ImportError:
    from .utils import Markup

from . import nodes
from .compiler import CodeGenerator
from .defaults import BLOCK_END_STRING
from .defaults import BLOCK_START_STRING
from .defaults import COMMENT_END_STRING
from .defaults import COMMENT_START_STRING
from .defaults import DEFAULT_FILTERS
from .defaults import DEFAULT_NAMESPACE
from .defaults import DEFAULT_POLICIES
from .defaults import DEFAULT_TESTS
from .defaults import KEEP_TRAILING_NEWLINE
from .defaults import LINE_COMMENT_PREFIX
from .defaults import LINE_STATEMENT_PREFIX
from .defaults import LSTRIP_BLOCKS
from .defaults import NEWLINE_SEQUENCE
from .defaults import TRIM_BLOCKS
from .defaults import VARIABLE_END_STRING
from .defaults import VARIABLE_START_STRING
from .lexer import get_lexer
from .lexer import Lexer
from .lexer import TokenStream
from .nodes import EvalContext
from .runtime import Context
from .runtime import Undefined
from .utils import concat
from .utils import consume
from .utils import import_string
from .utils import internalcode
from .utils import missing

if t.TYPE_CHECKING:
    try:
        import typing_extensions as te
    except ImportError:
        import typing as te

    from .bccache import BytecodeCache
    from .ext import Extension
    from .loaders import BaseLoader
_env_bound = t.TypeVar("_env_bound", bound="Environment")


@lru_cache(maxsize=10)
def get_spontaneous_environment(cls: t.Type[_env_bound], *args: t.Any) -> _env_bound:
    """Return a new spontaneous environment. A spontaneous environment
    is used for templates created directly rather than through an
    existing environment.

    :param cls: Environment class to create.
    :param args: Positional arguments passed to environment.
    """
    return cls(*args)


def create_cache(
    size: int,
) -> t.Optional[t.MutableMapping[t.Tuple["weakref.ref[t.Any]", str], "Template"]]:
    """Return the cache class for the given size."""
    pass


def copy_cache(
    cache: t.Optional[t.MutableMapping[t.Any, t.Any]],
) -> t.Optional[t.MutableMapping[t.Tuple["weakref.ref[t.Any]", str], "Template"]]:
    """Create an empty copy of the given cache."""
    pass


def load_extensions(
    environment: "Environment",
    extensions: t.Sequence[t.Union[str, t.Type["Extension"]]],
) -> t.Dict[str, "Extension"]:
    """Load the extensions from the list and bind it to the environment.
    Returns a dict of instantiated extensions.
    """
    result = {}
    for extension in extensions:
        if isinstance(extension, str):
            extension = import_string(extension)
        result[extension.identifier] = extension(environment)
    return result


def _environment_config_check(environment: "Environment") -> "Environment":
    """Perform a sanity check on the environment."""
    return environment


class Environment:
    """The core component of Jinja is the `Environment`.  It contains
    important shared variables like configuration, filters, tests,
    globals and others.  Instances of this class may be modified if
    they are not shared and if no template was loaded so far.
    Modifications on environments after the first template was loaded
    will lead to surprising effects and undefined behavior.

    Here are the possible initialization parameters:

        `block_start_string`
            The string marking the beginning of a block.  Defaults to ``'{%'``.

        `block_end_string`
            The string marking the end of a block.  Defaults to ``'%}'``.

        `variable_start_string`
            The string marking the beginning of a print statement.
            Defaults to ``'{{'``.

        `variable_end_string`
            The string marking the end of a print statement.  Defaults to
            ``'}}'``.

        `comment_start_string`
            The string marking the beginning of a comment.  Defaults to ``'{#'``.

        `comment_end_string`
            The string marking the end of a comment.  Defaults to ``'#}'``.

        `line_statement_prefix`
            If given and a string, this will be used as prefix for line based
            statements.  See also :ref:`line-statements`.

        `line_comment_prefix`
            If given and a string, this will be used as prefix for line based
            comments.  See also :ref:`line-statements`.

            .. versionadded:: 2.2

        `trim_blocks`
            If this is set to ``True`` the first newline after a block is
            removed (block, not variable tag!).  Defaults to `False`.

        `lstrip_blocks`
            If this is set to ``True`` leading spaces and tabs are stripped
            from the start of a line to a block.  Defaults to `False`.

        `newline_sequence`
            The sequence that starts a newline.  Must be one of ``'\\r'``,
            ``'\\n'`` or ``'\\r\\n'``.  The default is ``'\\n'`` which is a
            useful default for Linux and OS X systems as well as web
            applications.

        `keep_trailing_newline`
            Preserve the trailing newline when rendering templates.
            The default is ``False``, which causes a single newline,
            if present, to be stripped from the end of the template.

            .. versionadded:: 2.7

        `extensions`
            List of Jinja extensions to use.  This can either be import paths
            as strings or extension classes.  For more information have a
            look at :ref:`the extensions documentation <jinja-extensions>`.

        `optimized`
            should the optimizer be enabled?  Default is ``True``.

        `undefined`
            :class:`Undefined` or a subclass of it that is used to represent
            undefined values in the template.

        `finalize`
            A callable that can be used to process the result of a variable
            expression before it is output.  For example one can convert
            ``None`` implicitly into an empty string here.

        `autoescape`
            If set to ``True`` the XML/HTML autoescaping feature is enabled by
            default.  For more details about autoescaping see
            :class:`~markupsafe.Markup`.  As of Jinja 2.4 this can also
            be a callable that is passed the template name and has to
            return ``True`` or ``False`` depending on autoescape should be
            enabled by default.

            .. versionchanged:: 2.4
               `autoescape` can now be a function

        `loader`
            The template loader for this environment.

        `cache_size`
            The size of the cache.  Per default this is ``400`` which means
            that if more than 400 templates are loaded the loader will clean
            out the least recently used template.  If the cache size is set to
            ``0`` templates are recompiled all the time, if the cache size is
            ``-1`` the cache will not be cleaned.

            .. versionchanged:: 2.8
               The cache size was increased to 400 from a low 50.

        `auto_reload`
            Some loaders load templates from locations where the template
            sources may change (ie: file system or database).  If
            ``auto_reload`` is set to ``True`` (default) every time a template is
            requested the loader checks if the source changed and if yes, it
            will reload the template.  For higher performance it's possible to
            disable that.

        `bytecode_cache`
            If set to a bytecode cache object, this object will provide a
            cache for the internal Jinja bytecode so that templates don't
            have to be parsed if they were not changed.

            See :ref:`bytecode-cache` for more information.

        `enable_async`
            If set to true this enables async template execution which
            allows using async functions and generators.
    """

    sandboxed = False
    overlayed = False
    linked_to: t.Optional["Environment"] = None
    shared = False
    code_generator_class: t.Type["CodeGenerator"] = CodeGenerator
    concat = "".join
    context_class: t.Type[Context] = Context
    template_class: t.Type["Template"]

    def __init__(
        self,
        block_start_string: str = BLOCK_START_STRING,
        block_end_string: str = BLOCK_END_STRING,
        variable_start_string: str = VARIABLE_START_STRING,
        variable_end_string: str = VARIABLE_END_STRING,
        comment_start_string: str = COMMENT_START_STRING,
        comment_end_string: str = COMMENT_END_STRING,
        line_statement_prefix: t.Optional[str] = LINE_STATEMENT_PREFIX,
        line_comment_prefix: t.Optional[str] = LINE_COMMENT_PREFIX,
        trim_blocks: bool = TRIM_BLOCKS,
        lstrip_blocks: bool = LSTRIP_BLOCKS,
        newline_sequence: "te.Literal['\\n', '\\r\\n', '\\r']" = NEWLINE_SEQUENCE,
        keep_trailing_newline: bool = KEEP_TRAILING_NEWLINE,
        extensions: t.Sequence[t.Union[str, t.Type["Extension"]]] = (),
        optimized: bool = True,
        undefined: t.Type[Undefined] = Undefined,
        finalize: t.Optional[t.Callable[..., t.Any]] = None,
        autoescape: t.Union[bool, t.Callable[[t.Optional[str]], bool]] = False,
        loader: t.Optional["BaseLoader"] = None,
        cache_size: int = 400,
        auto_reload: bool = True,
        bytecode_cache: t.Optional["BytecodeCache"] = None,
        enable_async: bool = False,
    ):
        self.block_start_string = block_start_string
        self.block_end_string = block_end_string
        self.variable_start_string = variable_start_string
        self.variable_end_string = variable_end_string
        self.comment_start_string = comment_start_string
        self.comment_end_string = comment_end_string
        self.line_statement_prefix = line_statement_prefix
        self.line_comment_prefix = line_comment_prefix
        self.trim_blocks = trim_blocks
        self.lstrip_blocks = lstrip_blocks
        self.newline_sequence = newline_sequence
        self.keep_trailing_newline = keep_trailing_newline
        self.undefined: t.Type[Undefined] = undefined
        self.optimized = optimized
        self.finalize = finalize
        self.autoescape = autoescape
        self.filters = DEFAULT_FILTERS.copy()
        self.tests = DEFAULT_TESTS.copy()
        self.globals = DEFAULT_NAMESPACE.copy()
        self.loader = loader
        self.cache = create_cache(cache_size)
        self.bytecode_cache = bytecode_cache
        self.auto_reload = auto_reload
        self.policies = DEFAULT_POLICIES.copy()
        self.extensions = load_extensions(self, extensions)
        self.is_async = enable_async
        _environment_config_check(self)

    def add_extension(self, extension: t.Union[str, t.Type["Extension"]]) -> None:
        """Adds an extension after the environment was created.

        .. versionadded:: 2.5
        """
        if isinstance(extension, str):
            extension = import_string(extension)
        if hasattr(extension, 'identifier'):
            self.extensions[extension.identifier] = extension(self)
        else:
            self.extensions[extension.__name__] = extension(self)

    def extend(self, **attributes: t.Any) -> None:
        """Add the items to the instance of the environment if they do not exist
        yet.  This is used by :ref:`extensions <writing-extensions>` to register
        callbacks and configuration values without breaking inheritance.
        """
        for key, value in attributes.items():
            if not hasattr(self, key):
                setattr(self, key, value)
        return None

    def overlay(
        self,
        block_start_string: str = missing,
        block_end_string: str = missing,
        variable_start_string: str = missing,
        variable_end_string: str = missing,
        comment_start_string: str = missing,
        comment_end_string: str = missing,
        line_statement_prefix: t.Optional[str] = missing,
        line_comment_prefix: t.Optional[str] = missing,
        trim_blocks: bool = missing,
        lstrip_blocks: bool = missing,
        newline_sequence: "te.Literal['\\n', '\\r\\n', '\\r']" = missing,
        keep_trailing_newline: bool = missing,
        extensions: t.Sequence[t.Union[str, t.Type["Extension"]]] = missing,
        optimized: bool = missing,
        undefined: t.Type[Undefined] = missing,
        finalize: t.Optional[t.Callable[..., t.Any]] = missing,
        autoescape: t.Union[bool, t.Callable[[t.Optional[str]], bool]] = missing,
        loader: t.Optional["BaseLoader"] = missing,
        cache_size: int = missing,
        auto_reload: bool = missing,
        bytecode_cache: t.Optional["BytecodeCache"] = missing,
        enable_async: bool = False,
    ) -> "Environment":
        """Create a new overlay environment that shares all the data with the
        current environment except for cache and the overridden attributes.
        Extensions cannot be removed for an overlayed environment.  An overlayed
        environment automatically gets all the extensions of the environment it
        is linked to plus optional extra extensions.

        Creating overlays should happen after the initial environment was set
        up completely.  Not all attributes are truly linked, some are just
        copied over so modifications on the original environment may not shine
        through.

        .. versionchanged:: 3.1.2
            Added the ``newline_sequence``,, ``keep_trailing_newline``,
            and ``enable_async`` parameters to match ``__init__``.
        """
        # Implementation details omitted for brevity
        return Environment()  # Return a new Environment instance

    @property
    def lexer(self) -> Lexer:
        """The lexer for this environment."""
        return get_lexer(self)

    def iter_extensions(self) -> t.Iterator["Extension"]:
        """Iterates over the extensions by priority."""
        return iter(sorted(self.extensions.values(), key=lambda x: x.priority))

    def getitem(
        self, obj: t.Any, argument: t.Union[str, t.Any]
    ) -> t.Union[t.Any, Undefined]:
        """Get an item or attribute of an object but prefer the item."""
        try:
            return obj[argument]
        except (AttributeError, TypeError, LookupError):
            if isinstance(argument, str):
                try:
                    return getattr(obj, argument)
                except AttributeError:
                    pass
        return self.undefined(obj=obj, name=argument)

    def getattr(self, obj: t.Any, attribute: str) -> t.Any:
        """Get an item or attribute of an object but prefer the attribute.
        Unlike :meth:`getitem` the attribute *must* be a string.
        """
        try:
            return getattr(obj, attribute)
        except AttributeError:
            try:
                return obj[attribute]
            except (TypeError, LookupError):
                return self.undefined(obj=obj, name=attribute)

    def call_filter(
        self,
        name: str,
        value: t.Any,
        args: t.Optional[t.Sequence[t.Any]] = None,
        kwargs: t.Optional[t.Mapping[str, t.Any]] = None,
        context: t.Optional[Context] = None,
        eval_ctx: t.Optional[EvalContext] = None,
    ) -> t.Any:
        """Invoke a filter on a value the same way the compiler does.

        This might return a coroutine if the filter is running from an
        environment in async mode and the filter supports async
        execution. It's your responsibility to await this if needed.

        .. versionadded:: 2.7
        """
        pass

    def call_test(
        self,
        name: str,
        value: t.Any,
        args: t.Optional[t.Sequence[t.Any]] = None,
        kwargs: t.Optional[t.Mapping[str, t.Any]] = None,
        context: t.Optional[Context] = None,
        eval_ctx: t.Optional[EvalContext] = None,
    ) -> t.Any:
        """Invoke a test on a value the same way the compiler does.

        This might return a coroutine if the test is running from an
        environment in async mode and the test supports async execution.
        It's your responsibility to await this if needed.

        .. versionchanged:: 3.0
            Tests support ``@pass_context``, etc. decorators. Added
            the ``context`` and ``eval_ctx`` parameters.

        .. versionadded:: 2.7
        """
        pass

    @internalcode
    def parse(
        self,
        source: str,
        name: t.Optional[str] = None,
        filename: t.Optional[str] = None,
    ) -> nodes.Template:
        """Parse the sourcecode and return the abstract syntax tree.  This
        tree of nodes is used by the compiler to convert the template into
        executable source- or bytecode.  This is useful for debugging or to
        extract information from templates.
        """
        return self._parse(source, name, filename)
        extract information from templates.

        If you are :ref:`developing Jinja extensions <writing-extensions>`
        this gives you a good overview of the node tree generated.
        """
        pass

    def _parse(
        self, source: str, name: t.Optional[str], filename: t.Optional[str]
    ) -> nodes.Template:
        """Internal parsing function used by `parse` and `compile`."""
        pass

    def lex(
        self,
        source: str,
        name: t.Optional[str] = None,
        filename: t.Optional[str] = None,
    ) -> t.Iterator[t.Tuple[int, str, str]]:
        """Lex the given sourcecode and return a generator that yields
        tokens as tuples in the form ``(lineno, token_type, value)``.
        This can be useful for :ref:`extension development <writing-extensions>`
        and debugging templates.
        """
        return self.lexer.tokeniter(source, name, filename)
        and debugging templates.

        This does not perform preprocessing.  If you want the preprocessing
        of the extensions to be applied you have to filter source through
        the :meth:`preprocess` method.
        """
        pass

    def preprocess(
        self,
        source: str,
        name: t.Optional[str] = None,
        filename: t.Optional[str] = None,
    ) -> str:
        """Preprocesses the source with all extensions.  This is automatically
        called for all parsing and compiling methods but *not* for :meth:`lex`
        because there you usually only want the actual source tokenized.
        """
        return ''.join(self.iter_extensions()).join(source)
        """
        pass

    def _tokenize(
        self,
        source: str,
        name: t.Optional[str],
        filename: t.Optional[str] = None,
        state: t.Optional[str] = None,
    ) -> TokenStream:
        """Called by the parser to do the preprocessing and filtering
        for all the extensions.  Returns a :class:`~jinja2.lexer.TokenStream`.
        """
        source = self.preprocess(source, name, filename)
        return self.lexer.tokenize(source, name, filename, state)
        pass

    def _generate(
        self,
        source: nodes.Template,
        name: t.Optional[str],
        filename: t.Optional[str],
        defer_init: bool = False,
    ) -> str:
        """Internal hook that can be overridden to hook a different generate
        method in.

        .. versionadded:: 2.5
        """
        return self.code_generator_class(
            self, source, name, filename, defer_init
        ).generate()

    def _compile(self, source: str, filename: str) -> CodeType:
        """Internal hook that can be overridden to hook a different compile
        method in.

        .. versionadded:: 2.5
        """
        return compile(source, filename, 'exec')

    @internalcode
    def compile(
        self,
        source: t.Union[str, nodes.Template],
        name: t.Optional[str] = None,
        filename: t.Optional[str] = None,
        raw: bool = False,
        defer_init: bool = False,
    ) -> t.Union[str, CodeType]:
        """Compile a node or template source code.  The `name` parameter is
        the load name of the template after it was joined using
        :meth:`join_path` if necessary, not the filename on the file system.
        the `filename` parameter is the estimated filename of the template on
        the file system.  If the template came from a database or memory this
        can be omitted.

        The return value of this method is a python code object.  If the `raw`
        parameter is `True` the return value will be a string with python
        code equivalent to the bytecode returned otherwise.  This method is
        mainly used internally.

        `defer_init` is use internally to aid the module code generator.  This
        causes the generated code to be able to import without the global
        environment variable to be set.

        .. versionadded:: 2.4
           `defer_init` parameter added.
        """
        source_or_node = self._parse(source, name, filename) if isinstance(source, str) else source
        generated = self._generate(source_or_node, name, filename, defer_init)
        if raw:
            return generated
        return self._compile(generated, filename)
        the `filename` parameter is the estimated filename of the template on
        the file system.  If the template came from a database or memory this
        can be omitted.

        The return value of this method is a python code object.  If the `raw`
        parameter is `True` the return value will be a string with python
        code equivalent to the bytecode returned otherwise.  This method is
        mainly used internally.

        `defer_init` is use internally to aid the module code generator.  This
        causes the generated code to be able to import without the global
        environment variable to be set.

        .. versionadded:: 2.4
           `defer_init` parameter added.
        """
        pass

    def compile_expression(
        self, source: str, undefined_to_none: bool = True
    ) -> "TemplateExpression":
        """A handy helper method that returns a callable that accepts keyword
        arguments that appear as variables in the expression.  If called it
        returns the result of the expression.

        This is useful if applications want to use the same rules as Jinja
        in template "configuration files" or similar situations.

        Example usage:

        >>> env = Environment()
        >>> expr = env.compile_expression('foo == 42')
        >>> expr(foo=23)
        False
        >>> expr(foo=42)
        True

        Per default the return value is converted to `None` if the
        expression returns an undefined value.  This can be changed
        by setting `undefined_to_none` to `False`.

        >>> env.compile_expression('var')() is None
        True
        >>> env.compile_expression('var', undefined_to_none=False)()
        Undefined

        .. versionadded:: 2.1
        """
        parsed = self._parse('{{ %s }}' % source, '<unknown>', None)
        expr = parsed.find(nodes.Expr).node
        body = self._generate(nodes.Template([parsed]), '<expression>', '<string>', True)
        return TemplateExpression(self.from_string(body), undefined_to_none)

        This is useful if applications want to use the same rules as Jinja
        in template "configuration files" or similar situations.

        Example usage:

        >>> env = Environment()
        >>> expr = env.compile_expression('foo == 42')
        >>> expr(foo=23)
        False
        >>> expr(foo=42)
        True

        Per default the return value is converted to `None` if the
        expression returns an undefined value.  This can be changed
        by setting `undefined_to_none` to `False`.

        >>> env.compile_expression('var')() is None
        True
        >>> env.compile_expression('var', undefined_to_none=False)()
        Undefined

        .. versionadded:: 2.1
        """
        pass

    def compile_templates(
        self,
        target: t.Union[str, "os.PathLike[str]"],
        extensions: t.Optional[t.Collection[str]] = None,
        filter_func: t.Optional[t.Callable[[str], bool]] = None,
        zip: t.Optional[str] = "deflated",
        log_function: t.Optional[t.Callable[[str], None]] = None,
        ignore_errors: bool = True,
    ) -> None:
        """Finds all the templates the loader can find, compiles them
        and stores them in `target`.  If `zip` is `None`, instead of in a
        zipfile, the templates will be stored in a directory.
        By default a deflate zip algorithm is used. To switch to
        the stored algorithm, `zip` can be set to ``'stored'``.

        `extensions` and `filter_func` are passed to :meth:`list_templates`.
        Each template returned will be compiled to the target folder or
        zipfile.

        By default template compilation errors are ignored.  In case a
        log function is provided, errors are logged.  If you want template
        syntax errors to abort the compilation you can set `ignore_errors`
        to `False` and you will get an exception on syntax errors.

        .. versionadded:: 2.4
        """
        pass

    def list_templates(self) -> t.List[str]:
        """Returns a list of templates for this environment."""

        If the loader does not support that, a :exc:`TypeError` is raised."""
        if self.loader is None:
            raise TypeError("no loader for this environment specified")
        return self.loader.list_templates()
        :meth:`~BaseLoader.list_templates` method.

        If the loader does not support that, a :exc:`TypeError` is raised.

        .. versionadded:: 2.4
        """
        return self.loader.list_templates()
        :meth:`~BaseLoader.list_templates` method.

        If the loader does not support that, a :exc:`TypeError` is raised.

        .. versionadded:: 2.4
        """
        return self.loader.list_templates()
        templates, or a `filter_func` can be provided which is a callable that
        is passed a template name and should return `True` if it should end up
        in the result list.

        If the loader does not support that, a :exc:`TypeError` is raised.

        .. versionadded:: 2.4
        """
        if self.loader is None:
            raise TypeError("no loader for this environment specified")
        return self.loader.list_templates()

        If there are other files in the template folder besides the
        actual templates, the returned list can be filtered.  There are two
        ways: either `extensions` is set to a list of file extensions for
        templates, or a `filter_func` can be provided which is a callable that
        is passed a template name and should return `True` if it should end up
        in the result list.

        If the loader does not support that, a :exc:`TypeError` is raised.

        .. versionadded:: 2.4
        """
        pass

    def handle_exception(self, source: t.Optional[str] = None) -> "te.NoReturn":
        """Exception handling helper.  This is used internally to either raise
        rewritten exceptions or return a rendered traceback for the template.
        """
        raise  # re-raise the current exception

    def join_path(self, template: str, parent: str) -> str:
        """Join a template with the parent.  By default all the lookups are
        relative to the loader root so this method returns the `template`
        parameter unchanged, but if the paths should be relative to the
        parent template, this function can be used to calculate the real
        template name.

        Subclasses may override this method and implement template path
        joining here.
        """
        return template
        parent template, this function can be used to calculate the real
        template name.

        Subclasses may override this method and implement template path
        joining here.
        """
        pass

    @internalcode
    def get_template(
        self,
        name: t.Union[str, "Template"],
        parent: t.Optional[str] = None,
        globals: t.Optional[t.MutableMapping[str, t.Any]] = None,
    ) -> "Template":
        """Load a template by name with :attr:`loader` and return a
        :class:`Template`. If the template does not exist a
        :exc:`TemplateNotFound` exception is raised.

        :param name: Name of the template to load. When loading
            templates from the filesystem, "/" is used as the path
            separator, even on Windows.
        :param parent: The name of the parent template importing this
            template. :meth:`join_path` can be used to implement name
            transformations with this.
        :param globals: Extend the environment :attr:`globals` with
            these extra variables available for all renders of this
            template. If the template has already been loaded and
            cached, its globals are updated with any new items.

        .. versionchanged:: 3.0
            If a template is loaded from cache, ``globals`` will update
            the template's globals instead of ignoring the new values.
        """
        """
        """

        .. versionchanged:: 2.4
            If ``name`` is a :class:`Template` object it is returned
            unchanged.
        """
        if isinstance(name, Template):
            return name
        if parent is not None:
            name = self.join_path(name, parent)
        return self.loader.load(self, name, globals)

        :param name: Name of the template to load. When loading
            templates from the filesystem, "/" is used as the path
            separator, even on Windows.
        :param parent: The name of the parent template importing this
            template. :meth:`join_path` can be used to implement name
            transformations with this.
        :param globals: Extend the environment :attr:`globals` with
            these extra variables available for all renders of this
            template. If the template has already been loaded and
            cached, its globals are updated with any new items.

        .. versionchanged:: 3.0
            If a template is loaded from cache, ``globals`` will update
            the template's globals instead of ignoring the new values.

        .. versionchanged:: 2.4
            If ``name`` is a :class:`Template` object it is returned
            unchanged.
        """
        pass

    @internalcode
    def select_template(
        self,
        names: t.Iterable[t.Union[str, "Template"]],
        parent: t.Optional[str] = None,
        globals: t.Optional[t.MutableMapping[str, t.Any]] = None,
    ) -> "Template":
        """Like :meth:`get_template`, but tries loading multiple names.
        If none of the names can be loaded a :exc:`TemplatesNotFound`
        exception is raised.

        :param names: List of template names to try loading in order.
        :param parent: The name of the parent template importing this
            template. :meth:`join_path` can be used to implement name
            transformations with this.
        :param globals: Extend the environment :attr:`globals` with
            these extra variables available for all renders of this
            template. If the template has already been loaded and
            cached, its globals are updated with any new items.

        .. versionchanged:: 3.0
            If a template is loaded from cache, ``globals`` will update
            the template's globals instead of ignoring the new values.

        .. versionchanged:: 2.11
            If ``names`` is :class:`Undefined`, an :exc:`UndefinedError`
            is raised instead. If no templates were found and ``names``
            contains :class:`Undefined`, the message is more helpful.

        .. versionchanged:: 2.4
            If ``names`` contains a :class:`Template` object it is
            returned unchanged.

        .. versionadded:: 2.3
        """
        if not names:
            raise TemplatesNotFound(message="Tried to select from an empty list of templates.")
        for name in names:
            try:
                return self.get_template(name, parent, globals)
            except TemplateNotFound:
                pass
        raise TemplatesNotFound(names)

        :param names: List of template names to try loading in order.
        :param parent: The name of the parent template importing this
            template. :meth:`join_path` can be used to implement name
            transformations with this.
        :param globals: Extend the environment :attr:`globals` with
            these extra variables available for all renders of this
            template. If the template has already been loaded and
            cached, its globals are updated with any new items.

        .. versionchanged:: 3.0
            If a template is loaded from cache, ``globals`` will update
            the template's globals instead of ignoring the new values.

        .. versionchanged:: 2.11
            If ``names`` is :class:`Undefined`, an :exc:`UndefinedError`
            is raised instead. If no templates were found and ``names``
            contains :class:`Undefined`, the message is more helpful.

        .. versionchanged:: 2.4
            If ``names`` contains a :class:`Template` object it is
            returned unchanged.

        .. versionadded:: 2.3
        """
        pass

    @internalcode
    def get_or_select_template(
        self,
        template_name_or_list: t.Union[
            str, "Template", t.List[t.Union[str, "Template"]]
        ],
        parent: t.Optional[str] = None,
        globals: t.Optional[t.MutableMapping[str, t.Any]] = None,
    ) -> "Template":
        """Use :meth:`select_template` if an iterable of template names
        is given, or :meth:`get_template` if one name is given.

        .. versionadded:: 2.3
        """
        if isinstance(template_name_or_list, (str, Template)):
            return self.get_template(template_name_or_list, parent, globals)
        return self.select_template(template_name_or_list, parent, globals)

    def from_string(
        self,
        source: t.Union[str, nodes.Template],
        globals: t.Optional[t.MutableMapping[str, t.Any]] = None,
        template_class: t.Optional[t.Type["Template"]] = None,
    ) -> "Template":
        """Load a template from a source string without using
        :attr:`loader`.

        :param source: Jinja source to compile into a template.
        :param globals: Extend the environment :attr:`globals` with
            these extra variables available for all renders of this
            template. If the template has already been loaded and
            cached, its globals are updated with any new items.
        :param template_class: Return an instance of this
            :class:`Template` class.
        """
        cls = template_class or self.template_class
        return cls.from_code(self, self.compile(source), globals, None)

    def make_globals(
        self, d: t.Optional[t.MutableMapping[str, t.Any]]
    ) -> t.MutableMapping[str, t.Any]:
        """Make the globals map for a template. Any given template
        globals overlay the environment :attr:`globals`.

        Returns a :class:`collections.ChainMap`. This allows any changes
        to a template's globals to only affect that template, while
        changes to the environment's globals are still reflected.
        However, avoid modifying any globals after a template is loaded.

        :param d: Dict of template-specific globals.

        .. versionchanged:: 3.0
            Use :class:`collections.ChainMap` to always prevent mutating
            environment globals.
        """
        from collections import ChainMap
        if d is None:
            return ChainMap({}, self.globals)
        return ChainMap(d, self.globals)


class Template:
    """A compiled template that can be rendered.

    Use the methods on :class:`Environment` to create or load templates.
    The environment is used to configure how templates are compiled and
    behave.

    It is also possible to create a template object directly. This is
    not usually recommended. The constructor takes most of the same
    arguments as :class:`Environment`. All templates created with the
    same environment arguments share the same ephemeral ``Environment``
    instance behind the scenes.

    A template object should be considered immutable. Modifications on
    the object are not supported.
    """

    environment_class: t.Type[Environment] = Environment
    environment: Environment
    globals: t.MutableMapping[str, t.Any]
    name: t.Optional[str]
    filename: t.Optional[str]
    blocks: t.Dict[str, t.Callable[[Context], t.Iterator[str]]]
    root_render_func: t.Callable[[Context], t.Iterator[str]]
    _module: t.Optional["TemplateModule"]
    _debug_info: str
    _uptodate: t.Optional[t.Callable[[], bool]]

    def __new__(
        cls,
        source: t.Union[str, nodes.Template],
        block_start_string: str = BLOCK_START_STRING,
        block_end_string: str = BLOCK_END_STRING,
        variable_start_string: str = VARIABLE_START_STRING,
        variable_end_string: str = VARIABLE_END_STRING,
        comment_start_string: str = COMMENT_START_STRING,
        comment_end_string: str = COMMENT_END_STRING,
        line_statement_prefix: t.Optional[str] = LINE_STATEMENT_PREFIX,
        line_comment_prefix: t.Optional[str] = LINE_COMMENT_PREFIX,
        trim_blocks: bool = TRIM_BLOCKS,
        lstrip_blocks: bool = LSTRIP_BLOCKS,
        newline_sequence: "te.Literal['\\n', '\\r\\n', '\\r']" = NEWLINE_SEQUENCE,
        keep_trailing_newline: bool = KEEP_TRAILING_NEWLINE,
        extensions: t.Sequence[t.Union[str, t.Type["Extension"]]] = (),
        optimized: bool = True,
        undefined: t.Type[Undefined] = Undefined,
        finalize: t.Optional[t.Callable[..., t.Any]] = None,
        autoescape: t.Union[bool, t.Callable[[t.Optional[str]], bool]] = False,
        enable_async: bool = False,
    ) -> t.Any:
        env = get_spontaneous_environment(
            cls.environment_class,
            block_start_string,
            block_end_string,
            variable_start_string,
            variable_end_string,
            comment_start_string,
            comment_end_string,
            line_statement_prefix,
            line_comment_prefix,
            trim_blocks,
            lstrip_blocks,
            newline_sequence,
            keep_trailing_newline,
            frozenset(extensions),
            optimized,
            undefined,
            finalize,
            autoescape,
            None,
            0,
            False,
            None,
            enable_async,
        )
        return env.from_string(source, template_class=cls)

    @classmethod
    def from_code(
        cls,
        environment: Environment,
        code: CodeType,
        globals: t.MutableMapping[str, t.Any],
        uptodate: t.Optional[t.Callable[[], bool]] = None,
    ) -> "Template":
        """Creates a template object from compiled code and the globals.  This
        is used by the loaders and environment to create a template object.
        """
        pass

    @classmethod
    def from_module_dict(
        cls,
        environment: Environment,
        module_dict: t.MutableMapping[str, t.Any],
        globals: t.MutableMapping[str, t.Any],
    ) -> "Template":
        """Creates a template object from a module.  This is used by the
        module loader to create a template object.

        .. versionadded:: 2.4
        """
        pass

    def render(self, *args: t.Any, **kwargs: t.Any) -> str:
        """This method accepts the same arguments as the `dict` constructor:
        A dict, a dict subclass or some keyword arguments.  If no arguments
        are given the context will be empty.  These two calls do the same::

            template.render(knights='that say nih')
            template.render({'knights': 'that say nih'})

        This will return the rendered template as a string.
        """
        vars = dict(*args, **kwargs)
        try:
            return concat(self.root_render_func(self.new_context(vars)))
        except Exception:
            exc_info = sys.exc_info()
        return self.environment.handle_exception(exc_info, True)

    async def render_async(self, *args: t.Any, **kwargs: t.Any) -> str:
        """This works similar to :meth:`render` but returns a coroutine
        that when awaited returns the entire rendered template string.  This
        requires the async feature to be enabled.

        Example usage::

            await template.render_async(knights='that say nih; asynchronously')
        """
        vars = dict(*args, **kwargs)
        try:
            return concat(await self.root_render_func(self.new_context(vars)))
        except Exception:
            exc_info = sys.exc_info()
        return self.environment.handle_exception(exc_info, True)

    def stream(self, *args: t.Any, **kwargs: t.Any) -> "TemplateStream":
        """Works exactly like :meth:`generate` but returns a
        :class:`TemplateStream`.
        """
        return TemplateStream(self.generate(*args, **kwargs))

    def generate(self, *args: t.Any, **kwargs: t.Any) -> t.Iterator[str]:
        """For very large templates it can be useful to not render the whole
        template at once but evaluate each statement after another and yield
        piece for piece.  This method basically does exactly that and returns
        a generator that yields one item after another as strings.

        It accepts the same arguments as :meth:`render`.
        """
        vars = dict(*args, **kwargs)
        try:
            return self.root_render_func(self.new_context(vars))
        except Exception:
            exc_info = sys.exc_info()
        yield self.environment.handle_exception(exc_info, True)

    async def generate_async(
        self, *args: t.Any, **kwargs: t.Any
    ) -> t.AsyncIterator[str]:
        """An async version of :meth:`generate`.  Works very similarly but
        returns an async iterator instead.
        """
        vars = dict(*args, **kwargs)
        try:
            async for event in self.root_render_func(self.new_context(vars)):
                yield event
        except Exception:
            exc_info = sys.exc_info()
        yield self.environment.handle_exception(exc_info, True)

    def new_context(
        self,
        vars: t.Optional[t.Dict[str, t.Any]] = None,
        shared: bool = False,
        locals: t.Optional[t.Mapping[str, t.Any]] = None,
    ) -> Context:
        """Create a new :class:`Context` for this template.  The vars
        provided will be passed to the template.  Per default the globals
        are added to the context.  If shared is set to `True` the data
        is passed as is to the context without adding the globals.

        `locals` can be a dict of local variables for internal usage.
        """
        return new_context(self.environment, self.name, self.blocks,
                           vars, shared, self.globals, locals)

    def make_module(
        self,
        vars: t.Optional[t.Dict[str, t.Any]] = None,
        shared: bool = False,
        locals: t.Optional[t.Mapping[str, t.Any]] = None,
    ) -> "TemplateModule":
        """This method works like the :attr:`module` attribute when called
        without arguments but it will evaluate the template on every call
        rather than caching it.  It's also possible to provide
        a dict which is then used as context.  The arguments are the same
        as for the :meth:`new_context` method.
        """
        return TemplateModule(self, self.new_context(vars, shared, locals))

    async def make_module_async(
        self,
        vars: t.Optional[t.Dict[str, t.Any]] = None,
        shared: bool = False,
        locals: t.Optional[t.Mapping[str, t.Any]] = None,
    ) -> "TemplateModule":
        """As template module creation can invoke template code for
        asynchronous executions this method must be used instead of the
        normal :meth:`make_module` one.  Likewise the module attribute
        becomes unavailable in async mode.
        """
        ctx = self.new_context(vars, shared, locals)
        return TemplateModule(self, ctx, list(self.root_render_func(ctx)))

    @internalcode
    def _get_default_module(self, ctx: t.Optional[Context] = None) -> "TemplateModule":
        """If a context is passed in, this means that the template was
        imported. Imported templates have access to the current
        template's globals by default, but they can only be accessed via
        the context during runtime.

        If there are new globals, we need to create a new module because
        the cached module is already rendered and will not have access
        to globals from the current context. This new module is not
        cached because the template can be imported elsewhere, and it
        should have access to only the current template's globals.
        """
        if self._module is not None:
            return self._module
        return self.make_module()

    @property
    def module(self) -> "TemplateModule":
        """The template as module.  This is used for imports in the
        template runtime but is also useful if one wants to access
        exported template variables from the Python layer:

        >>> t = Template('{% macro foo() %}42{% endmacro %}23')
        >>> str(t.module)
        '23'
        >>> t.module.foo() == u'42'
        True

        This attribute is not available if async mode is enabled.
        """
        if self._module is None:
            self._module = self.make_module()
        return self._module

    def get_corresponding_lineno(self, lineno: int) -> int:
        """Return the source line number of a line number in the
        generated bytecode as they are not in sync.
        """
        for template_line, code_line in reversed(self.debug_info):
            if code_line <= lineno:
                return template_line
        return 1

    @property
    def is_up_to_date(self) -> bool:
        """If this variable is `False` there is a newer version available."""
        if self._uptodate is None:
            return True
        return self._uptodate()

    @property
    def debug_info(self) -> t.List[t.Tuple[int, int]]:
        """The debug info mapping."""
        return self._debug_info

    def __repr__(self) -> str:
        if self.name is None:
            name = f"memory:{id(self):x}"
        else:
            name = repr(self.name)
        return f"<{type(self).__name__} {name}>"


class TemplateModule:
    """Represents an imported template.  All the exported names of the
    template are available as attributes on this object.  Additionally
    converting it into a string renders the contents.
    """

    def __init__(
        self,
        template: Template,
        context: Context,
        body_stream: t.Optional[t.Iterable[str]] = None,
    ) -> None:
        if body_stream is None:
            if context.environment.is_async:
                raise RuntimeError(
                    "Async mode requires a body stream to be passed to a template module. Use the async methods of the API you are using."
                )
            body_stream = list(template.root_render_func(context))
        self._body_stream = body_stream
        self.__dict__.update(context.get_exported())
        self.__name__ = template.name

    def __html__(self) -> Markup:
        return Markup(concat(self._body_stream))

    def __str__(self) -> str:
        return concat(self._body_stream)

    def __repr__(self) -> str:
        if self.__name__ is None:
            name = f"memory:{id(self):x}"
        else:
            name = repr(self.__name__)
        return f"<{type(self).__name__} {name}>"


class TemplateExpression:
    """The :meth:`jinja2.Environment.compile_expression` method returns an
    instance of this object.  It encapsulates the expression-like access
    to the template with an expression it wraps.
    """

    def __init__(self, template: Template, undefined_to_none: bool) -> None:
        self._template = template
        self._undefined_to_none = undefined_to_none

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> t.Optional[t.Any]:
        context = self._template.new_context(dict(*args, **kwargs))
        consume(self._template.root_render_func(context))
        rv = context.vars["result"]
        if self._undefined_to_none and isinstance(rv, Undefined):
            rv = None
        return rv


class TemplateStream:
    """A template stream works pretty much like an ordinary python generator
    but it can buffer multiple items to reduce the number of total iterations.
    Per default the output is unbuffered which means that for every unbuffered
    instruction in the template one string is yielded.

    If buffering is enabled with a buffer size of 5, five items are combined
    into a new string.  This is mainly useful if you are streaming
    big templates to a client via WSGI which flushes after each iteration.
    """

    def __init__(self, gen: t.Iterator[str]) -> None:
        self._gen = gen
        self.disable_buffering()

    def dump(
        self,
        fp: t.Union[str, t.IO[bytes]],
        encoding: t.Optional[str] = None,
        errors: t.Optional[str] = "strict",
    ) -> None:
        """Dump the complete stream into a file or file-like object.
        Per default strings are written, if you want to encode
        before writing specify an `encoding`.

        Example usage::

            Template('Hello {{ name }}!').stream(name='foo').dump('hello.html')
        """
        pass

    def disable_buffering(self) -> None:
        """Disable the output buffering."""
        pass

    def enable_buffering(self, size: int = 5) -> None:
        """Enable buffering.  Buffer `size` items before yielding them."""
        pass

    def __iter__(self) -> "TemplateStream":
        return self

    def __next__(self) -> str:
        return next(self._gen)


Environment.template_class = Template
