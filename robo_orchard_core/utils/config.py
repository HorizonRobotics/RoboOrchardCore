# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.


"""Configuration class that extends Pydantic's model type."""

from __future__ import annotations
import ast
import importlib
import inspect
import io
import typing
from copy import deepcopy
from typing import (
    TYPE_CHECKING as _TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    Literal,
    Type,
    overload,
)

import fsspec
import rtoml as toml
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    ValidatorFunctionWrapHandler,
)
from pydantic.functional_serializers import (
    PlainSerializer,
    WrapSerializer,
    model_serializer,
)
from pydantic.functional_validators import (
    PlainValidator,
    model_validator,
)
from pydantic.version import VERSION as PYDANTIC_VERSION
from pydantic_core import PydanticCustomError, core_schema, from_json, to_json
from typing_extensions import Callable, ParamSpec, Self, TypeVar

from robo_orchard_core.utils.logging import LoggerManager
from robo_orchard_core.utils.patches import patch_class_method
from robo_orchard_core.utils.registry import Registry

if _TYPE_CHECKING:
    from robo_orchard_core.utils._config_tensor_types import (
        NumpyTensor,  # noqa: F401
        TorchTensor,  # noqa: F401
    )

logger = LoggerManager().get_child(__name__)


T = TypeVar("T")

T_co = TypeVar("T_co", covariant=True)


T_contra = TypeVar("T_contra", contravariant=True)
V = TypeVar("V")
TYPE_LIST = ParamSpec("TYPE_LIST")

PYDANTIC_CONFIGCLASS = Registry("PYDANTIC_CONFIGCLASS")

TOML_NULL = "null"


# Pydantic compatibility


def _pydantic_version_part(part: str) -> int:
    digits = []
    for char in part:
        if not char.isdigit():
            break
        digits.append(char)
    return int("".join(digits) or "0")


def _pydantic_version_at_least(major: int, minor: int) -> bool:
    release = PYDANTIC_VERSION.split("+", 1)[0].split("-", 1)[0].split(".")
    padded_release = [*release, "0", "0"]
    current = (
        _pydantic_version_part(padded_release[0]),
        _pydantic_version_part(padded_release[1]),
    )
    return current >= (major, minor)


_PYDANTIC_SUPPORTS_POLYMORPHIC_SERIALIZATION = _pydantic_version_at_least(
    2, 13
)
_TYPEVAR_DEFAULT_SCHEMA_PATCH_ENABLED = (
    not _PYDANTIC_SUPPORTS_POLYMORPHIC_SERIALIZATION
)


def _with_polymorphic_serialization(kwargs: dict[str, Any]) -> dict[str, Any]:
    if _PYDANTIC_SUPPORTS_POLYMORPHIC_SERIALIZATION:
        kwargs.setdefault("polymorphic_serialization", True)
    return kwargs


if _TYPEVAR_DEFAULT_SCHEMA_PATCH_ENABLED:
    try:
        from pydantic._internal._generate_schema import GenerateSchema
    except ImportError:
        from pydantic import GenerateSchema  # type: ignore

    @patch_class_method(GenerateSchema, "_unsubstituted_typevar_schema")
    def _wrap_unsubstituted_typevar_schema(self, typevar: TypeVar):
        """Wraps the default `_unsubstituted_typevar_schema` method.

        This patch is used to support the serialization of TypeVar with
        default values that are the same as the bound type.

        """

        assert isinstance(typevar, typing.TypeVar)

        bound = typevar.__bound__

        try:
            typevar_has_default = typevar.has_default()  # type: ignore
        except AttributeError:
            typevar_has_default = (
                getattr(typevar, "__default__", None) is not None
            )

        if (
            typevar_has_default
            and bound is not None
            and (
                typing.get_origin(bound)
                == typing.get_origin(
                    typevar.__default__,
                )
            )
        ):
            schema = self.generate_schema(bound)
            schema["serialization"] = (
                core_schema.wrap_serializer_function_ser_schema(
                    lambda x, h: h(x), schema=core_schema.any_schema()
                )
            )
            return schema

        return self.__old__unsubstituted_typevar_schema(typevar)


# Callable serialization and field annotations


def is_lambda_expression(name: str) -> bool:
    """Checks if the input string is a lambda expression.

    A copy of omni.isaac.lab.utils.config.is_lambda_expression.

    Args:
        name: The input string.

    Returns:
        Whether the input string is a lambda expression.
    """
    try:
        parsed = ast.parse(name, mode="eval")
        return isinstance(parsed.body, ast.Lambda)
    except SyntaxError:
        return False


def _validate_lambda_ast(node: ast.AST, arg_names: set[str]) -> None:
    """Validate that a lambda AST only contains side-effect free nodes."""
    if isinstance(node, ast.Lambda):
        if (
            node.args.posonlyargs
            or node.args.kwonlyargs
            or node.args.vararg is not None
            or node.args.kwarg is not None
            or node.args.defaults
            or node.args.kw_defaults
        ):
            raise ValueError(
                "Only simple lambda arguments without defaults are supported."
            )
        body_arg_names = {arg.arg for arg in node.args.args}
        _validate_lambda_ast(node.body, body_arg_names)
        return

    if isinstance(node, ast.Name):
        if node.id not in arg_names:
            raise ValueError(
                "Lambda expressions may only reference their own arguments."
            )
        return

    if isinstance(node, ast.Constant):
        return

    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        for elt in node.elts:
            _validate_lambda_ast(elt, arg_names)
        return

    if isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values, strict=True):
            if key is None:
                raise ValueError("Dictionary unpacking is not supported.")
            _validate_lambda_ast(key, arg_names)
            _validate_lambda_ast(value, arg_names)
        return

    if isinstance(node, ast.BinOp):
        if not isinstance(
            node.op,
            (
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.FloorDiv,
                ast.Mod,
                ast.Pow,
            ),
        ):
            raise ValueError(
                "Unsupported binary operator in lambda expression."
            )
        _validate_lambda_ast(node.left, arg_names)
        _validate_lambda_ast(node.right, arg_names)
        return

    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
            raise ValueError(
                "Unsupported unary operator in lambda expression."
            )
        _validate_lambda_ast(node.operand, arg_names)
        return

    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, (ast.And, ast.Or)):
            raise ValueError(
                "Unsupported boolean operator in lambda expression."
            )
        for value in node.values:
            _validate_lambda_ast(value, arg_names)
        return

    if isinstance(node, ast.Compare):
        for op in node.ops:
            if not isinstance(
                op,
                (
                    ast.Eq,
                    ast.NotEq,
                    ast.Lt,
                    ast.LtE,
                    ast.Gt,
                    ast.GtE,
                    ast.In,
                    ast.NotIn,
                    ast.Is,
                    ast.IsNot,
                ),
            ):
                raise ValueError(
                    "Unsupported comparison operator in lambda expression."
                )
        _validate_lambda_ast(node.left, arg_names)
        for comparator in node.comparators:
            _validate_lambda_ast(comparator, arg_names)
        return

    if isinstance(node, ast.IfExp):
        _validate_lambda_ast(node.test, arg_names)
        _validate_lambda_ast(node.body, arg_names)
        _validate_lambda_ast(node.orelse, arg_names)
        return

    if isinstance(node, ast.Subscript):
        _validate_lambda_ast(node.value, arg_names)
        _validate_lambda_ast(node.slice, arg_names)
        return

    if isinstance(node, ast.Slice):
        for item in (node.lower, node.upper, node.step):
            if item is not None:
                _validate_lambda_ast(item, arg_names)
        return

    raise ValueError(
        f"Unsupported expression '{type(node).__name__}' in lambda string."
    )


def safe_lambda_from_string(name: str) -> Callable:
    """Create a lambda from a restricted, side-effect free expression string.

    Args:
        name (str): The lambda expression string.

    Returns:
        Callable: The reconstructed lambda function.
    """
    parsed = ast.parse(name, mode="eval")
    if not isinstance(parsed.body, ast.Lambda):
        raise ValueError("Input is not a lambda expression.")
    _validate_lambda_ast(parsed.body, set())
    code = compile(parsed, "<config-lambda>", "eval")
    return eval(code, {"__builtins__": {}}, {})


def callable_to_string(value: Callable) -> str:
    """Converts a callable object to a string.

    A copy of omni.isaac.lab.utils.config.callable_to_string.

    Note:
        This function only works for the following types of callable objects:
        - Class type
        - Function type, should be imported from a module.
        - Lambda function, should be defined in the same file. Once lambda
        function is deserialized from string, this function will not work!

    Args:
        value: A callable object.

    Raises:
        ValueError: When the input argument is not a callable object.

    Returns:
        str: A string representation of the callable object.

    """
    # check if callable

    if not callable(value):
        raise ValueError(f"The input argument is not callable: {value}.")
    # check if lambda function
    if value.__name__ == "<lambda>":
        return f"lambda {inspect.getsourcelines(value)[0][0].strip().split('lambda')[1].strip().split(',')[0]}"  # noqa
    else:
        # get the module and function name
        module_name = value.__module__
        function_name = value.__name__
        # handle nested class
        if isinstance(value, type):
            function_name = value.__qualname__
        # return the string
        return f"{module_name}:{function_name}"


def string_to_callable(name: str) -> Callable:
    """Resolves the module and function names to return the function.

    A copy of omni.isaac.lab.utils.config.string_to_callable.

    Args:
        name: The function name. The format should be 'module:attribute_name'
            or a lambda expression of format: 'lambda x: x'.

    Note:
        This function only works for the following types of callable objects:

        - Class type
        - Function type, should be imported from a module.
        - Lambda function, should be defined in the same file. Once lambda
          function is deserialized from string, this function will
          not work!


    Raises:
        ValueError: When the resolved attribute is not a function.
        ValueError: When the module cannot be found.

    Returns:
        The function loaded from the module.

    """  # noqa: E501
    try:
        if is_lambda_expression(name):
            callable_object = safe_lambda_from_string(name)
        else:
            mod_name, attr_name = name.split(":")
            mod = importlib.import_module(mod_name)
            # handle nested class
            attr_names = attr_name.split(".")
            attr_name = attr_names[-1]
            if len(attr_names) > 1:
                for attr in attr_names[0:-1]:
                    mod = getattr(mod, attr)
            callable_object = getattr(mod, attr_name)
        # check if attribute is callable
        if callable(callable_object):
            return callable_object
        else:
            raise AttributeError(
                f"The imported object is not callable: '{name}'"
            )
    except (ValueError, ModuleNotFoundError, SyntaxError) as e:
        msg = (
            f"Could not resolve the input string '{name}' into callable object."  # noqa: E501
            " The format of input should be 'module:attribute_name'.\n"
            f"Received the error:\n {e}."
        )
        raise ValueError(msg)


_CallableSerializer = PlainSerializer(
    lambda x: callable_to_string(x) if x is not None else None,
    return_type=str,
    # when_used="always",
    when_used="json",
)

ClassType_co = Annotated[
    type[T_co],
    PlainValidator(
        lambda x: string_to_callable(x) if isinstance(x, str) else x
    ),
    _CallableSerializer,
]

ClassType = ClassType_co


CallableType = Annotated[
    Callable[TYPE_LIST, T],
    PlainValidator(
        lambda x: string_to_callable(x) if isinstance(x, str) else x
    ),
    _CallableSerializer,
]

SliceType = Annotated[
    slice,
    PlainValidator(lambda x: slice(*x["slice"]) if isinstance(x, dict) else x),
    PlainSerializer(
        lambda x: {"slice": [x.start, x.stop, x.step]}, return_type=dict
    ),
]


# Optional robotics aliases


_TENSOR_ALIAS_NAMES = frozenset({"TorchTensor", "NumpyTensor"})
_TENSOR_OPTIONAL_MODULES = frozenset({"torch", "numpy", "numpydantic"})


def __getattr__(name: str) -> Any:
    """Lazily load Tensor config aliases only when a caller requests them."""
    if name not in _TENSOR_ALIAS_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        from robo_orchard_core.utils._config_tensor_types import (
            NumpyTensor,
            TorchTensor,
        )
    except ModuleNotFoundError as error:
        module_name = (error.name or "").split(".", maxsplit=1)[0]
        if module_name not in _TENSOR_OPTIONAL_MODULES:
            raise
        raise ModuleNotFoundError(
            f"{name} requires the robotics runtime. Install "
            "'robo_orchard_core[robotics]'.",
            name=error.name,
        ) from None

    globals().update(
        {
            "TorchTensor": TorchTensor,
            "NumpyTensor": NumpyTensor,
        }
    )
    return globals()[name]


# Base configuration model


class Config(BaseModel):
    """Base model for RoboOrchard configuration objects.

    Configurations reject undeclared public fields, preserve their concrete
    type through ``__config_type__`` when requested, and serialize to JSON,
    TOML, or YAML. Subclasses may keep private runtime-only attributes; those
    attributes are not model fields and are not serialized.
    """

    __exclude_config_type__: bool = False
    """Whether this config omits ``__config_type__`` during serialization."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=(),
        extra="forbid",
    )

    # Runtime state and Pydantic hooks

    def __setattr__(self, name: str, value: Any) -> None:
        """Allow only private runtime-only attributes after construction.

        Config construction and validation remain strict via
        ``extra="forbid"``. After instantiation, some call sites attach
        ephemeral private attributes for runtime bookkeeping; keep allowing
        those without treating them as model fields.
        """
        if (
            name in type(self).model_fields
            or name.startswith("_")
            or name in getattr(self, "__dict__", {})
            or hasattr(type(self), name)
        ):
            super().__setattr__(name, value)
            return
        raise ValueError(
            f'"{type(self).__name__}" object has no field "{name}"'
        )

    @model_serializer(mode="wrap", return_type=dict, when_used="always")
    def wrapped_model_ser(
        self, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ):
        """Optionally add a concrete-type discriminator to Pydantic output.

        The discriminator lets a typed field restore a subclass rather than
        only its declared base type. It is omitted for builtins, for configs
        opting out through ``__exclude_config_type__``, and when the caller
        sets ``context['exclude_config_type']``.
        """
        if (
            (
                hasattr(self, "__exclude_config_type__")
                and self.__exclude_config_type__
            )
            or self.__class__.__module__ == "builtins"
            or (
                isinstance(info.context, dict)
                and info.context.get("exclude_config_type", True)
            )
        ):
            return handler(self)

        ret = {"__config_type__": callable_to_string(type(self))}
        ret.update(handler(self))
        return ret

    @model_validator(mode="wrap")
    @classmethod
    def wrapped_model_val(
        cls, data: Any, handler: ValidatorFunctionWrapHandler
    ):
        """Restore a discriminator-selected config class before validation."""
        if isinstance(data, str):
            data = from_json(data, allow_partial=True)
        if isinstance(data, dict):
            if "__config_type__" in data:
                data = data.copy()
                target_cls = string_to_callable(data.pop("__config_type__"))
                if target_cls == cls:
                    return handler(data)
                else:
                    return target_cls.model_validate(data)
            else:
                return handler(data)
        return data

    def __post_init__(self):
        """Compatibility lifecycle hook for configclass-style subclasses."""
        pass

    def model_post_init(self, *args, **kwargs):
        """Route Pydantic's post-init hook through ``__post_init__``."""
        self.__post_init__()

    # In-memory and text serialization

    def to_dict(
        self,
        mode: Literal["python", "json"] = "python",
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        include_config_type: bool = False,
        **kwargs,
    ) -> dict:
        """Return the configuration as a Python or JSON-compatible mapping.

        Set ``include_config_type`` when another config field must restore the
        concrete subclass rather than its declared base type. Use ``to_str``
        for JSON, TOML, or YAML persistence.

        Args:
            mode (Literal["python", "json"]): The mode of the output
                dictionary. If 'python', the output will be a Python
                dictionary. If 'json', the output will be a JSON serializable
                dictionary. Default is 'python'.
            exclude_unset (bool): Whether to exclude unset values from the
                dictionary. Default is False.
            exclude_defaults (bool): Whether to exclude default values from the
                dictionary. Default is False.
            exclude_none (bool): Whether to exclude None values from the
                dictionary. Default is False.
            include_config_type (bool): Whether to include the
                `__config_type__` key in the dictionary. If False, the
                deserialization uses the class type declared by the field,
                not the concrete serialized class.
                Default is False.

        """
        context = {
            "exclude_config_type": not include_config_type,
        }
        _with_polymorphic_serialization(kwargs)
        ret = self.model_dump(
            mode=mode,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            context=context,
            **kwargs,
        )
        return ret

    def to_str(
        self,
        format: Literal["json", "toml", "yaml"] = "json",
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        include_config_type: bool = True,
        round_trip: bool = False,
        **kwargs,
    ) -> str:
        """Serialize this configuration as JSON, TOML, or YAML.

        Unlike ``to_dict``, this method includes ``__config_type__`` by
        default so a polymorphic configuration can be reconstructed. YAML
        preserves ordinary strings, explicit trailing newlines, and true
        multiline values distinctly.

        Args:
            format (str): The format of the output string. Can be 'json',
                'yaml' or 'toml'. Default is 'json'.
            exclude_unset (bool): Whether to exclude unset values from the
                dictionary. Default is False.
            exclude_defaults (bool): Whether to exclude default values from the
                dictionary. Default is False.
            exclude_none (bool): Whether to exclude None values from the
                dictionary. Default is False.
            include_config_type (bool): Whether to include the
                `__config_type__` key in the string. If False, the
                deserialization uses the class type declared by the field,
                not the concrete serialized class.
                Default is True.
            round_trip (bool, optional): If True, the serialization will
                preserve the original data types as much as possible. This is
                useful for round-trip serialization and deserialization.
                Default is False.
            **kwargs: Additional keyword arguments to be passed to the
                serialization method :meth:`BaseModel.model_dump_json`
                and the string conversion methods of toml and yaml.

        Returns:
            str: The string representation of the configuration.

        """

        toml_kwargs = {}
        toml_kwargs["pretty"] = kwargs.pop("pretty", False)

        yaml_kwargs = {}
        yaml_kwargs["indent"] = kwargs.get("indent", None)

        json_str = self.model_dump_json(
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            context={
                "exclude_config_type": not include_config_type,
            },
            round_trip=round_trip,
            **_with_polymorphic_serialization(kwargs),
        )

        if format == "json":
            return json_str
        elif format == "toml":
            data = from_json(json_str)
            return toml.dumps(data, none_value=TOML_NULL, **toml_kwargs)
        elif format == "yaml":
            data = from_json(json_str)
            ret = yaml.dump(
                data,
                Dumper=_ConfigYamlDumper,
                sort_keys=False,
                **yaml_kwargs,
            )
            assert isinstance(ret, str)
            return ret
        else:
            raise ValueError(f"Unsupported format: {format}.")

    @classmethod
    def from_dict(cls: Type[Self], data: dict, **kwargs) -> Self:
        """Validate ``data`` as this concrete configuration class."""
        return cls.model_validate(data, **kwargs)

    @classmethod
    def from_str(
        cls: Type[Self],
        data: str,
        format: Literal["json", "toml", "yaml"] = "json",
        **kwargs,
    ) -> Self:
        """Deserialize JSON, TOML, or YAML into this concrete config class.

        Args:
            data (str): Serialized configuration data.
            format (str): The format of the input string. Can be 'json',
                'yaml' or 'toml'. Default is 'json'.
            **kwargs: Additional keyword arguments to be passed to the
                deserialization method.

        """
        if format == "json":
            return cls.model_validate_json(data, **kwargs)
        elif format == "toml":
            dict_data = toml.loads(data, none_value=TOML_NULL)
            json_str = to_json(dict_data).decode("utf-8")
            return cls.model_validate_json(json_str, **kwargs)
        elif format == "yaml":
            dict_data = yaml.load(io.StringIO(data), Loader=yaml.FullLoader)
            json_str = to_json(dict_data).decode("utf-8")
            return cls.model_validate_json(json_str, **kwargs)

    # Copying, comparison, and persistence

    def copy(self) -> Self:
        """Returns a copy of the configuration."""
        return self.model_copy()

    def replace(self, **kwargs) -> Self:
        """Return a copy with declared fields replaced.

        Unknown public field names fail eagerly instead of becoming untracked
        runtime attributes.
        """
        unknown_field_names = set(kwargs).difference(type(self).model_fields)
        if unknown_field_names:
            field_names_str = ", ".join(sorted(unknown_field_names))
            raise ValueError(
                f"Unknown field(s) for {type(self).__name__}: "
                f"{field_names_str}."
            )
        return self.model_copy(update=kwargs)

    def content_equal(self, other: Self) -> bool:
        """Compare configurations by serialized JSON content."""
        self_data = self.to_str(format="json")
        other_data = other.to_str(format="json")

        return self_data == other_data

    def __eq__(self, other: Any) -> bool:
        """Compare same-class configurations by serialized content."""
        if not isinstance(other, self.__class__):
            return False
        return self.content_equal(other)

    def save(self, path: str, indent: int = 2, **kwargs):
        """Serialize to a JSON, TOML, or YAML path through fsspec.

        The filename extension selects the format; unsupported extensions are
        rejected before a file is opened.
        """
        ext = path.split(".")[-1]
        if ext in ["toml", "json", "yaml"]:
            data_str = self.to_str(format=ext, indent=indent, **kwargs)  # type: ignore
        else:
            raise ValueError(f"Unsupported file extension: {ext}.")
        with fsspec.open(path, "w") as f:
            f.write(data_str)  # type: ignore

    @overload
    @classmethod
    def load(
        cls: type[Self],
        path: str,
        ensure_type: None = None,
    ) -> Self:
        pass

    @overload
    @classmethod
    def load(
        cls: type[Self],
        path: str,
        ensure_type: type[ConfigT],
    ) -> ConfigT:
        pass

    @classmethod
    def load(
        cls: type[Self],
        path: str,
        ensure_type: type[ConfigT] | None = None,
    ) -> Self | ConfigT:
        """Load a configuration from a JSON, TOML, or YAML file.

        A top-level ``__config_type__`` discriminator takes precedence when
        present. Without one, an explicit ``ensure_type`` is used as the
        target type; calls through a concrete subclass, such as
        ``MyConfig.load(path)``, use that subclass as the target type.

        Args:
            path (str): Configuration file path.
            ensure_type (type[ConfigT] | None, optional): Expected result type
                and fallback target when the file has no top-level type
                discriminator. Default is None.

        Returns:
            Self | ConfigT: Loaded configuration instance.
        """
        if ensure_type is not None:
            return load_from(path, ensure_type=ensure_type)
        if cls is Config:
            return load_from(path)
        return load_from(path, ensure_type=cls)


# Configured constructors


ConfigT = TypeVar("ConfigT", bound=Config)


class ClassInitFromConfigMixin:
    """Convenience marker for classes initialized with their full config.

    ``ClassConfig`` uses the ``InitFromConfig`` attribute as its dispatch
    predicate. This mixin declares that attribute as ``True`` for targets
    that receive their configuration object as the first constructor argument.
    """

    InitFromConfig: bool = True


class ClassConfig(Config, Generic[T_co]):
    """Store constructor data and build the configured class.

    Targets receive field values as keyword arguments by default. Targets
    with ``InitFromConfig=True`` receive this config object first, followed by
    caller-provided positional and non-field keyword arguments.
    ``ClassInitFromConfigMixin`` is the convenience marker for that contract.
    """

    class_type: ClassType_co[T_co]

    def __call__(self, *args, **kwargs) -> T_co:
        """Build ``class_type`` using the target's declared init convention."""

        if getattr(self.class_type, "InitFromConfig", False):
            return self.create_instance_by_cfg(*args, **kwargs)
        else:
            return self.create_instance_by_kwargs(*args, **kwargs)

    def create_instance_by_kwargs(self, *args, **kwargs) -> T_co:
        """Build the target from config fields plus caller overrides.

        Args:
            *args: Additional positional arguments to be passed to the class
                constructor.
            **kwargs: Additional keyword arguments to be passed to the class
                constructor. These will override the configuration data.

        """
        dict_data = self.to_dict()
        dict_data.pop("class_type")
        if "__config_type__" in dict_data:
            dict_data.pop("__config_type__")

        dict_data.update(kwargs)
        return self.class_type(*args, **dict_data)

    def create_instance_by_cfg(self, *args, **kwargs) -> T_co:
        """Build the target with a replaced config object as its first arg.

        Args:
            *args: Additional positional arguments to be passed to the class
                constructor after the configuration object.
            **kwargs: Additional keyword arguments to be passed to the class
                constructor. These will override the configuration data.

        """
        to_replace_kwargs = {}
        bypass_kwargs = {}
        for k, v in kwargs.items():
            if k in self.model_fields and k != "class_type":
                to_replace_kwargs[k] = v
            else:
                bypass_kwargs[k] = v

        cfg = self.replace(**to_replace_kwargs)
        return self.class_type(cfg, *args, **bypass_kwargs)  # type: ignore


class CallableConfig(Config, Generic[T_co]):
    """Store keyword arguments and invoke the configured callable."""

    func: CallableType[..., T_co]

    def __call__(self, **kwargs) -> T_co:
        """Invoke ``func`` with config fields and caller keyword overrides.

        Args:
            **kwargs: Additional keyword arguments to be passed to the
                function. These will override the configuration data.

        """
        dict_data = self.to_dict()
        dict_data.pop("func")
        if "__config_type__" in dict_data:
            dict_data.pop("__config_type__")
        dict_data.update(kwargs)
        return self.func(**dict_data)


# Config loading


def load_config_class(
    data: str | dict,
    format: Literal["json", "toml", "yaml"] = "json",
    fallback_type: type[Config] | None = None,
) -> Any:
    """Loads the configuration class from a JSON string or dictionary.

    Args:
        data (str | dict): The string data or dictionary.
        format (str): The format of the string input data. Can be 'json',
            'yaml' or 'toml'. Default is 'json'.
        fallback_type (type[Config] | None, optional): Target type used when
            the input has no top-level ``__config_type__`` discriminator.
            Default is None.
    """
    if isinstance(data, str):
        if format == "json":
            data = from_json(data, allow_partial=True)
        elif format == "toml":
            data = toml.loads(data, none_value=TOML_NULL)
        elif format == "yaml":
            data = yaml.load(io.StringIO(data), Loader=yaml.FullLoader)  # type: ignore
        else:
            raise ValueError(f"Unsupported format: {format}.")
    if isinstance(data, dict):
        if "__config_type__" in data:
            data = deepcopy(data)
            target_cls = string_to_callable(data.pop("__config_type__"))
        elif fallback_type is not None:
            target_cls = fallback_type
        else:
            raise ValueError(
                "The input data does not contain '__config_type__' key."
            )
        return target_cls.model_validate(data)
    raise ValueError("The input data is not a dictionary or string.")


@overload
def load_from(path: str, ensure_type: None = None) -> Any:
    pass


@overload
def load_from(path: str, ensure_type: type[ConfigT]) -> ConfigT:
    pass


def load_from(path: str, ensure_type: type[ConfigT] | None = None):
    """Load a configuration from a file and optionally constrain its type.

    ``ensure_type`` is used as the target type when the file has no top-level
    ``__config_type__`` discriminator. When the discriminator is present, it
    determines the concrete type and ``ensure_type`` remains a post-load type
    check.

    Args:
        path (str): JSON, TOML, or YAML configuration file path.
        ensure_type (type[ConfigT] | None, optional): Expected result type and
            fallback target for files without a top-level discriminator.
            Default is None.

    Returns:
        Any: Loaded configuration, narrowed to ``ConfigT`` when
        ``ensure_type`` is provided.
    """
    with fsspec.open(path, "r") as f:
        data = f.read()  # type: ignore
        if path.endswith(".json"):
            config_format = "json"
        elif path.endswith(".toml"):
            config_format = "toml"
        elif path.endswith(".yaml") or path.endswith(".yml"):
            config_format = "yaml"
        else:
            raise ValueError(f"Unsupported file format: {path}.")
        ret = load_config_class(
            data,
            format=config_format,
            fallback_type=ensure_type,
        )

    if ensure_type is not None and not isinstance(ret, ensure_type):
        raise TypeError(
            f"The loaded configuration is not of type {ensure_type.__name__}, "
            f"but {type(ret).__name__}."
        )

    return ret


# Config-instance annotations and YAML serialization


def add_cfg_type_ser_wrap(v: Any, nxt: SerializerFunctionWrapHandler) -> Any:
    """Wraps the serialization function to add the `__config_type__` key."""
    if isinstance(v, Config):
        ret = nxt(v)
        if "__config_type__" not in ret:
            ret["__config_type__"] = callable_to_string(type(v))
        return ret
    return nxt(v)


class _ConfigInstanceOfValidator:
    """Enforce the concrete or TypeVar-bound config type at runtime."""

    def __get_pydantic_core_schema__(
        self,
        source_type: Any,
        _handler: Any,
    ) -> core_schema.CoreSchema:
        expected_type = source_type
        if isinstance(expected_type, TypeVar):
            expected_type = expected_type.__bound__ or Config
        generic_origin = typing.get_origin(expected_type)
        pydantic_generic_metadata = getattr(
            expected_type,
            "__pydantic_generic_metadata__",
            None,
        )
        if generic_origin is None and isinstance(
            pydantic_generic_metadata, dict
        ):
            generic_origin = pydantic_generic_metadata.get("origin")
        if generic_origin is not None:
            expected_type = generic_origin

        expected_type_name = getattr(
            expected_type,
            "__name__",
            str(expected_type),
        )

        def validate(value: Any) -> Any:
            if isinstance(value, (str, dict)):
                value = load_config_class(value)
            if not isinstance(value, expected_type):
                raise PydanticCustomError(
                    "is_instance_of",
                    "Input should be an instance of {class_name}",
                    {"class_name": expected_type_name},
                )
            return value

        return core_schema.no_info_plain_validator_function(
            validate,
            metadata={
                "pydantic_js_input_core_schema": core_schema.union_schema(
                    [
                        core_schema.str_schema(),
                        core_schema.typed_dict_schema(
                            {
                                "__config_type__": (
                                    core_schema.typed_dict_field(
                                        core_schema.str_schema()
                                    )
                                )
                            },
                            extra_behavior="allow",
                        ),
                    ]
                )
            },
        )


ConfigInstanceOf = Annotated[
    ConfigT,
    _ConfigInstanceOfValidator(),
    WrapSerializer(add_cfg_type_ser_wrap, when_used="always"),
]
"""Preserve concrete config types while enforcing a declared config family.

Use ``ConfigInstanceOf[BaseConfig]`` for a field that accepts ``BaseConfig``
or any of its subclasses. Serialization preserves the concrete config's
``__config_type__`` discriminator; deserialization restores that concrete
type and then verifies it is an instance of the declared base type or TypeVar
bound.

This contract is stronger than ``SerializeAsAny`` alone. ``SerializeAsAny``
can preserve subclass fields during Pydantic serialization, but it neither
provides RoboOrchard's concrete-type discriminator nor enforces that the
restored object belongs to the field's expected config family.
"""


class _ConfigYamlDumper(yaml.SafeDumper):
    """Render configuration strings with readable YAML scalar styles."""


def _represent_config_yaml_string(
    dumper: yaml.SafeDumper,
    value: str,
) -> yaml.ScalarNode:
    """Choose YAML scalar style while preserving every newline exactly."""
    if "\n" not in value:
        style = None
    elif "\n" not in value.rstrip("\n"):
        # Make otherwise invisible trailing line feeds explicit.
        style = '"'
    else:
        # Let PyYAML select |-, |, or |+ from the trailing line feeds.
        style = "|"
    return dumper.represent_scalar(
        "tag:yaml.org,2002:str",
        value,
        style=style,
    )


_ConfigYamlDumper.add_representer(
    str,
    _represent_config_yaml_string,
)


# Preserve the historical wildcard-import surface while ensuring the two
# optional aliases are resolved through this module's lazy attribute hook.
__all__ = [
    *[name for name in globals() if not name.startswith("_")],
    "TorchTensor",
    "NumpyTensor",
]
