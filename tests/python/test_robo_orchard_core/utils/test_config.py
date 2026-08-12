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


import functools
from typing import Generic, Literal

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from typing_extensions import TypeVar

from robo_orchard_core.utils import config as config_module
from robo_orchard_core.utils.config import (
    CallableConfig,
    CallableType,
    ClassConfig,
    ClassInitFromConfigMixin,
    ClassType,
    Config,
    ConfigInstanceOf,
    NumpyTensor,
    TorchTensor,
    string_to_callable,
)


def f1() -> int:
    return 1


def plus_10(v: int) -> int:
    return v + 10


def dummy_decorator_with_wraps(func):
    @functools.wraps(func)
    def wrapper():
        return func() + 10

    return wrapper


def dummy_decorator(func):
    def wrapper():
        return func() + 20

    return wrapper


@dummy_decorator
def f1_decorated() -> int:
    return 1


@dummy_decorator_with_wraps
def f1_decorated_with_wraps() -> int:
    return 1


class DummyConfig(Config):
    int_value: int = 100


class ExtendedDummyConfig(DummyConfig):
    string_value: str = "child"


class YamlStringConfig(Config):
    value: str


class DummyClassConfig(DummyConfig, ClassConfig[DummyConfig]):
    class_type: ClassType[DummyConfig] = DummyConfig


class DummyConfigInstanceHolder(Config):
    cfg: ConfigInstanceOf[DummyConfig]


class DummyClassConfigInstanceHolder(Config):
    cfg: ConfigInstanceOf[ClassConfig[DummyConfig]]


DummyConfigType = TypeVar("DummyConfigType", bound=DummyConfig)


class GenericDummyConfigInstanceHolder(Config, Generic[DummyConfigType]):
    cfg: ConfigInstanceOf[DummyConfigType]


class DummyConfig2(Config):
    cfg1: DummyClassConfig
    cfg2: DummyClassConfig


class DummyClassConfig2(DummyConfig2, ClassConfig[DummyConfig2]):
    class_type: ClassType[DummyConfig2] = DummyConfig2


class DummyCallableConfig(CallableConfig[int]):
    func: CallableType = f1


class Plus10CallableConfig(CallableConfig[int]):
    func: CallableType = plus_10
    v: int = 10


class DummyConfigInitializerMeta(ClassInitFromConfigMixin):
    def __init__(self, cfg: "DummyConfigInitializerMetaCfg"):
        self.cfg = cfg
        self.init_value = cfg.int_value

    def __str__(self):
        return f"DummyConfigInitializerMeta({self.init_value})"


class DummyConfigInitializerMetaCfg(
    DummyConfig, ClassConfig[DummyConfigInitializerMeta]
):
    class_type: ClassType[DummyConfigInitializerMeta] = (
        DummyConfigInitializerMeta
    )


class TypeVarDefaultBaseConfig(Config):
    base_value: int = 100


class TypeVarDefaultChildConfig(TypeVarDefaultBaseConfig):
    child_value: int = 200


TypeVarDefaultConfigType_co = TypeVar(
    "TypeVarDefaultConfigType_co",
    bound=TypeVarDefaultBaseConfig,
    covariant=True,
    default=TypeVarDefaultBaseConfig,
)


class TypeVarDefaultHolderConfig(
    Config, Generic[TypeVarDefaultConfigType_co]
):
    cfg: TypeVarDefaultConfigType_co


class TestSimpleConfig:
    def test_simple_config(self):
        config = DummyConfig()
        assert config.int_value == 100
        config.int_value = 200
        assert config.int_value == 200

    def test_constructor_rejects_unknown_fields(self):
        with pytest.raises(ValidationError, match="unknown_value"):
            DummyConfig.model_validate({"unknown_value": 1})

    def test_private_dynamic_attribute_assignment_is_allowed(self):
        config = DummyConfig()
        private_name = "_runtime_only_value"

        setattr(config, private_name, 123)

        assert getattr(config, private_name) == 123
        assert config.to_dict() == {"int_value": 100}

    def test_public_dynamic_attribute_assignment_is_rejected(self):
        config = DummyConfig()
        public_name = "runtime_only_value"

        with pytest.raises(ValueError, match="object has no field"):
            setattr(config, public_name, 123)

    def test_replace_rejects_unknown_fields(self):
        config = DummyConfig()

        with pytest.raises(ValueError, match="unknown_value"):
            config.replace(unknown_value=1)

    def test_json_dump(self):
        config = DummyConfig()
        config.int_value = 200
        assert (
            config.to_str(format="json")
            == '{"__config_type__":"test_config:DummyConfig","int_value":200}'
        )

    def test_from_json(self):
        config = DummyConfig.from_str('{"int_value":200}', format="json")
        assert config.int_value == 200

    def test_dict_dump(self):
        config = DummyConfig()
        config.int_value = 200
        assert config.to_dict() == {
            "int_value": 200,
        }

    def test_dict_dump_with_type(self):
        config = DummyConfig()
        config.int_value = 200
        assert config.to_dict(include_config_type=True) == {
            "__config_type__": "test_config:DummyConfig",
            "int_value": 200,
        }

    def test_from_dict(self):
        config = DummyConfig.from_dict({"int_value": 200})
        assert config.int_value == 200


class TestSimpleClassConfig:
    def test_to_dict_python(self):
        config = DummyClassConfig()
        config.int_value = 200
        assert config.to_dict() == {
            # "class_type": "test_config:DummyConfig",
            "class_type": DummyConfig,
            "int_value": 200,
        }

    def test_class_config_json_serialization(self):
        config = DummyClassConfig()
        config.int_value = 200
        assert callable(config.class_type)
        json_str = config.to_str(format="json")

        new_config = DummyClassConfig.from_str(json_str, format="json")
        assert new_config.int_value == config.int_value
        assert new_config.class_type == config.class_type

        # make sure that class_type is still callable
        assert callable(new_config.class_type)

    def test_class_config_create_instance(self):
        config = DummyClassConfig()
        config.int_value = 200
        instance = config()
        assert instance.int_value == 200
        assert isinstance(instance, DummyConfig)

    def test_class_config_create_instance_with_override(self):
        config = DummyClassConfig()
        config.int_value = 200
        instance = config(int_value=300)
        assert instance.int_value == 300
        assert isinstance(instance, DummyConfig)

    def test_class_config_call_with_ConfigAsArgInitMeta(self):
        config = DummyConfigInitializerMetaCfg()
        config.int_value = 200
        instance = config()
        assert instance.init_value == 200
        assert isinstance(instance, DummyConfigInitializerMeta)


class TestConfigInstanceOf:
    @pytest.mark.parametrize(
        "model_type",
        [
            DummyConfigInstanceHolder,
            DummyClassConfigInstanceHolder,
            GenericDummyConfigInstanceHolder,
        ],
    )
    def test_model_json_schema_supports_config_instance_of(
        self,
        model_type: type[Config],
    ) -> None:
        schema = model_type.model_json_schema()

        assert "cfg" in schema["properties"]

    def test_model_json_schema_matches_config_instance_input_contract(
        self,
    ) -> None:
        schema = DummyConfigInstanceHolder.model_json_schema()
        cfg_schema = schema["properties"]["cfg"]
        serialized_config_schema = cfg_schema["anyOf"][1]

        assert cfg_schema["anyOf"][0] == {"type": "string"}
        assert serialized_config_schema["type"] == "object"
        assert serialized_config_schema["required"] == ["__config_type__"]
        assert (
            serialized_config_schema["properties"]["__config_type__"][
                "type"
            ]
            == "string"
        )

        serialized_cfg = DummyConfig(int_value=200).to_dict(
            include_config_type=True
        )
        holder = DummyConfigInstanceHolder.model_validate(
            {"cfg": serialized_cfg}
        )

        assert holder.cfg == DummyConfig(int_value=200)

        with pytest.raises(ValidationError, match="__config_type__"):
            DummyConfigInstanceHolder.model_validate(
                {"cfg": {"int_value": 200}}
            )

    @pytest.mark.parametrize("format", ["json", "yaml"])
    def test_round_trip_preserves_valid_subclass(
        self,
        format: Literal["json", "yaml"],
    ) -> None:
        holder = DummyConfigInstanceHolder(
            cfg=ExtendedDummyConfig(
                int_value=200,
                string_value="preserved",
            )
        )

        restored = DummyConfigInstanceHolder.from_str(
            holder.to_str(format=format),
            format=format,
        )

        assert isinstance(restored.cfg, ExtendedDummyConfig)
        assert restored.cfg.int_value == 200
        assert restored.cfg.string_value == "preserved"

    def test_rejects_unrelated_config_instance(self) -> None:
        with pytest.raises(ValidationError, match="DummyConfig"):
            DummyConfigInstanceHolder(
                cfg=YamlStringConfig(  # type: ignore[arg-type]
                    value="not a DummyConfig"
                )
            )

    def test_rejects_deserialized_unrelated_config_type(self) -> None:
        unrelated_config = YamlStringConfig(value="not a DummyConfig")

        with pytest.raises(ValidationError, match="DummyConfig"):
            DummyConfigInstanceHolder.model_validate(
                {"cfg": unrelated_config.to_dict(include_config_type=True)}
            )

    def test_unresolved_typevar_uses_its_runtime_bound(self) -> None:
        holder = GenericDummyConfigInstanceHolder(
            cfg=DummyClassConfig(int_value=200)
        )

        assert isinstance(holder.cfg, DummyClassConfig)

        with pytest.raises(ValidationError, match="DummyConfig"):
            GenericDummyConfigInstanceHolder(
                cfg=YamlStringConfig(  # type: ignore[arg-type]
                    value="not a DummyConfig"
                )
            )

    def test_parameterized_generic_uses_its_runtime_origin(self) -> None:
        holder = DummyClassConfigInstanceHolder(
            cfg=DummyClassConfig(int_value=200)
        )

        assert isinstance(holder.cfg, DummyClassConfig)


class TestCallableConfig:
    def test_callable_config(self):
        config = DummyCallableConfig()
        assert config.func() == 1

    def test_callable_config_json_serialization(self):
        config = DummyCallableConfig()
        json_str = config.to_str(format="json")
        new_config = DummyCallableConfig.from_str(json_str, format="json")
        assert new_config.func() == 1

    def test_callable_config_call(self):
        config = Plus10CallableConfig()
        assert config() == 20

    def test_callable_config_call_with_override(self):
        config = Plus10CallableConfig()
        assert config(v=20) == 30

    def test_callable_config_call_with_decorator(self):
        config = DummyCallableConfig()
        config.func = f1_decorated
        assert config() == 21

    def test_callable_config_call_with_decorator_with_wraps(self):
        config = DummyCallableConfig()
        config.func = f1_decorated_with_wraps
        assert config() == 11

    def test_string_to_callable_supports_safe_lambda(self):
        func = string_to_callable("lambda x: x + 1")
        assert func(2) == 3

    def test_string_to_callable_rejects_lambda_with_side_effects(self):
        with pytest.raises(ValueError, match="Could not resolve"):
            string_to_callable(
                "lambda x=__import__('os').system("
                "'echo hi >/tmp/forbidden'"
                "): x"
            )


class TestCascadeClassConfig:
    def test_cascade_class_config(self):
        config = DummyClassConfig2(
            cfg1=DummyClassConfig(int_value=200),
            cfg2=DummyClassConfig(int_value=300),
        )
        assert config.to_dict() == {
            # "class_type": "test_config:DummyConfig2",
            "class_type": DummyConfig2,
            "cfg1": {
                # "class_type": "test_config:DummyConfig",
                "class_type": DummyConfig,
                "int_value": 200,
            },
            "cfg2": {
                # "class_type": "test_config:DummyConfig",
                "class_type": DummyConfig,
                "int_value": 300,
            },
        }

    def test_cascade_class_config_json_serialization(self):
        config = DummyClassConfig2(
            cfg1=DummyClassConfig(int_value=200),
            cfg2=DummyClassConfig(int_value=300),
        )
        json_str = config.to_str(format="json")
        new_config = DummyClassConfig2.from_str(json_str, format="json")
        assert new_config.to_dict() == config.to_dict()

    def test_cascade_class_config_create_instance(self):
        config = DummyClassConfig2(
            cfg1=DummyClassConfig(int_value=200),
            cfg2=DummyClassConfig(int_value=300),
        )
        instance = config.create_instance_by_kwargs()
        assert isinstance(instance, DummyConfig2)
        assert instance.to_dict() == {
            "cfg1": {
                # "class_type": "test_config:DummyConfig",
                "class_type": DummyConfig,
                "int_value": 200,
            },
            "cfg2": {
                # "class_type": "test_config:DummyConfig",
                "class_type": DummyConfig,
                "int_value": 300,
            },
        }


class TensorConfig(Config):
    np_tensor: NumpyTensor | None = None
    torch_tensor: TorchTensor | None = None


class TestTensorConfig:
    def test_model_json_schema_supports_numpy_tensor(self):
        schema = TensorConfig.model_json_schema()

        assert "np_tensor" in schema["properties"]

    def test_np_tensor(self):
        np_tensor = np.array([[1, 2.22], [3, 4]])
        config = TensorConfig(np_tensor=np_tensor, torch_tensor=None)
        assert isinstance(config.np_tensor, np.ndarray)

        json_str = config.to_str(format="json")
        new_config = TensorConfig.from_str(json_str, format="json")
        print("json_str:", json_str)
        assert isinstance(new_config.np_tensor, np.ndarray)
        assert np.array_equal(new_config.np_tensor, np_tensor)

    def test_torch_tensor(self):
        torch_tensor = torch.tensor([[1, 2.22], [3, 4]])
        config = TensorConfig(np_tensor=None, torch_tensor=torch_tensor)
        assert isinstance(config.torch_tensor, torch.Tensor)

        json_str = config.to_str(format="json")
        new_config = TensorConfig.from_str(json_str, format="json")
        print("json_str:", json_str)
        assert isinstance(new_config.torch_tensor, torch.Tensor)
        assert torch.equal(new_config.torch_tensor, torch_tensor)


class TestTypeVarDefaultConfig:
    def test_pydantic_213_uses_polymorphic_serialization_not_patch(self):
        if not config_module._pydantic_version_at_least(2, 13):
            pytest.skip("Pydantic < 2.13 needs the TypeVar schema patch.")

        assert config_module._PYDANTIC_SUPPORTS_POLYMORPHIC_SERIALIZATION
        assert not config_module._TYPEVAR_DEFAULT_SCHEMA_PATCH_ENABLED

    def test_typevar_default_serializes_runtime_subclass_fields(self):
        config = TypeVarDefaultHolderConfig(
            cfg=TypeVarDefaultChildConfig(base_value=10, child_value=20)
        )

        json_str = config.to_str(format="json")
        new_config = TypeVarDefaultHolderConfig.from_str(
            json_str, format="json"
        )

        assert config.to_dict() == {
            "cfg": {"base_value": 10, "child_value": 20}
        }
        assert isinstance(new_config.cfg, TypeVarDefaultChildConfig)
        assert new_config.to_dict() == config.to_dict()

    def test_explicit_polymorphic_serialization_false_is_respected(self):
        if not config_module._PYDANTIC_SUPPORTS_POLYMORPHIC_SERIALIZATION:
            pytest.skip("Pydantic < 2.13 does not support this dump option.")

        config = TypeVarDefaultHolderConfig(
            cfg=TypeVarDefaultChildConfig(base_value=10, child_value=20)
        )

        assert config.to_dict(polymorphic_serialization=False) == {
            "cfg": {"base_value": 10}
        }


class TestConfigSaveLoad:
    @pytest.mark.parametrize(
        ("value", "expected_yaml"),
        [
            ("plain text", "value: plain text"),
            ("<|start|>agent\n", 'value: "<|start|>agent\\n"'),
            ("<|start|>agent\n\n", 'value: "<|start|>agent\\n\\n"'),
            (
                "first line\n\nsecond line",
                "value: |-\n  first line\n\n  second line",
            ),
            (
                "first line\nsecond line\n",
                "value: |\n  first line\n  second line",
            ),
            (
                "first line\nsecond line\n\n",
                "value: |+\n  first line\n  second line",
            ),
        ],
    )
    def test_yaml_string_styles_round_trip(
        self,
        value: str,
        expected_yaml: str,
    ):
        cfg = YamlStringConfig(value=value)

        yaml_str = cfg.to_str(format="yaml")
        restored = YamlStringConfig.from_str(yaml_str, format="yaml")

        assert expected_yaml in yaml_str
        assert restored == cfg
        assert restored.value == value

    @pytest.mark.parametrize(
        "value",
        [
            "plain text",
            "<|start|>agent\n",
            "<|start|>agent\n\n",
            "first line\n\nsecond line",
            "first line\nsecond line\n",
            "first line\nsecond line\n\n",
        ],
    )
    def test_yaml_string_styles_save_and_load(
        self,
        tmp_path,
        value: str,
    ):
        cfg = YamlStringConfig(value=value)
        path = tmp_path / "yaml_string_cfg.yaml"

        cfg.save(str(path))
        restored = Config.load(str(path), ensure_type=YamlStringConfig)

        assert restored == cfg
        assert restored.value == value

    def test_to_str_with_yaml_indent(self):
        cfg = DummyClassConfig2(
            cfg1=DummyClassConfig(int_value=200),
            cfg2=DummyClassConfig(int_value=300),
        )
        yaml_str = cfg.to_str(format="yaml", indent=4)
        assert "    int_value: 200" in yaml_str
        assert "    int_value: 300" in yaml_str

    def test_to_str_with_toml_pretty(self):
        cfg = DummyConfig(int_value=321)
        toml_str = cfg.to_str(format="toml", pretty=True)
        loaded = DummyConfig.from_str(toml_str, format="toml")
        assert loaded == cfg

    @pytest.mark.parametrize("ext", ["json", "toml", "yaml"])
    def test_save_and_load(self, tmp_path, ext):
        cfg = DummyConfig(int_value=4321)
        path = tmp_path / f"dummy_cfg.{ext}"
        cfg.save(str(path))

        loaded = Config.load(str(path), ensure_type=DummyConfig)
        assert isinstance(loaded, DummyConfig)
        assert loaded == cfg

    @pytest.mark.parametrize("ext", ["json", "toml", "yaml"])
    def test_load_without_config_type_uses_ensure_type(self, tmp_path, ext):
        """An explicit expected type loads a headerless config file."""
        cfg = DummyConfig(int_value=4321)
        path = tmp_path / f"dummy_cfg.{ext}"
        cfg.save(str(path), include_config_type=False)

        loaded = Config.load(str(path), ensure_type=DummyConfig)

        assert type(loaded) is DummyConfig
        assert loaded == cfg

    @pytest.mark.parametrize("ext", ["json", "toml", "yaml"])
    def test_concrete_config_loads_without_config_type(self, tmp_path, ext):
        """A concrete class load uses that class for headerless files."""
        cfg = DummyConfig(int_value=4321)
        path = tmp_path / f"dummy_cfg.{ext}"
        cfg.save(str(path), include_config_type=False)

        loaded = DummyConfig.load(str(path))

        assert type(loaded) is DummyConfig
        assert loaded == cfg

    def test_base_config_load_without_type_information_fails(self, tmp_path):
        """The base loader cannot infer a type from a headerless file."""
        path = tmp_path / "dummy_cfg.json"
        DummyConfig(int_value=4321).save(
            str(path),
            include_config_type=False,
        )

        with pytest.raises(ValueError, match="__config_type__"):
            Config.load(str(path))

    def test_file_config_type_takes_precedence_for_subclass_load(
        self,
        tmp_path,
    ):
        """A file discriminator preserves a compatible runtime subclass."""
        cfg = ExtendedDummyConfig(
            int_value=4321,
            string_value="preserved",
        )
        path = tmp_path / "extended_dummy_cfg.yaml"
        cfg.save(str(path))

        loaded = DummyConfig.load(str(path))

        assert type(loaded) is ExtendedDummyConfig
        assert loaded == cfg

    def test_subclass_load_rejects_unrelated_file_type(self, tmp_path):
        """A concrete loader rejects an incompatible file discriminator."""
        path = tmp_path / "yaml_string_cfg.yaml"
        YamlStringConfig(value="unrelated").save(str(path))

        with pytest.raises(TypeError, match="not of type DummyConfig"):
            DummyConfig.load(str(path))

    def test_save_with_unsupported_ext(self, tmp_path):
        cfg = DummyConfig(int_value=1)
        with pytest.raises(ValueError):
            cfg.save(str(tmp_path / "dummy_cfg.txt"))


if __name__ == "__main__":
    pytest.main(["-s", "test_config.py"])
