# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
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

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import rtoml

PROJECT_ROOT = Path(__file__).parents[3]


class TestSlimProfile:
    def test_metadata_separates_base_and_optional_runtime_closures(self):
        """Default stays light; official extras stay complete."""
        pyproject = rtoml.load(PROJECT_ROOT / "pyproject.toml")
        dependencies = set(pyproject["project"]["dependencies"])
        extras = pyproject["project"]["optional-dependencies"]

        assert {
            "rtoml",
            "pyyaml",
            "fsspec",
            "pydantic>=2.9,<3",
            "pydantic-core",
            "pydantic-settings>=2.9.1",
            "ordered-set",
            "typing-extensions",
            "typer",
        } <= dependencies
        assert (
            not {
                "numpy",
                "numpydantic>=1.8.1",
                "torch",
                "gymnasium",
                "pillow",
                "opencv-python>=4",
            }
            & dependencies
        )
        assert extras["kinematic"] == ["robo_orchard_core[robotics]"]
        assert {
            "numpy",
            "numpydantic>=1.8.1",
            "torch",
            "pillow",
            "opencv-python>=4",
            "gymnasium",
            "pytorch_kinematics",
        } <= set(extras["robotics"])
        assert "pydantic-settings>=2.9.1" not in extras["tools"]
        assert extras["ray"] == ["robo_orchard_core[robotics]", "ray"]
        assert extras["all"] == [
            "robo_orchard_core[robotics,tools,ray,ipy_viz,virtual_desktop]"
        ]

    def test_slim_imports_do_not_load_or_require_robotics_runtime(self):
        """Base imports and root help work when robotics modules are absent."""
        script = textwrap.dedent(
            """
            import importlib.abc
            import sys

            BLOCKED = {
                "torch",
                "numpy",
                "numpydantic",
                "gymnasium",
                "PIL",
                "cv2",
                "pytorch_kinematics",
            }

            class BlockOptionalRuntime(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    root = fullname.split(".", maxsplit=1)[0]
                    if root in BLOCKED:
                        raise ModuleNotFoundError(
                            f"blocked optional runtime module: {fullname}",
                            name=root,
                        )
                    return None

            sys.meta_path.insert(0, BlockOptionalRuntime())

            import robo_orchard_core
            import robo_orchard_core.datatypes as datatypes
            import robo_orchard_core.policy as policy
            import robo_orchard_core.tools.cli as cli
            import robo_orchard_core.utils.config as config
            from typer.testing import CliRunner

            assert not (BLOCKED & set(sys.modules))

            original_find_spec = cli.find_spec
            missing_cli_modules = {
                "fastapi",
                "aiofiles",
                "uvicorn",
                "pydantic_settings",
            }
            cli.find_spec = lambda name: (
                None
                if name.split(".", maxsplit=1)[0]
                in BLOCKED | missing_cli_modules
                else original_find_spec(name)
            )
            result = CliRunner().invoke(cli.create_app(), ["--help"])
            assert result.exit_code == 0, result.output
            assert "file-server" in result.output

            for module, name in (
                (config, "TorchTensor"),
                (config, "NumpyTensor"),
                (datatypes, "TorchTensor"),
                (datatypes, "NumpyTensor"),
                (datatypes, "BatchJointsState"),
                (policy, "PolicyMixin"),
            ):
                try:
                    getattr(module, name)
                except ModuleNotFoundError as error:
                    assert "robo_orchard_core[robotics]" in str(error)
                else:
                    raise AssertionError(f"{name} unexpectedly loaded")
            """
        )

        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            env=_without_proxies(),
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr

    def test_tensor_aliases_remain_available_to_wildcard_imports(self):
        """Keep the pre-lazy public aliases available through ``import *``."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "namespace = {}; "
                    "exec('from robo_orchard_core.utils.config import *', "
                    "namespace); "
                    "assert {'TorchTensor', 'NumpyTensor'} <= namespace.keys()"
                ),
            ],
            cwd=PROJECT_ROOT,
            env=_without_proxies(),
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr

    def test_datatype_tensor_aliases_remain_available(self):
        """Keep tensor aliases available from the historical datatype root."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from robo_orchard_core.datatypes import "
                    "NumpyTensor, TorchTensor; "
                    "assert NumpyTensor is not None; "
                    "assert TorchTensor is not None"
                ),
            ],
            cwd=PROJECT_ROOT,
            env=_without_proxies(),
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr

    def test_source_import_falls_back_when_generated_version_is_missing(self):
        """A clean source checkout exposes stable fallback version metadata."""
        expected_version = (PROJECT_ROOT / "VERSION").read_text().strip()
        with tempfile.TemporaryDirectory() as temp_dir:
            source_root = Path(temp_dir)
            shutil.copytree(
                PROJECT_ROOT / "robo_orchard_core",
                source_root / "robo_orchard_core",
                ignore=shutil.ignore_patterns("version.py", "__pycache__"),
            )
            shutil.copy2(PROJECT_ROOT / "VERSION", source_root / "VERSION")
            env = _without_proxies()
            env["PYTHONPATH"] = os.pathsep.join(
                filter(None, [str(source_root), env.get("PYTHONPATH")])
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import robo_orchard_core as core; "
                        "print(core.__version__); "
                        "print(core.__full_version__); "
                        "print(core.__git_hash__)"
                    ),
                ],
                cwd=source_root,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == [
            expected_version,
            f"{expected_version}.dev",
            "unknown",
        ]

    def test_built_base_wheel_installs_and_runs_without_robotics(self):
        """A built base wheel exposes the documented light entrypoints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source_root = temp_root / "source"
            wheel_dir = temp_root / "wheel"
            install_dir = temp_root / "install"
            blocker_dir = temp_root / "blocker"
            source_root.mkdir()
            shutil.copytree(
                PROJECT_ROOT / "robo_orchard_core",
                source_root / "robo_orchard_core",
                ignore=shutil.ignore_patterns("version.py", "__pycache__"),
            )
            for filename in (
                "pyproject.toml",
                "setup.py",
                "VERSION",
                "VERSION_POSTFIX",
                "LICENSE",
                "NOTICE",
                "README.md",
            ):
                shutil.copy2(PROJECT_ROOT / filename, source_root / filename)

            build_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    "--no-build-isolation",
                    "--no-deps",
                    "--wheel-dir",
                    str(wheel_dir),
                    str(source_root),
                ],
                cwd=temp_root,
                env=_without_proxies(),
                check=False,
                capture_output=True,
                text=True,
            )
            assert build_result.returncode == 0, build_result.stderr

            install_dir.mkdir()
            install_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--target",
                    str(install_dir),
                    str(next(wheel_dir.glob("*.whl"))),
                ],
                cwd=temp_root,
                env=_without_proxies(),
                check=False,
                capture_output=True,
                text=True,
            )
            assert install_result.returncode == 0, install_result.stderr

            blocker_dir.mkdir()
            (blocker_dir / "sitecustomize.py").write_text(
                _BLOCK_OPTIONAL_RUNTIME_SITE_CUSTOMIZE
            )
            env = _without_proxies()
            env.update(
                {
                    "EXPECTED_INSTALL": str(install_dir),
                    "PYTHONPATH": os.pathsep.join(
                        (str(blocker_dir), str(install_dir))
                    ),
                }
            )
            import_result = subprocess.run(
                [sys.executable, "-c", _INSTALLED_BASE_IMPORT_SCRIPT],
                cwd=temp_root,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            assert import_result.returncode == 0, import_result.stderr

            console_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robo_orchard_core.tools.cli",
                    "--help",
                ],
                cwd=temp_root,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            assert console_result.returncode == 0, console_result.stderr
            assert "file-server" in console_result.stdout


def _without_proxies() -> dict[str, str]:
    """Return the subprocess environment without inherited HTTP proxies."""
    env = os.environ.copy()
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        env.pop(name, None)
    return env


_BLOCK_OPTIONAL_RUNTIME_SITE_CUSTOMIZE = """
import importlib.abc
import importlib.util
import sys

BLOCKED = {
    "torch",
    "numpy",
    "numpydantic",
    "gymnasium",
    "PIL",
    "cv2",
    "pytorch_kinematics",
    "fastapi",
    "aiofiles",
    "uvicorn",
    "pydantic_settings",
}


_original_find_spec = importlib.util.find_spec


def find_spec(name, package=None):
    if name.split(".", maxsplit=1)[0] in BLOCKED:
        return None
    return _original_find_spec(name, package)


importlib.util.find_spec = find_spec


class BlockOptionalRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".", maxsplit=1)[0]
        if root in BLOCKED:
            raise ModuleNotFoundError(
                f"blocked optional runtime module: {fullname}",
                name=root,
            )
        return None


sys.meta_path.insert(0, BlockOptionalRuntime())
"""


_INSTALLED_BASE_IMPORT_SCRIPT = """
import importlib.abc
import os
from pathlib import Path
import sys

import robo_orchard_core as core
import robo_orchard_core.datatypes as datatypes
import robo_orchard_core.policy as policy
import robo_orchard_core.tools.cli as cli
import robo_orchard_core.utils.config as config
from typer.testing import CliRunner

expected_install = Path(os.environ["EXPECTED_INSTALL"]).resolve()
assert expected_install in Path(core.__file__).resolve().parents
blocked = {
    "torch",
    "numpy",
    "numpydantic",
    "gymnasium",
    "PIL",
    "cv2",
    "pytorch_kinematics",
    "fastapi",
    "aiofiles",
    "uvicorn",
    "pydantic_settings",
}
assert not (blocked & set(sys.modules))

original_find_spec = cli.find_spec
cli.find_spec = lambda name: (
    None
    if name.split(".", maxsplit=1)[0] in blocked
    else original_find_spec(name)
)
result = CliRunner().invoke(cli.create_app(), ["--help"])
assert result.exit_code == 0, result.output
assert "file-server" in result.output

for module, name in (
    (config, "TorchTensor"),
    (config, "NumpyTensor"),
    (datatypes, "BatchJointsState"),
    (policy, "PolicyMixin"),
):
    try:
        getattr(module, name)
    except ModuleNotFoundError as error:
        assert "robo_orchard_core[robotics]" in str(error)
    else:
        raise AssertionError(f"{name} unexpectedly loaded")
"""
