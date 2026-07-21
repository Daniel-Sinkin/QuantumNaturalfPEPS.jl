import contextlib
import importlib.util
import io
import os
import shutil
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock


PACKAGE_SOURCE = Path(__file__).resolve().parents[1]
PYTHON_SETUP_SOURCE = PACKAGE_SOURCE / "util" / "setup.py"
SHELL_SETUP_SOURCE = PACKAGE_SOURCE / "setup.sh"


class PythonSetupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.package = self.root / "cuQuantumNaturalfPEPS"
        for relative in ("build/cuda", "cuda", "src", "util"):
            (self.package / relative).mkdir(parents=True, exist_ok=True)
        shutil.copy2(PYTHON_SETUP_SOURCE, self.package / "util" / "setup.py")
        (self.package / "c_api_version.txt").write_text("0.0.5\n", encoding="utf-8")
        (self.package / "setup.sh").write_text("fixture\n", encoding="utf-8")
        (self.package / "cuda" / "source.cu").write_text("cuda\n", encoding="utf-8")
        (self.package / "src" / "source.jl").write_text("julia\n", encoding="utf-8")
        (self.package / "util" / "configure_julia.jl").write_text(
            "configure\n", encoding="utf-8"
        )
        build = self.package / "util" / "build_cuda.sh"
        build.write_text("build\n", encoding="utf-8")
        build.chmod(0o755)
        name = f"qnpeps_setup_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(name, self.package / "util" / "setup.py")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.environment = mock.patch.dict(os.environ, {}, clear=False)
        self.environment.start()
        for key in (
            "CUDA_HOME",
            "JULIA_DEPOT_PATH",
            "QNPEPS_CUDA_VERSION",
            "QNPEPS_LIB",
        ):
            os.environ.pop(key, None)
        os.environ["QNPEPS_ACTIVE_ROOT"] = str(self.package)

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    @property
    def library(self):
        return self.package / "build" / "cuda" / "qnpeps.so"

    def write_library(self, *versions, path=None):
        target = self.library if path is None else path
        payload = b"fixture"
        for version in versions:
            payload += b"\0cuQuantumNaturalfPEPS " + version.encode("ascii") + b"\0"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return target

    def prepare_current(self):
        self.write_library("0.0.5")
        self.module.STAMP_FILE.write_text(
            self.module.setup_signature() + "\n", encoding="utf-8"
        )

    def successful_runner(self, calls, observations=None):
        def execute(command):
            calls.append(tuple(command))
            if observations is not None:
                observations.append(self.library.exists())
            if Path(command[0]).name == "build_cuda.sh":
                self.write_library("0.0.5")
            return True

        return execute

    def run_setup(self, force=False, show_path=False, observations=None):
        calls = []
        with mock.patch.object(
            self.module,
            "run",
            side_effect=self.successful_runner(calls, observations),
        ), mock.patch.object(self.module, "visible_gpu_exists", return_value=False):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                status = self.module.setup(force=force, show_path=show_path)
        return status, calls

    def test_current_library_and_signature_are_noop(self):
        self.prepare_current()
        with mock.patch.object(self.module, "run") as run:
            with contextlib.redirect_stdout(io.StringIO()):
                status = self.module.setup(force=False, show_path=False)
        self.assertEqual(status, 0)
        run.assert_not_called()
        self.assertTrue(self.library.is_file())

    def test_source_signature_change_rebuilds(self):
        self.prepare_current()
        (self.package / "cuda" / "source.cu").write_text("changed\n", encoding="utf-8")
        observations = []
        status, calls = self.run_setup(observations=observations)
        self.assertEqual(status, 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(observations[0], True)
        self.assertEqual(
            self.module.STAMP_FILE.read_text(encoding="utf-8").strip(),
            self.module.setup_signature(),
        )

    def test_toolchain_signature_change_rebuilds(self):
        self.prepare_current()
        os.environ["QNPEPS_CUDA_VERSION"] = "13.0"
        status, calls = self.run_setup()
        self.assertEqual(status, 0)
        self.assertEqual(len(calls), 2)

    def test_version_upgrade_removes_production_library_before_build(self):
        self.write_library("0.0.4")
        observations = []
        status, calls = self.run_setup(observations=observations)
        self.assertEqual(status, 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(observations, [False, False])
        self.assertEqual(
            self.module.inspect_version(self.library)[0], self.module.VersionState.MATCH
        )

    def test_force_rebuilds_and_forwards_show_path(self):
        self.prepare_current()
        observations = []
        status, calls = self.run_setup(force=True, show_path=True, observations=observations)
        self.assertEqual(status, 0)
        self.assertEqual(observations, [False, False])
        self.assertEqual(calls[1][-1], "--show-path")
        self.assertNotIn("--show-path", calls[0])

    def test_show_path_does_not_change_noop_decision(self):
        self.prepare_current()
        with mock.patch.object(self.module, "run") as run:
            with contextlib.redirect_stdout(io.StringIO()):
                status = self.module.setup(force=False, show_path=True)
        self.assertEqual(status, 0)
        run.assert_not_called()

    def test_missing_library_builds(self):
        status, calls = self.run_setup()
        self.assertEqual(status, 0)
        self.assertEqual(len(calls), 2)
        self.assertTrue(self.library.is_file())

    def test_invalid_library_is_replaced(self):
        self.write_library()
        observations = []
        status, calls = self.run_setup(observations=observations)
        self.assertEqual(status, 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(observations[0], False)

    def test_multiple_compiled_versions_are_replaced(self):
        self.write_library("0.0.4", "0.0.5")
        state, detail, wanted = self.module.inspect_version(self.library)
        self.assertIs(state, self.module.VersionState.INVALID)
        self.assertEqual(detail, "0.0.4, 0.0.5")
        self.assertEqual(wanted, "0.0.5")
        observations = []
        status, calls = self.run_setup(observations=observations)
        self.assertEqual(status, 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(observations[0], False)

    def test_check_reports_active_and_inactive_exact_checkout(self):
        with mock.patch.object(self.module.shutil, "which", return_value="/fixture/tool"):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                active_status = self.module.main(["--check"])
                os.environ["QNPEPS_ACTIVE_ROOT"] = str(self.root / "other")
                inactive_status = self.module.main(["--check"])
        self.assertEqual(active_status, 0)
        self.assertEqual(inactive_status, 1)

    def test_api_version_does_not_call_setup(self):
        output = io.StringIO()
        with mock.patch.object(self.module, "setup") as setup, contextlib.redirect_stdout(output):
            status = self.module.main(["--api-version"])
        self.assertEqual(status, 0)
        self.assertEqual(output.getvalue(), "0.0.5\n")
        setup.assert_not_called()

    def test_compiled_version_check_uses_qnpeps_lib(self):
        self.write_library("0.0.5")
        override = self.write_library("0.0.4", path=self.root / "override.so")
        os.environ["QNPEPS_LIB"] = str(override)
        with mock.patch.object(self.module, "setup") as setup, contextlib.redirect_stdout(
            io.StringIO()
        ), contextlib.redirect_stderr(io.StringIO()):
            status = self.module.main(["--check-version"])
        self.assertEqual(status, 1)
        setup.assert_not_called()

    def test_qnpeps_lib_cannot_redirect_production_deletion(self):
        override = self.write_library("0.0.4", path=self.root / "override.so")
        override_contents = override.read_bytes()
        self.write_library("0.0.4")
        os.environ["QNPEPS_LIB"] = str(override)
        observations = []
        status, calls = self.run_setup(observations=observations)
        self.assertEqual(status, 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(observations[0], False)
        self.assertEqual(override.read_bytes(), override_contents)

    def test_invalid_and_help_options_do_not_call_setup(self):
        for arguments, expected_status in (
            (["--invalid"], 2),
            (["--check", "--force"], 2),
            (["--help"], 0),
        ):
            with self.subTest(arguments=arguments):
                with mock.patch.object(self.module, "setup") as setup, contextlib.redirect_stdout(
                    io.StringIO()
                ), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        self.module.main(arguments)
                self.assertEqual(raised.exception.code, expected_status)
                setup.assert_not_called()

    def test_unset_cuda_home_signature_is_independent_of_cwd(self):
        os.environ.pop("CUDA_HOME", None)
        first = self.root / "first"
        second = self.root / "second"
        first.mkdir()
        second.mkdir()
        previous = Path.cwd()
        try:
            os.chdir(first)
            first_signature = self.module.setup_signature()
            os.chdir(second)
            second_signature = self.module.setup_signature()
        finally:
            os.chdir(previous)
        self.assertEqual(first_signature, second_signature)


class ShellSetupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.package = self.root / "cuQuantumNaturalfPEPS"
        self.util = self.package / "util"
        self.fake_bin = self.root / "bin"
        self.util.mkdir(parents=True)
        self.fake_bin.mkdir()
        shutil.copy2(SHELL_SETUP_SOURCE, self.package / "setup.sh")
        environment = self.util / "environment.sh"
        environment.write_text(
            "#!/usr/bin/env bash\n"
            "printf 'environment' >> \"$SETUP_TEST_LOG\"\n"
            "printf '|%s' \"$@\" >> \"$SETUP_TEST_LOG\"\n"
            "printf '\\n' >> \"$SETUP_TEST_LOG\"\n"
            "export QNPEPS_ACTIVE_ROOT=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")/..\" && pwd)\"\n"
            "export SETUP_ENV_PERSISTED=1\n"
            "export PATH=\"$PATH:$SETUP_TEST_PATH_SUFFIX\"\n",
            encoding="utf-8",
        )
        environment.chmod(0o755)
        python = self.fake_bin / "python3"
        python.write_text(
            "#!/usr/bin/env bash\n"
            "printf 'python' >> \"$SETUP_TEST_LOG\"\n"
            "printf '|%s' \"$@\" >> \"$SETUP_TEST_LOG\"\n"
            "printf '\\n' >> \"$SETUP_TEST_LOG\"\n"
            "status=\"${SETUP_TEST_PYTHON_STATUS:-0}\"\n"
            "check=0\n"
            "force=0\n"
            "for argument in \"$@\"; do\n"
            "    case \"$argument\" in\n"
            "        -c|--check|-cv|--check-version|--api-version) check=1 ;;\n"
            "        -f|--force) force=1 ;;\n"
            "        --invalid) status=2 ;;\n"
            "    esac\n"
            "done\n"
            "[ \"$check\" -eq 0 ] || [ \"$force\" -eq 0 ] || status=2\n"
            "exit \"$status\"\n",
            encoding="utf-8",
        )
        python.chmod(0o755)
        self.log = self.root / "calls.log"
        self.environment = os.environ.copy()
        self.environment.update(
            {
                "PATH": f"{self.fake_bin}:{self.environment['PATH']}",
                "SETUP_TEST_LOG": str(self.log),
                "SETUP_TEST_PATH_SUFFIX": str(self.root / "persisted"),
            }
        )

    def tearDown(self):
        self.temporary.cleanup()

    def source(self, *arguments, python_status=0):
        environment = self.environment.copy()
        environment["SETUP_TEST_PYTHON_STATUS"] = str(python_status)
        command = (
            "_setup_test_source() {\n"
            '    local setup_file="$1"\n'
            "    shift\n"
            '    source "$setup_file" "$@"\n'
            "}\n"
            '_setup_test_source "$@"\n'
            "source_status=$?\n"
            "declare -F _qnpeps_setup >/dev/null; setup_function=$?\n"
            "declare -F _qnpeps_setup_cleanup >/dev/null; cleanup_function=$?\n"
            "printf 'result|%s|%s|%s|%s|%s|%s\\n' \"$source_status\" "
            '"${QNPEPS_ACTIVE_ROOT:-}" "${SETUP_ENV_PERSISTED:-}" '
            '"$PATH" "$setup_function" "$cleanup_function"\n'
        )
        return subprocess.run(
            ["bash", "-c", command, "setup-test", str(self.package / "setup.sh"), *arguments],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            check=False,
        )

    def calls(self):
        return self.log.read_text(encoding="utf-8").splitlines() if self.log.exists() else []

    def test_activation_changes_persist_and_cleanup_is_complete(self):
        result = self.source()
        fields = result.stdout.strip().split("|")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(fields[1], "0")
        self.assertEqual(Path(fields[2]), self.package)
        self.assertEqual(fields[3], "1")
        self.assertTrue(fields[4].endswith(str(self.root / "persisted")))
        self.assertEqual(fields[5:], ["1", "1"])
        self.assertEqual(len(self.calls()), 2)
        self.assertTrue(self.calls()[0].startswith("environment"))
        self.assertTrue(self.calls()[1].startswith("python|-B|"))

    def test_show_path_is_forwarded_to_activation_and_python(self):
        result = self.source("--show-path")
        self.assertEqual(result.returncode, 0)
        calls = self.calls()
        self.assertEqual(calls[0], "environment|--show-path")
        self.assertTrue(calls[1].endswith("|--show-path"))

    def test_diagnostic_help_and_invalid_modes_do_not_activate(self):
        for arguments, expected_status in (
            (("--check",), "0"),
            (("--api-version",), "0"),
            (("--check-version",), "0"),
            (("--help",), "0"),
            (("--invalid",), "2"),
            (("--check", "--force"), "2"),
        ):
            with self.subTest(arguments=arguments):
                if self.log.exists():
                    self.log.unlink()
                result = self.source(*arguments)
                self.assertEqual(result.returncode, 0)
                self.assertEqual(result.stdout.split("|", 2)[1], expected_status)
                calls = self.calls()
                self.assertEqual(len(calls), 1)
                self.assertTrue(calls[0].startswith("python|-B|"))

    def test_nonzero_python_status_is_returned_and_functions_are_removed(self):
        result = self.source("--force", python_status=7)
        fields = result.stdout.strip().split("|")
        self.assertEqual(fields[1], "7")
        self.assertEqual(fields[5:], ["1", "1"])

    def test_cleanup_runs_with_errexit(self):
        environment = self.environment.copy()
        environment["SETUP_TEST_PYTHON_STATUS"] = "7"
        command = (
            "trap 'if declare -F _qnpeps_setup >/dev/null; then "
            "printf \"setup-present\\n\"; else printf \"setup-absent\\n\"; fi; "
            "if declare -F _qnpeps_setup_cleanup >/dev/null; then "
            "printf \"cleanup-present\\n\"; else printf \"cleanup-absent\\n\"; fi' EXIT\n"
            "set -e\n"
            'source "$1" --force\n'
        )
        result = subprocess.run(
            ["bash", "-c", command, "setup-test", str(self.package / "setup.sh")],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            check=False,
        )
        self.assertEqual(result.returncode, 7)
        self.assertIn("setup-absent", result.stdout)
        self.assertIn("cleanup-absent", result.stdout)

    def test_direct_execution_is_rejected_before_activation_or_python(self):
        result = subprocess.run(
            [str(self.package / "setup.sh"), "--force"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.environment,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("must be sourced", result.stderr)
        self.assertEqual(self.calls(), [])


if __name__ == "__main__":
    unittest.main()
