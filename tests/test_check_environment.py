import unittest

from scripts import check_environment as env


class VersionParsingTests(unittest.TestCase):
    def test_parse_python_version(self):
        version = env.parse_python_version("Python 3.12.10")

        self.assertIsNotNone(version)
        self.assertEqual((version.major, version.minor, version.patch), (3, 12, 10))

    def test_parse_node_version(self):
        version = env.parse_node_version("v24.14.1")

        self.assertIsNotNone(version)
        self.assertEqual((version.major, version.minor, version.patch), (24, 14, 1))

    def test_parse_java_version(self):
        version = env.parse_java_version('java version "17.0.12"')

        self.assertIsNotNone(version)
        self.assertEqual((version.major, version.minor, version.patch), (17, 0, 12))

    def test_parse_legacy_java_version_major(self):
        version = env.parse_java_version('java version "1.8.0_402"')

        self.assertIsNotNone(version)
        self.assertEqual(version.major, 8)

    def test_parse_git_version(self):
        version = env.parse_git_version("git version 2.51.2.windows.1")

        self.assertIsNotNone(version)
        self.assertEqual((version.major, version.minor, version.patch), (2, 51, 2))

    def test_parse_compose_version(self):
        version = env.parse_compose_version("Docker Compose version v2.35.1-desktop.1")

        self.assertIsNotNone(version)
        self.assertEqual((version.major, version.minor, version.patch), (2, 35, 1))

    def test_parse_docker_ostype(self):
        self.assertEqual(env.parse_docker_ostype("linux"), "linux")
        self.assertEqual(env.parse_docker_ostype('Server:\n OSType: windows'), "windows")
        self.assertIsNone(env.parse_docker_ostype("unknown"))

    def test_infer_wsl2_from_default_version(self):
        self.assertTrue(env.infer_wsl2_available("WSL version: 2.4.13.0", "Default Version: 2"))
        self.assertFalse(env.infer_wsl2_available("WSL version: 1.0.0", "Default Version: 1"))

    def test_infer_wsl2_from_nul_separated_output(self):
        nul_output = "\x00".join("WSL version: 2.4.13.0")

        self.assertTrue(env.infer_wsl2_available(nul_output))


class RequirementEvaluationTests(unittest.TestCase):
    def test_python_requires_312(self):
        passing = env.evaluate_python_312(env.ParsedVersion(3, 12, 10, "3.12.10"))
        failing = env.evaluate_python_312(env.ParsedVersion(3, 11, 9, "3.11.9"))

        self.assertEqual(passing.status, env.Status.PASS)
        self.assertEqual(failing.status, env.Status.FAIL)

    def test_exact_major_requirement(self):
        passing = env.require_exact_major("Node.js", env.ParsedVersion(24, 14, 1, "24.14.1"), 24)
        failing = env.require_exact_major("Node.js", env.ParsedVersion(22, 12, 0, "22.12.0"), 24)

        self.assertEqual(passing.status, env.Status.PASS)
        self.assertEqual(failing.status, env.Status.FAIL)

    def test_minimum_major_requirement(self):
        passing = env.require_minimum_major("Java", env.ParsedVersion(21, 0, 1, "21.0.1"), 17)
        failing = env.require_minimum_major("Java", env.ParsedVersion(11, 0, 20, "11.0.20"), 17)

        self.assertEqual(passing.status, env.Status.PASS)
        self.assertEqual(failing.status, env.Status.FAIL)

    def test_npm_has_no_strict_patch_requirement(self):
        result = env.evaluate_npm_version(env.ParsedVersion(11, 6, 2, "11.6.2"))

        self.assertEqual(result.status, env.Status.PASS)

    def test_resolve_npm_command_prefers_cmd_on_windows(self):
        def fake_which(command):
            paths = {
                "npm": r"C:\Program Files\nodejs\npm",
                "npm.cmd": r"C:\Program Files\nodejs\npm.cmd",
            }
            return paths.get(command)

        result = env.resolve_npm_command("Windows", fake_which)

        self.assertEqual(result, r"C:\Program Files\nodejs\npm.cmd")

    def test_resolve_npm_command_uses_plain_name_off_windows(self):
        def fake_which(command):
            return "/usr/bin/npm" if command == "npm" else None

        result = env.resolve_npm_command("Linux", fake_which)

        self.assertEqual(result, "/usr/bin/npm")

    def test_check_npm_runs_resolved_windows_cmd(self):
        executed_commands = []

        def fake_which(command):
            return r"C:\Program Files\nodejs\npm.cmd" if command == "npm.cmd" else None

        def fake_command_runner(args):
            executed_commands.append(tuple(args))
            return env.CommandResult(tuple(args), 0, stdout="11.11.0\n")

        result = env.check_npm("Windows", fake_which, fake_command_runner)

        self.assertEqual(result.status, env.Status.PASS)
        self.assertEqual(executed_commands, [(r"C:\Program Files\nodejs\npm.cmd", "--version")])

    def test_git_requires_reasonably_modern_major(self):
        passing = env.evaluate_git_version(env.ParsedVersion(2, 51, 2, "2.51.2"))
        failing = env.evaluate_git_version(env.ParsedVersion(1, 9, 5, "1.9.5"))

        self.assertEqual(passing.status, env.Status.PASS)
        self.assertEqual(failing.status, env.Status.FAIL)

    def test_docker_container_mode_evaluation(self):
        self.assertEqual(env.evaluate_docker_ostype("linux").status, env.Status.PASS)
        self.assertEqual(env.evaluate_docker_ostype("windows").status, env.Status.FAIL)
        self.assertEqual(env.evaluate_docker_ostype(None).status, env.Status.WARN)

    def test_exit_code_uses_mandatory_failures(self):
        passing_results = [env.CheckResult("A", env.Status.PASS, "ok")]
        failing_results = [env.CheckResult("A", env.Status.FAIL, "bad")]
        warning_results = [env.CheckResult("A", env.Status.WARN, "check")]

        self.assertEqual(env.exit_code_for(passing_results), 0)
        self.assertEqual(env.exit_code_for(warning_results), 0)
        self.assertEqual(env.exit_code_for(failing_results), 1)


if __name__ == "__main__":
    unittest.main()