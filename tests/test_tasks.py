# Copyright (c) 2025 Soares
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Invoke task helpers in tasks.py.
"""

import tasks


class DummyContext:
    """Minimal Invoke-like context with a run method that records commands."""

    def __init__(self):
        self.commands: list[tuple[str, bool]] = []

    def run(self, cmd: str, pty: bool | None = None, warn: bool | None = None):  # noqa: D401, ARG002
        self.commands.append((cmd, bool(pty)))


def test_task_test_builds_expected_command():
    """
    Tests that the `test` task generates the expected command based on the input
    parameters. This ensures the appropriate pytest flags are included in the
    constructed command.

    Raises:
        AssertionError: If any of the command checks fail, indicating incorrect
                        command construction.
    """
    ctx = DummyContext()
    # Call the underlying function body to avoid Invoke's Context type check
    tasks.test.body(ctx, coverage_enable=True, verbose=False, markers="unit")

    assert ctx.commands
    cmd, _ = ctx.commands[0]
    assert "uv run pytest" in cmd
    assert "--cov" in cmd
    assert "-q" in cmd
    assert "-m unit" in cmd


def test_task_lint_and_formatter_commands():
    """
    Test the tasks for linting and formatting commands execution.

    This test ensures that the lint and formatter commands are executed
    correctly with the specified parameters and verifies the presence
    of expected commands in the context.

    Parameters:
    ctx (DummyContext): A dummy context object simulating the context in which
    the commands are executed.

    Assertions:
    Verifies:
    - That "uv run ruff check" command is executed with the "--fix" option.
    - That "uv run ruff format" command is executed with the "--check" option.
    """
    ctx = DummyContext()
    tasks.lint.body(ctx, fix=True)
    tasks.run_ruff_formatter.body(ctx, check_flag=True)

    cmds = [cmd for cmd, _ in ctx.commands]
    assert any("uv run ruff check" in c and "--fix" in c for c in cmds)
    assert any("uv run ruff format" in c and "--check" in c for c in cmds)


def test_task_typecheck_and_coverage_command_building():
    """
    Tests the command execution for type checking and coverage in task management.

    This test ensures that the appropriate type checking (mypy) and code coverage
    (pytest with coverage) commands are constructed and stored in the task context
    when invoked.

    Raises:
        AssertionError: If the expected commands for type checking or coverage
        are not found in the context commands.
    """

    ctx = DummyContext()
    tasks.typecheck.body(ctx)
    tasks.coverage.body(ctx, report=False)

    cmds = [cmd for cmd, _ in ctx.commands]
    assert any("uv run mypy" in c for c in cmds)
    assert any("uv run pytest --cov" in c for c in cmds)


def test_task_pipeline_builds_expected_command():
    """
    Tests the construction of the expected command for the task pipeline.

    This test verifies that the `pipeline.body` method creates the correct
    command based on the provided parameters and that the constructed command
    matches the expected format. It also checks that the `pty` parameter
    is set as expected.

    Raises:
        AssertionError: If the constructed command does not meet
        the expected values, or the `pty` value is incorrect.

    """
    ctx = DummyContext()
    tasks.pipeline.body(
        ctx, country="XX", start_date="2024-01-01", end_date="2024-01-07", fetch=True
    )

    cmd, pty = ctx.commands[0]
    assert "uv run python run_pipeline.py" in cmd
    assert "--country XX" in cmd
    assert "--start-date 2024-01-01" in cmd
    assert "--end-date 2024-01-07" in cmd
    assert "--fetch" in cmd
    assert pty == tasks.PTY
