"""Shared product labels for HTML presentation."""

HARNESS_LABELS = {
    'pi': 'Pi',
    'codex': 'Codex',
    'claude': 'Claude Code',
    'copilot-vscode': 'Copilot in VS Code',
    'copilot-cli': 'Copilot CLI',
}


def harness_label(value: str) -> str:
    return HARNESS_LABELS.get(value, 'Unknown source')
