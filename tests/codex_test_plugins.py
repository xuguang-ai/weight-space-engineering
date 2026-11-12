"""Operator-side extensions for the Codex adapter tests (``--setup codex_test_plugins:setup``).

``stop_at_once@1`` reads the experiment and then stops, so the Codex
stand-in ends its turn with a deliberate ``WSE-STOP:`` message.
"""

from __future__ import annotations

import wse_test_plugins

from wse.strategies.base import Action


class StopAtOnce:
    strategy_id = "stop_at_once"
    version = "1"

    def config(self) -> dict:
        return {}

    def next_action(self, objective, observations, remaining_budget) -> Action:
        return Action.stop("nothing is worth the budget")


def setup(registry):
    wse_test_plugins.setup(registry)
    registry.register("strategy", StopAtOnce.strategy_id, StopAtOnce.version, StopAtOnce)
