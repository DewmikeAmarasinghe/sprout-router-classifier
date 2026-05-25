"""
CostSimulator — estimate cost savings from the binary routing strategy.

Compares routing strategies against the baseline of sending all messages
to gpt-4o. Uses token pricing from config/settings.py.

Strategy naming convention:
    "all_gpt4o"          — baseline: everything to gpt-4o
    "all_mini"           — everything to gpt-4o-mini (unsafe)
    "script_only"        — only route pure-script messages to gpt-4o
    "router_{model_key}" — use a trained classifier

Usage:
    sim     = CostSimulator()
    results = sim.simulate_all("v1", comparator_rows)
    print(results[0].daily_savings_usd)
"""

from __future__ import annotations

import json
import logging

from backend.evaluation.pymodels import CostSimResult
from backend.shared.path_resolver import get_experiment_path
from backend.shared.settings_manager import settings_manager

log = logging.getLogger(__name__)


class CostSimulator:
    """Estimates daily API cost for different routing strategies."""

    def simulate(
        self,
        strategy_name: str,
        pct_routed_to_mini: float,
        recall_1: float,
        daily_messages: int = 10_000,
        notes: str = "",
    ) -> CostSimResult:
        """Compute daily cost for one routing strategy.

        Args:
            strategy_name: Human-readable strategy name.
            pct_routed_to_mini: Fraction of messages sent to gpt-4o-mini [0, 1].
            recall_1: Recall for label=1 — used to flag unsafe strategies.
            daily_messages: Estimated daily message volume.
            notes: Optional notes.

        Returns:
            CostSimResult with daily and monthly cost/savings.
        """
        pct_to_4o = 1.0 - pct_routed_to_mini

        gpt4o_input_cost = float(settings_manager.get("GPT4O_INPUT_PER_1M"))
        gpt4o_output_cost = float(settings_manager.get("GPT4O_OUTPUT_PER_1M"))
        mini_input_cost = float(settings_manager.get("GPT4O_MINI_INPUT_PER_1M"))
        mini_output_cost = float(settings_manager.get("GPT4O_MINI_OUTPUT_PER_1M"))
        avg_input_tokens = int(settings_manager.get("AVG_INPUT_TOKENS"))
        avg_output_tokens = int(settings_manager.get("AVG_OUTPUT_TOKENS"))

        baseline_cost = compute_daily_cost(
            n_messages=daily_messages,
            pct_to_4o=1.0,
            pct_to_mini=0.0,
            input_tokens=avg_input_tokens,
            output_tokens=avg_output_tokens,
            gpt4o_in=gpt4o_input_cost,
            gpt4o_out=gpt4o_output_cost,
            mini_in=mini_input_cost,
            mini_out=mini_output_cost,
        )

        strategy_cost = compute_daily_cost(
            n_messages=daily_messages,
            pct_to_4o=pct_to_4o,
            pct_to_mini=pct_routed_to_mini,
            input_tokens=avg_input_tokens,
            output_tokens=avg_output_tokens,
            gpt4o_in=gpt4o_input_cost,
            gpt4o_out=gpt4o_output_cost,
            mini_in=mini_input_cost,
            mini_out=mini_output_cost,
        )

        daily_savings = baseline_cost - strategy_cost
        monthly_savings = daily_savings * 30

        return CostSimResult(
            strategy_name=strategy_name,
            daily_messages=daily_messages,
            pct_routed_to_mini=pct_routed_to_mini,
            pct_routed_to_4o=pct_to_4o,
            daily_cost_usd=round(strategy_cost, 4),
            baseline_cost_usd=round(baseline_cost, 4),
            daily_savings_usd=round(daily_savings, 4),
            monthly_savings_usd=round(monthly_savings, 2),
            recall_1=recall_1,
            notes=notes,
        )

    def simulate_all(
        self,
        dataset_name: str,
        comparison_rows: list,
        daily_messages: int | None = None,
    ) -> list[CostSimResult]:
        """Simulate cost for all strategies + comparison rows.

        Always includes baselines: all_gpt4o and all_mini.

        Args:
            dataset_name: e.g. "v1".
            comparison_rows: List of ComparisonRow from ModelComparator.compare().
            daily_messages: Estimated daily message volume.

        Returns:
            List of CostSimResult sorted by daily_savings descending.
        """
        n_messages = (
            daily_messages
            if daily_messages is not None
            else int(settings_manager.get("DAILY_MESSAGES_ESTIMATE"))
        )
        results: list[CostSimResult] = [
            self.simulate(
                "all_gpt4o",
                pct_routed_to_mini=0.0,
                recall_1=1.0,
                daily_messages=n_messages,
                notes="Baseline: no routing",
            ),
            self.simulate(
                "all_mini",
                pct_routed_to_mini=1.0,
                recall_1=0.0,
                daily_messages=n_messages,
                notes="Unsafe: no gpt-4o at all",
            ),
        ]

        for row in comparison_rows:
            pct_to_mini = getattr(row, "precision_0", 0.0) * (1 - row.recall_1)
            pct_to_mini = max(0.0, min(1.0, pct_to_mini))

            results.append(
                self.simulate(
                    strategy_name=f"router_{row.experiment_id}",
                    pct_routed_to_mini=pct_to_mini,
                    recall_1=row.recall_1,
                    daily_messages=n_messages,
                    notes=f"{row.approach} | MCC={row.mcc:.4f}",
                )
            )

        results = sort_cost_results(results)
        save_results(results, dataset_name)
        return results


def sort_cost_results(results: list[CostSimResult]) -> list[CostSimResult]:
    """Order results for display: baselines first, then routers by savings."""
    baseline_order = {"all_gpt4o": 0, "all_mini": 1}
    baselines = sorted(
        (r for r in results if r.strategy_name in baseline_order),
        key=lambda r: baseline_order[r.strategy_name],
    )
    routers = sorted(
        (r for r in results if r.strategy_name.startswith("router_")),
        key=lambda r: -r.daily_savings_usd,
    )
    return baselines + routers


def compute_daily_cost(
    n_messages: int,
    pct_to_4o: float,
    pct_to_mini: float,
    input_tokens: int,
    output_tokens: int,
    gpt4o_in: float,
    gpt4o_out: float,
    mini_in: float,
    mini_out: float,
) -> float:
    """Compute daily API cost for a split routing strategy."""
    n_4o = n_messages * pct_to_4o
    n_mini = n_messages * pct_to_mini

    cost_4o = n_4o * (input_tokens * gpt4o_in + output_tokens * gpt4o_out) / 1_000_000
    cost_mini = n_mini * (input_tokens * mini_in + output_tokens * mini_out) / 1_000_000

    return cost_4o + cost_mini


def save_results(results: list[CostSimResult], dataset_name: str) -> None:
    output_dir = get_experiment_path(dataset_name, "classical").parent
    path = output_dir / "cost_simulation.json"
    path.write_text(
        json.dumps([r.model_dump() for r in results], indent=2),
        encoding="utf-8",
    )
    log.info(f"Saved cost simulation: {path}")
