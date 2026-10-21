"""Reporting and summary table generation for evaluation workflows."""

from typing import Any


def print_threshold_calibration_table(calibration_sweep: list[dict[str, Any]]) -> float:
    """Print the calibration sweep table and determine the best threshold."""
    print("\n### Threshold Calibration Sweep (DEV Only)\n")
    print("| Threshold | Reference Mean | Max Neg Ctrl | Margin | Status |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    for sw in calibration_sweep:
        st = "PASS" if sw["passed"] else "FAIL"
        print(
            f"| {sw['threshold']:.2f} | {sw['mean_reference']:.3f} | {sw['max_negative_control']:.3f} | {sw['margin']:+.3f} | {st} |"
        )

    valid_sweeps = [sw for sw in calibration_sweep if sw["passed"]]
    if valid_sweeps:
        best_sweep = max(valid_sweeps, key=lambda sw: (sw["margin"], sw["threshold"]))
        best_calibrated = best_sweep["threshold"]
        print(
            f"\nOptimal Calibrated Threshold on DEV: {best_calibrated:.2f} (margin: {best_sweep['margin']:+.3f})"
        )
    else:
        best_sweep = max(calibration_sweep, key=lambda sw: sw["margin"])
        best_calibrated = best_sweep["threshold"]
        print(
            f"\nNo threshold met pass rule on DEV. Best candidate threshold: {best_calibrated:.2f} (margin: {best_sweep['margin']:+.3f})"
        )

    return float(best_calibrated)


def print_validation_set_summary(
    name: str,
    chosen_threshold: float,
    refs: list[dict[str, Any]],
    negs: list[dict[str, Any]],
    m_ref: float,
    m_neg: float,
    max_neg: float,
    ref_accept_rate: float,
    ref_accept_cnt: int,
    scorable_ref_count: int,
    neg_false_accept_rate: float,
    neg_false_accept_cnt: int,
    scorable_neg_count: int,
    neg_fully_accepted_rate: float,
    neg_fully_accepted_cnt: int,
    unscorable_ref_cnt: int,
    unscorable_neg_cnt: int,
    passed: bool,
) -> None:
    """Print formatted summary table and metrics for a validation set."""
    print(f"\n### {name} Summary Table (Threshold: {chosen_threshold:.2f})\n")
    print("| Item ID | Type | Supported | Total | Faithfulness | Status |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in refs:
        if r.get("unscorable"):
            f_str, st = "n/a", "UNSCORABLE (COVERAGE FAIL)"
        else:
            f_str = f"{r['faithfulness']:.2f}"
            st = "PASS" if r["faithfulness"] >= 0.80 else "FAIL"
        print(
            f"| {r['id']} | REFERENCE | {r.get('supported_items', 0)} | {r.get('total_items', 0)} | {f_str} | {st} |"
        )

    for r in negs:
        if r.get("unscorable"):
            f_str, st = "n/a", "UNSCORABLE"
        else:
            f_str = f"{r['faithfulness']:.2f}"
            st = "PASS" if r["faithfulness"] <= 0.50 else "FAIL"
        print(
            f"| {r['id']} | NEG_CTRL | {r.get('supported_items', 0)} | {r.get('total_items', 0)} | {f_str} | {st} |"
        )

    print(f"\n{name} Metrics:")
    print(f"  Mean Reference Faithfulness:         {m_ref:.3f} (Required >= 0.800)")
    print(f"  Mean Negative Control Faithfulness:  {m_neg:.3f}")
    print(f"  Max Negative Control Faithfulness:   {max_neg:.3f} (Required <= 0.500)")
    print(
        f"  Reference Accept Rate:               {ref_accept_rate:.3f} ({ref_accept_cnt}/{scorable_ref_count})"
    )
    print(
        f"  Negative Control False-Accept Rate:  {neg_false_accept_rate:.3f} ({neg_false_accept_cnt}/{scorable_neg_count})"
    )
    print(
        f"  Negative Control Fully-Accepted Rate:{neg_fully_accepted_rate:.3f} ({neg_fully_accepted_cnt}/{scorable_neg_count})"
    )
    print(f"  Unscorable References:               {unscorable_ref_cnt} (Required == 0)")
    print(f"  Unscorable Negative Controls:        {unscorable_neg_cnt}")
    print(f"  Set Status:                          {'PASS' if passed else 'FAIL'}")


def print_failure_diagnostics(dev_summary: dict[str, Any], heldout_summary: dict[str, Any]) -> None:
    """Print detailed failure diagnostics for unvalidated judges."""
    print("\n================================================================================")
    print("[FAIL] JUDGE NOT TRUSTWORTHY: Detailed Failure Diagnostic")
    print("================================================================================")
    for set_label, s_data in [("DEV", dev_summary), ("HELD-OUT", heldout_summary)]:
        all_set_items = s_data.get("reference_results", []) + s_data.get(
            "negative_control_results", []
        )
        for itm in all_set_items:
            is_nc = bool(itm.get("is_negative_control", False))
            f_val = itm.get("faithfulness")
            unsc = bool(itm.get("unscorable", False))
            item_failed = (
                unsc
                or (not is_nc and (f_val is None or f_val < 0.80))
                or (is_nc and f_val is not None and f_val > 0.50)
            )
            if item_failed:
                print(
                    f"\nItem [{itm['id']}] ({set_label} | {'NEG_CTRL' if is_nc else 'REFERENCE'}): faithfulness={f_val} unscorable={unsc}"
                )
                print(f"  Question: {itm.get('question')}")
                print(f"  Answer:   {itm.get('answer')}")
                for s_i, s_entry in enumerate(itm.get("details", [])):
                    print(f'    Sentence {s_i + 1}: "{s_entry.get("sentence")}"')
                    print(
                        f"      Supported: {s_entry.get('supported')} (entail={s_entry.get('entailment_prob')}, contra={s_entry.get('contradiction_prob')})"
                    )
                    print(
                        f'      Best Window [{s_entry.get("best_window_index")}]: "{s_entry.get("best_window")}"'
                    )


def print_faithfulness_summary_table(
    results: dict[str, Any],
    active_judge_name: str,
    judge_val_verdict: str,
    heldout_accept_rate: float | None,
    heldout_false_accept_rate: float | None,
) -> tuple[float, int]:
    """Print the final evaluation summary table and calculate mean faithfulness."""
    print("\n### Faithfulness Evaluation Summary Table\n")
    print(f"**Judge Model**: `{active_judge_name}`")
    print(f"**Judge Validation Verdict**: `{judge_val_verdict}`")
    h_acc_str = f"{heldout_accept_rate:.3f}" if heldout_accept_rate is not None else "n/a"
    h_fa_str = (
        f"{heldout_false_accept_rate:.3f}" if heldout_false_accept_rate is not None else "n/a"
    )
    print(f"**HELD-OUT Accept Rate**: `{h_acc_str}`")
    print(f"**HELD-OUT False-Accept Rate**: `{h_fa_str}`\n")
    if judge_val_verdict != "VALID":
        print("UNVALIDATED JUDGE: treat scores as directional\n")

    print("| Question ID | Question Excerpt | Supported | Total | Faithfulness | Status |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    applicable_scores: list[float] = []
    unscorable_count = 0

    for r in results.values():
        if "error" in r:
            status_str = "ERROR"
            f_str = "n/a"
        elif not r.get("applicable", True):
            status_str = "REFUSAL / ABSTAINED"
            f_str = "n/a"
        elif r.get("unscorable", False):
            status_str = "UNSCORABLE"
            f_str = "n/a"
            unscorable_count += 1
        else:
            score_val = r.get("faithfulness")
            if score_val is not None:
                applicable_scores.append(score_val)
                f_str = f"{score_val:.2f}"
                status_str = "PASS" if score_val >= 0.7 else "WARN"
            else:
                f_str = "n/a"
                status_str = "UNSCORABLE"
                unscorable_count += 1

        q_excerpt = (r["question"][:40] + "...") if len(r["question"]) > 40 else r["question"]
        print(
            f"| {r['id']} | {q_excerpt} | {r.get('supported_items', 0)} | {r.get('total_items', 0)} | {f_str} | {status_str} |"
        )

    mean_f = sum(applicable_scores) / len(applicable_scores) if applicable_scores else 1.0
    print(
        f"\n**Mean Answer Faithfulness**: `{mean_f:.3f}` across {len(applicable_scores)} applicable answers."
    )
    if unscorable_count > 0:
        print(f"**Unscorable Answers Excluded from Mean**: {unscorable_count}")

    return mean_f, unscorable_count
