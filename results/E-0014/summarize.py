"""Сводка воспроизведённых траекторий и парных вмешательств."""
from collections import Counter
from pathlib import Path
from statistics import median

from neural_rl.experiment import read, save
from neural_rl.stress_hundreds import SOURCE

OUT = Path(__file__).resolve().parent


def summarize(rows):
    losses = [r["loss"] for r in rows]
    wins_first = Counter(
        "safer_higher" if x["reward"] < x["best_viable_reward"] else
        "unsafe_higher" if x["reward"] > x["best_viable_reward"] else "tie" for x in losses)
    return dict(n=len(rows),outcomes=dict(Counter(r["outcome"] for r in rows)),
                median_loss_step=median(x["step"] for x in losses),
                loss_at_step1=sum(x["step"]==1 for x in losses),
                loss_within5=sum(x["step"]<=5 for x in losses),
                loss_within10=sum(x["step"]<=10 for x in losses),
                loss_before_service=sum(not x["after_service"] for x in losses),
                visits_either=sum(bool(r["refuels"] or r["repairs"]) for r in rows),
                visits_both=sum(bool(r["refuels"] and r["repairs"]) for r in rows),
                median_steps=median(r["steps"] for r in rows),
                median_unique_positions=median(r["unique_positions"] for r in rows),
                median_fraction_immediate_returns=median(r["immediate_returns"]/(r["steps"]-1) for r in rows),
                positive_reward_on_loss=sum(x["reward"]>0 for x in losses),
                local_reward_comparison=dict(wins_first),
                median_witness_return=median(r["witness_return"] for r in rows),
                median_policy_return=median(r["discounted_return"] for r in rows),
                failed_return_gt_witness=sum(r["discounted_return"]>r["witness_return"] for r in rows))


def main():
    output={}
    all_rows=[]
    maps=None
    for family in read(SOURCE/"frozen.json"):
        rows=read(OUT/(family+"-traces.json"))
        names={r["map"] for r in rows}
        assert len(rows)==len(names)==300 and (maps is None or names==maps)
        maps=names
        all_rows.extend(rows)
        controls=read(OUT/(family+"-controls.json"))
        assert set(controls)=={"fuel100","health100","both100","last5"}
        for records in controls.values():
            assert len(records)==300 and {r["map"] for r in records}==maps
        output[family]=dict(original=summarize(rows),control_wins={
            k:sum(r["outcome"]=="SUCCESS" for r in records) for k,records in controls.items()})
    output["pooled"]=summarize(all_rows)
    save(OUT/"trajectory-summary.json",output)
    for family,summary in output.items():
        print(family,summary)


if __name__=="__main__":
    main()
