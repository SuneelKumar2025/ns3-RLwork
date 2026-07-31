#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose whether negative-reward stretches are caused by a structurally
unreachable SLA (both links saturated) rather than by policy mistakes.

Reconstructs, from the saved step-level training CSV, which branch of
MyGetReward() each step actually fell into, and reports what fraction of
steps had NO possible +5 outcome available at all (both links >= SLA)
versus steps where the agent was on the wrong side of a reachable choice.

Usage:
    python3 diagnose_reward_ceiling.py pg_training_step_metrics.csv
"""

import sys
import numpy as np
import pandas as pd

# --- Must match sim.cc's MyGetReward exactly ---
SLA = 300.0
DEADBAND = 20.0

path = sys.argv[1] if len(sys.argv) > 1 else "pg_training_step_metrics.csv"
df = pd.read_csv(path)
df.columns = [c.strip() for c in df.columns]

active = df["active_delay_ms"].to_numpy()
l1 = df["l1_delay_ms"].to_numpy()
l2 = df["l2_delay_ms"].to_numpy()
selected = df["selected_link"].to_numpy()

inactive = np.where(selected == 0, l2, l1)

both_over_sla = (active >= SLA) & (inactive >= SLA)
under_sla = active < SLA
within_deadband = np.abs(active - inactive) <= DEADBAND
worse_than_other = active > inactive + DEADBAND
better_than_other = (active < inactive - DEADBAND) & ~under_sla  # case 4 territory

n = len(df)
print(f"Total steps analyzed: {n}\n")

print("=== Reachability breakdown ===")
print(f"Active link under SLA (case 1, reward=+5)      : {under_sla.mean()*100:6.2f}%  ({under_sla.sum()})")
print(f"Both links >= SLA  -> +5 IMPOSSIBLE this step   : {both_over_sla.mean()*100:6.2f}%  ({both_over_sla.sum()})")
print()

print("=== Among steps where active link is >= SLA (ceiling is 0, not +5) ===")
over_sla_mask = ~under_sla
n_over = over_sla_mask.sum()
if n_over > 0:
    db = (within_deadband & over_sla_mask).sum()
    worse = (worse_than_other & over_sla_mask).sum()
    better = (better_than_other & over_sla_mask).sum()
    print(f"  Within deadband (reward=0, correct/no-op)     : {db/n_over*100:6.2f}%  ({db})")
    print(f"  Worse than other link (reward=-5)             : {worse/n_over*100:6.2f}%  ({worse})")
    print(f"  Better than other, still >SLA (reward 0/-2)   : {better/n_over*100:6.2f}%  ({better})")
print()

print("=== Interpretation ===")
print(f"{both_over_sla.mean()*100:.1f}% of all steps had BOTH links over SLA simultaneously.")
print("In those steps, +5 was mathematically unreachable regardless of policy quality —")
print("the best possible outcome was 0 (deadband) or avoiding -5 (picking the lesser-bad link).")
print()
if n_over > 0:
    mistake_rate = worse / n_over * 100
    print(f"Of steps where +5 wasn't reachable, {mistake_rate:.1f}% still landed on -5")
    print("(agent on the worse link by more than the deadband). If this rate is low")
    print("(e.g. under ~15-20%), most of your negative reward is coming from the")
    print("SLA-unreachable regime itself, not from policy error — supporting your hypothesis.")
    print("If this rate stays high even late in training, some of it is still real mistakes.")