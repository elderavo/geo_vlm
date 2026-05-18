"""Learning-rate and weight-decay schedules."""

from __future__ import annotations

import math


class WarmupCosineSchedule:
    def __init__(self, optimizer, warmup_steps, start_lr, ref_lr, t_max, final_lr=0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.t_max = max(1, t_max - warmup_steps)
        self.step_count = 0.0

    def step(self) -> float:
        self.step_count += 1
        if self.step_count < self.warmup_steps:
            progress = float(self.step_count) / float(max(1, self.warmup_steps))
            lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            progress = float(self.step_count - self.warmup_steps) / float(self.t_max)
            lr = max(
                self.final_lr,
                self.final_lr
                + (self.ref_lr - self.final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)),
            )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr


class CosineWDSchedule:
    def __init__(self, optimizer, ref_wd, final_wd, t_max):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.t_max = max(1, t_max)
        self.step_count = 0.0

    def step(self) -> float:
        self.step_count += 1
        progress = self.step_count / self.t_max
        wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
        if self.final_wd <= self.ref_wd:
            wd = max(self.final_wd, wd)
        else:
            wd = min(self.final_wd, wd)
        for group in self.optimizer.param_groups:
            if not group.get("wd_exclude", group.get("WD_exclude", False)):
                group["weight_decay"] = wd
        return wd
