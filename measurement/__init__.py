"""Measurement, raw-data storage and anomaly tracking for the iLoRA reproduction runs.

`schema`, `io` and `report` import no torch, so figures and reports can be rebuilt from a run
folder on a machine without a GPU or the model. `activation`, `anomaly`, `recorder` and `callback`
are the training-side parts. See experiments/IMPLEMENTATION_NOTES.md section 11j.
"""
