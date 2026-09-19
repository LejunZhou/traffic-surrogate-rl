`python run.py report` writes figures here after final evaluation.
The historical plotting workflow reports **return**, which is not identical to
full-episode TTS. Raw evaluation JSONL includes `tts_veh_h` for traffic comparisons.

Timing in `runs/commands.jsonl` is measured command elapsed time. Summed simulator
episode runtimes, reported as accounted runtime, are a different quantity.
