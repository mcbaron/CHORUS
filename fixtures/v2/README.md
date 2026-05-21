# CHORUS v2 Fixtures

These fixtures are generated from the Python v2 reference implementation and define the expected Rust v3 DSP behavior.

Regenerate:

```bash
poetry run python scripts/generate_v2_fixtures.py
```

Rust bypass fixtures must match within `bypass_max_abs`. Filtered fixtures may use `filtered_max_abs` because Rust and Python filter implementations can differ slightly in floating-point order.
