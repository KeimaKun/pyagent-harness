---
name: python_data_analysis
description: Skill for CSV/JSON data analysis workflows in Python — covering file I/O, statistical computation, and structured report generation.
tier: core
version: "1.0"
tags: [csv, json, pandas, statistics, reporting]
---

# Python Data Analysis Skill

## Overview

This skill provides patterns and best practices for:
- Generating synthetic tabular datasets as CSV files
- Parsing and validating CSV data using the standard library or `csv` module
- Computing descriptive statistics (mean, median, std, min, max)
- Writing structured summary reports as JSON

---

## Accepted Tool Calls

### `generate_csv`
Generates a CSV file at a given path with specified column names and row count.

```json
{"tool": "generate_csv", "args": {"path": "output.csv", "columns": ["col1", "col2"], "rows": 10}}
```

### `parse_csv`
Parses a CSV file and returns column-wise data as a dict of lists.

```json
{"tool": "parse_csv", "args": {"path": "output.csv"}}
```

### `compute_stats`
Computes mean, median, std, min, max for each numeric column.

```json
{"tool": "compute_stats", "args": {"data": {"col1": [1.0, 2.0], "col2": [3.0, 4.0]}}}
```

### `write_json_report`
Writes a summary dict to a JSON file.

```json
{"tool": "write_json_report", "args": {"report": {}, "path": "summary.json"}}
```

---

## Implementation Patterns

### CSV Generation (stdlib only — no pandas required)
```python
import csv
import random

columns = ["account_id", "balance", "transactions", "credit_score", "loan_amount",
           "interest_rate", "days_overdue", "region_code", "product_type", "risk_flag"]

with open("banking_metrics.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    for i in range(10):
        writer.writerow({
            "account_id": f"ACC{i:04d}",
            "balance": round(random.uniform(100, 50000), 2),
            "transactions": random.randint(1, 200),
            "credit_score": random.randint(300, 850),
            "loan_amount": round(random.uniform(0, 100000), 2),
            "interest_rate": round(random.uniform(1.5, 25.0), 2),
            "days_overdue": random.randint(0, 90),
            "region_code": random.choice(["NA", "EU", "AP", "LA"]),
            "product_type": random.choice(["savings", "checking", "loan", "credit"]),
            "risk_flag": random.choice([0, 1]),
        })
```

### Column Average Computation (stdlib only)
```python
import csv

def column_averages(path: str) -> dict:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    averages = {}
    for col in rows[0].keys():
        try:
            values = [float(row[col]) for row in rows]
            averages[col] = round(sum(values) / len(values), 4)
        except (ValueError, ZeroDivisionError):
            averages[col] = None  # non-numeric column
    return averages
```

### JSON Report Writing
```python
import json

def write_report(data: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
```

---

## Acceptance Criteria

A task using this skill is considered complete when:
- [ ] The CSV file exists on disk
- [ ] The CSV file has exactly the expected number of data rows (excluding header)
- [ ] The JSON report file exists on disk
- [ ] The JSON report file is > 0 bytes
- [ ] All numeric column averages in the report are valid floats (not None)

---

## Safety Notes

- Do NOT use `eval()` or `exec()` to parse CSV data
- Do NOT use `os.system()` or `subprocess` for file operations
- Prefer the `csv` standard library module over third-party parsers
- File paths MUST be relative to the project working directory
