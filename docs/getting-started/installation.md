# Installation

PriceSentinel is a Python project managed with `uv`.

## Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (recommended)

## Steps

1. **Clone the repository**:

    ```sh
    git clone https://github.com/JDPS/pricesentinel.git
    cd pricesentinel
    ```

2. **Sync dependencies**:

    ```sh
    uv sync --all-extras
    ```

3. **Install pre-commit hooks**:

    ```sh
    uv run pre-commit install
    ```
