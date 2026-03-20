# Model Predictive Control

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library for formulating and solving Optimal Control Problems (OCP) and Model Predictive Control (MPC) using [CasADi](https://web.casadi.org/).

This package is designed to provide a modular, structured approach to defining dynamics, constraints, and objective functions, separating the problem setup from the solver execution for efficient repeated solves.

## Features

- **CasADi Backend:** Utilizes CasADi's `Opti` interface for efficient numerical optimization.
- **Strict Setup/Solve Separation:** Formulate the OCP once (`setup()`), then repeatedly execute the optimization (`solve(x0)`) for MPC applications.
- **Robust Validation:** Enforces explicit dimension validation for states, controls, and functions to catch errors early.
- **Multiple Integration Methods:** Supports continuous (Runge-Kutta 4, Forward Euler) and discrete dynamics.
- **Multiple Discretization Schemes:** Implements single shooting, multiple shooting, and Hermite-Simpson direct collocation.
- **Constructor Injection:** Mandates fully formed objects via constructor injection for dynamics, objectives, and constraints.

## Installation

### As a Package

You can install the package directly from the repository using `pip`:

```bash
pip install git+https://github.com/maxi-fr/model_predictive_control.git
```

### For Development

This project uses `uv` for dependency and project management. To set up the environment for development:

1. Clone the repository:
   ```bash
   git clone https://github.com/maxi-fr/model_predictive_control.git
   cd model_predictive_control
   ```
2. Sync the dependencies using `uv`:
   ```bash
   uv sync
   ```

## Examples

Usage examples are provided as Jupyter notebooks in the `examples/` directory.

- **Inverted Pendulum:** See [`examples/inverted_pendulum.ipynb`](examples/inverted_pendulum.ipynb) for a complete demonstration of formulating an OCP for an inverted pendulum, setting up the solver, and visualizing the results.

## Testing

The project uses `pytest` for unit testing. To run the tests:

```bash
uv run pytest
```

## License

This project is licensed under the MIT License.
