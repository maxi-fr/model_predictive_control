from typing import Optional

import casadi as ca
import numpy as np
import pytest
import torch
from torch import nn

from learning.constraints import (
    LearnedConstraint,
    LearnedControlConstraint,
    LearnedStateConstraint,
)
from learning.dynamics import LearnedDynamics
from learning.objective import LearnedCostFunction


class DummyDynamics(nn.Module):
    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x + u


class DummyCost(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor | None = None,
        x_ref: torch.Tensor | None = None,
        u_ref: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cost = torch.sum(x**2)
        if u is not None:
            cost += torch.sum(u**2)
        if x_ref is not None:
            cost += torch.sum((x - x_ref) ** 2)
        if u_ref is not None and u is not None:
            cost += torch.sum((u - u_ref) ** 2)
        # Ensure 2D output
        return cost.view(1, 1)


class DummyConstraint(nn.Module):
    def forward(self, x: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        val = torch.sum(x)
        if u is not None:
            val += torch.sum(u)
        return val.view(1, 1)


def test_learned_dynamics() -> None:
    nx, nu = 2, 2
    model = DummyDynamics()
    dyn = LearnedDynamics(model, nx, nu)

    x_val = np.array([1.0, 2.0])
    u_val = np.array([3.0, 4.0])

    # PyTorch evaluation
    torch_out = model(torch.tensor(x_val), torch.tensor(u_val)).detach().numpy()

    # CasADi evaluation
    casadi_out = dyn(x_val, u_val).full().flatten()

    np.testing.assert_allclose(torch_out.flatten(), casadi_out, atol=1e-5)


def test_learned_cost_function() -> None:
    nx, nu = 2, 2
    model = DummyCost()
    cost = LearnedCostFunction(model, nx, nu, has_reference=False)

    x_val = np.array([1.0, 2.0])
    u_val = np.array([3.0, 4.0])

    # PyTorch evaluation
    torch_out = model(torch.tensor(x_val), torch.tensor(u_val)).detach().numpy()

    # CasADi evaluation
    casadi_out = cost(x_val, u_val).full().flatten()

    np.testing.assert_allclose(torch_out.flatten(), casadi_out, atol=1e-5)


def test_learned_cost_with_reference() -> None:
    nx, nu = 2, 2
    model = DummyCost()
    cost = LearnedCostFunction(model, nx, nu, has_reference=True)

    x_val = np.array([1.0, 2.0])
    u_val = np.array([3.0, 4.0])
    x_ref = np.array([0.5, 0.5])
    u_ref = np.array([1.0, 1.0])

    # PyTorch evaluation
    torch_out = (
        model(torch.tensor(x_val), torch.tensor(u_val), torch.tensor(x_ref), torch.tensor(u_ref)).detach().numpy()
    )

    # CasADi evaluation
    casadi_out = cost(x_val, u_val, x_ref, u_ref).full().flatten()

    np.testing.assert_allclose(torch_out.flatten(), casadi_out, atol=1e-5)


def test_learned_state_constraint() -> None:
    nx = 3
    model = DummyConstraint()
    con = LearnedStateConstraint(model, nx)

    x_val = np.array([1.0, 2.0, 3.0])

    # PyTorch evaluation
    torch_out = model(torch.tensor(x_val)).detach().numpy()

    # CasADi evaluation
    casadi_out = con.f(x_val).full().flatten()

    np.testing.assert_allclose(torch_out.flatten(), casadi_out, atol=1e-5)


def test_learned_control_constraint() -> None:
    nu = 2
    # Reuse DummyConstraint logic by passing u into x's argument slot
    model = DummyConstraint()
    con = LearnedControlConstraint(model, nu)

    u_val = np.array([1.0, 2.0])

    # PyTorch evaluation
    torch_out = model(torch.tensor(u_val)).detach().numpy()

    # CasADi evaluation
    casadi_out = con.f(u_val).full().flatten()

    np.testing.assert_allclose(torch_out.flatten(), casadi_out, atol=1e-5)


def test_learned_mixed_constraint() -> None:
    nx, nu = 2, 3
    model = DummyConstraint()
    con = LearnedConstraint(model, nx, nu)

    x_val = np.array([1.0, 2.0])
    u_val = np.array([3.0, 4.0, 5.0])

    # PyTorch evaluation
    torch_out = model(torch.tensor(x_val), torch.tensor(u_val)).detach().numpy()

    # CasADi evaluation
    casadi_out = con.f(x_val, u_val).full().flatten()

    np.testing.assert_allclose(torch_out.flatten(), casadi_out, atol=1e-5)
