"""Solve Bernoulli's ODE using Euler's method."""

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


def euler(x0, b, y0, P, Q, n):
    """Solve Bernoulli with Euler's method."""
    x = torch.tensor([float(x0)], requires_grad=True)
    y = torch.tensor([float(y0)], requires_grad=True)
    h = 0.01
    xs = [x.detach().numpy()]
    ys = [y.detach().numpy()]
    while x < b:
        y = y - h * (P(x) * y - Q(x) * y**n)  # Euler's method
        x = x + h
        xs.append(x.detach().numpy())
        ys.append(y.detach().numpy())
    return xs, ys


def neural(a, b, x0, y0, P, Q, n):
    """Solve Bernoulli with neural network."""

    def cost(y_pred, x):
        dydx = torch.autograd.grad(
            y_pred, x, grad_outputs=torch.ones_like(y_pred), create_graph=True
        )[0]
        return ((dydx + P(x) * y_pred - Q(x) * y_pred**n) ** 2).mean()

    NUM_NEURONS = 200
    BATCH_SIZE = 1000
    nn = torch.nn.Sequential(
        torch.nn.Linear(1, NUM_NEURONS),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(NUM_NEURONS, NUM_NEURONS),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(NUM_NEURONS, NUM_NEURONS),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(NUM_NEURONS, 1),
    )
    optimizer = torch.optim.Adam(nn.parameters(), lr=0.001)

    def model(x):
        return (x - x0) * nn(x.unsqueeze(-1)).squeeze(-1) + y0

    losses = []
    for _ in tqdm(range(4000)):
        optimizer.zero_grad()

        x_batch = torch.linspace(a, b, BATCH_SIZE)
        x_batch.requires_grad = True

        y_pred = model(x_batch)

        loss = cost(y_pred, x_batch)
        loss.backward()

        optimizer.step()
        losses.append(loss.item())

    return model, losses


if __name__ == "__main__":
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title("Nike logo on reversed x axis")

    ########################################
    # P = lambda x: 2 / x
    # Q = lambda x: -x * x * torch.cos(x)
    # n = 2
    # x0 = 1
    # y0 = y_sol(torch.tensor(x0)).item()
    ########################################
    y_sol = lambda x: (torch.exp(x) * (1 - 3 * x)) ** (-1 / 3)
    a = -8
    b = 0.3
    P = lambda x: 1 / 3
    Q = lambda x: torch.exp(x)
    n = 4
    x0 = -5
    y0 = y_sol(torch.tensor(x0)).item()
    (analytical_chart,) = ax1.plot(
        torch.linspace(a, b, 100), y_sol(torch.linspace(a, b, 100))
    )
    ########################################
    # y_sol = lambda x: torch.exp(torch.cos(x)) / (
    #     torch.exp(torch.cos(torch.tensor(1)))
    #     + quad(
    #         lambda s: torch.cos(torch.tensor(s))
    #         * torch.exp(torch.cos(torch.tensor(s))),
    #         1,
    #         x,
    #     )[0]
    # )
    # P = lambda x: torch.sin(x)
    # Q = lambda x: -torch.cos(x)
    # n = 2
    # x0 = 1
    # y0 = y_sol(torch.tensor(x0)).item()
    # from scipy.integrate import quad

    # a = 0.3
    # b = 25
    # ax1.plot(
    #     torch.linspace(a, b, 100),
    #     [y_sol(torch.tensor(i)) for i in torch.linspace(a, b, 100)],
    # )
    ########################################

    xs, ys = euler(a, b, y_sol(torch.tensor(a)), P, Q, n)
    (euler_chart,) = ax1.plot(xs, ys)

    model, losses = neural(a, b, x0, y0, P, Q, n)
    (neural_chart,) = ax1.plot(
        torch.linspace(a, b, 100),
        [model(x).detach().numpy() for x in torch.linspace(a, b, 100)],
    )

    ax1.legend(
        [analytical_chart, euler_chart, neural_chart], ["Analytical", "Euler", "Neural"]
    )

    ax2.plot(losses)
    ax2.set_title("Is this loss?")
    ax2.set_yscale("log")

    plt.show()
