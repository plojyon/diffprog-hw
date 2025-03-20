import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


def gauss_seidel(X, u0, u1, v0, v1):
    """Solve Laplace with Gauss-Seidel method."""

    N = torch.tensor(X.shape[0])
    iterations = int(-8 / (2 * torch.log(1 - 2 * torch.sin(torch.pi / (2 * N)) ** 2)))

    grid = torch.zeros((1, 1, N, N))
    grid[0, 0, 0, :] = u0(X)
    grid[0, 0, -1, :] = u1(X)
    grid[0, 0, :, 0] = v0(X)
    grid[0, 0, :, -1] = v1(X)
    kernel = torch.tensor([[[[0, 1, 0], [1, 0, 1], [0, 1, 0]]]]) / 4

    for _ in range(iterations):
        grid[0, 0, 1:-1, 1:-1] = torch.nn.functional.conv2d(grid, kernel)
    return grid[0, 0]


def neural(X, u0, u1, v0, v1):
    """Solve Laplace with neural network."""

    def cost(y_pred, X):
        f_prime = torch.autograd.grad(
            y_pred, X, grad_outputs=torch.ones_like(y_pred), create_graph=True
        )[0]
        f_2prime = torch.autograd.grad(
            f_prime, X, grad_outputs=torch.ones_like(f_prime), create_graph=True
        )[0]

        return torch.nn.functional.mse_loss(f_2prime.sum(1), torch.zeros(X.shape[0]))

    NUM_NEURONS = 200
    BATCH_SIZE = 100
    nn = torch.nn.Sequential(
        torch.nn.Linear(2, NUM_NEURONS),
        torch.nn.Tanh(),
        torch.nn.Linear(NUM_NEURONS, NUM_NEURONS),
        torch.nn.Tanh(),
        torch.nn.Linear(NUM_NEURONS, 1),
    )
    optimizer = torch.optim.Adam(nn.parameters(), lr=0.001)

    def model(X):
        h1 = lambda x, y: (1 - x) * u0(y) + x * u1(y) + (1 - y) * v0(x) + y * v1(x)
        h2 = lambda x, y: x * (1 - x) * y * (1 - y)
        return h1(X[:, 0], X[:, 1]).unsqueeze(1) + nn(X) * h2(
            X[:, 0], X[:, 1]
        ).unsqueeze(1)

    losses = []
    for _ in tqdm(range(150)):
        optimizer.zero_grad()
        avg_loss = 0

        for j in range(0, x.shape[0], BATCH_SIZE):
            batch = X[j : j + BATCH_SIZE]
            batch.grad = None
            optimizer.zero_grad()
            y_pred = model(batch)
            loss = cost(y_pred, batch)

            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

        losses.append(avg_loss / (x.shape[0] // BATCH_SIZE))
        visualize_field(train_X, model(train_X), filename=f"animation/{_}.png")

    return model, losses


def visualize_field(X, Y, filename=None):
    dx = torch.autograd.grad(Y, X, grad_outputs=torch.ones_like(Y), create_graph=True)[
        0
    ][:, 0]
    dy = torch.autograd.grad(Y, X, grad_outputs=torch.ones_like(Y), create_graph=True)[
        0
    ][:, 1]

    fig, ax = plt.subplots()
    ax.streamplot(
        y.reshape(N, N).detach().cpu().numpy(),
        x.reshape(N, N).detach().cpu().numpy(),
        dx.detach().cpu().numpy().reshape(N, N),
        dy.detach().cpu().numpy().reshape(N, N),
        density=2,
    )

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


def visualize_gs(X, Y):
    kernel = (
        torch.tensor([[[[0, 1, 0], [1, 0, 1], [0, 1, 0]]]], dtype=torch.float32) / 4
    )
    Y = Y.unsqueeze(0).unsqueeze(0)
    dx = torch.nn.functional.conv2d(Y, kernel)

    fig, ax = plt.subplots()
    ax.streamplot(
        y.reshape(N, N).detach().cpu().numpy(),
        x.reshape(N, N).detach().cpu().numpy(),
        dx.detach().numpy().reshape(N, N),
        dy.detach().numpy().reshape(N, N),
        density=2,
    )
    plt.show()


N = 20
x = torch.linspace(0, 1, N, requires_grad=True)
x, y = torch.meshgrid(x, x)
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
train_X = torch.cat((x, y), 1)


# def y_sol(x, y):
#     twopi = torch.tensor(2 * torch.pi)
#     return torch.sin(twopi * x) * (
#         torch.cosh(twopi * y)
#         + ((1 - torch.sinh(twopi)) / torch.cosh(twopi)) * torch.sinh(twopi * y)
#     )


# y_analytical = y_sol(x, y).detach().numpy().reshape(N, N)

zero = lambda _: 0
cos2pi = lambda x: torch.cos(2 * torch.pi * x)

# Visualize neural
model, loss_history = neural(train_X, zero, zero, cos2pi, cos2pi)

plt.plot(loss_history)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()

visualize_field(train_X, model(train_X))


# # Visualize numerical
# X = torch.linspace(0, 1, 100)
# numeric = gauss_seidel(X, zero, zero, sin2pi, sin2pi)
# visualize_gs(X, numeric)
