import json

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
from rationals import RationalsModel

device = "cuda" if torch.cuda.is_available() else "cpu"


def store_coefficients(func_name, coeff_numerator, coeff_denominator):
    """
    The store_coefficients function takes the coefficients of a rational function
    and stores them in a JSON file. The name of the JSON file is based on the name
    of the function that was used to generate it. This allows us to store multiple
    trained models for different functions, and then load them later.

    :param self: Refer to the object itself
    :return: A dictionary with the function name and coefficients of numerator and
    denominator
    """
    coefficients_dict = {
        "function": func_name,
        "coeff_numerator": coeff_numerator.tolist(),
        "coeff_denominator": coeff_denominator.tolist(),
    }
    with open(f"rational_trained_models/coeff_{func_name}.json", "w") as file:
        json.dump(coefficients_dict, file, indent=1, separators=(", ", " : "))


def visualize():
    """
    The visualize function is used to train a rational function on the sinus function.
    The trained model is then compared with the true sinus and the start model (gelu).
    The training process can be seen in an animation.

    :return: A gif of the training process
    """
    rat = RationalsModel(n=5, m=5, function="gelu", use_coefficients=True)
    interval = 10000

    space = torch.linspace(-2, 2, interval)
    y_pred = rat.forward(space)

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(rat.parameters(), lr=0.1)
    start_relu = y_pred.cpu().detach().numpy().copy()
    plt.plot(space, start_relu, label=f"gelu (start function)")
    plt.legend()

    fig, ax = plt.subplots()
    predictions = []
    epoch_values = []

    def update_plot(epoch):
        ax.clear()
        ax.plot(space, predictions[epoch], label=f"trained sin")
        ax.plot(space, start_relu, label=f"gelu (start function)")
        ax.plot(space, torch.sin(space).cpu().detach().numpy(), label=f"true sin")
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Epoch {epoch_values[epoch]}")

    for epoch in range(1000):
        inp = ((torch.rand(interval) - 0.5) * 5).to(device)
        exp = torch.sin(inp)
        optimizer.zero_grad()
        out = rat(inp)
        loss = criterion(out, exp)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch} : loss = {loss}")
            predictions.append(rat.forward(space).cpu().detach().numpy())
            epoch_values.append(epoch)

    store_coefficients("sin", rat.coeff_numerator, rat.coeff_denominator)
    trained_sin = rat.forward(space)
    print(f"Start animation")
    ani = animation.FuncAnimation(
        fig, update_plot, frames=len(epoch_values), interval=1
    )
    print(f"Save animation")
    ani.save(
        f"rational_trained_models/comparison_function.gif",
        writer="ffmpeg",
        fps=30,
    )
    plt.close()

    plt.plot(space, trained_sin.cpu().detach().numpy(), label=f"trained sin")
    plt.plot(space, torch.sin(space).cpu().detach().numpy(), label=f"true sin")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    visualize()
