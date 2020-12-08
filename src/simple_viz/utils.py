import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm


def save_training_batch(train_examples, solver):
    """Save training images as gif
        Reference: https://ndres.me/post/matplotlib-animated-gifs-easily/
    """

    if not os.path.exists("train_pics"):
        os.makedirs('train_pics')
    cwd = os.path.abspath('')
    filename = cwd + "/train_pics/train.gif"
    images = []

    for ex, i in train_examples:
        ex = ex.detach().numpy()
        if solver.DC:
            ex = ex[0][0]
            x, y = np.mgrid[-2:2:.0625, -2:2:.0625]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title("Generated Data - Epoch: %d" % i)
            ax.scatter(x, y, ex, cmap=cm.jet, linewidth=0)
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(image)
            plt.close()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title("Generated Data - Epoch: %d" % i)
            ax.scatter(ex[:, 0], ex[:, 1], ex[:, 2], cmap=cm.jet, linewidth=0)
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(image)
            plt.close()

    kwargs_write = {'fps':2.0, 'quantizer':'nq'}
    imageio.mimsave(filename, images, fps=1)


def plot_training_result(solver) -> None:
    """Plot generator and discriminator loss and gradient norms from training
       This is all really ugly and hacked together, so go look at other stuff.
    """
    gen_vals, disc_vals, train_ex = solver.get_plot_values()
    gen_loss, gen_grads = gen_vals
    disc_loss, disc_grads = disc_vals

    # Generate Data
    with torch.no_grad():
        noise = solver.get_noise(3)
        generated_data = np.asarray(solver.generator(noise).cpu())  # 3 x 1 x ...

    # Plot Generated data
    fig0 = plt.figure()
    if solver.DC:
        x, y = np.mgrid[-2:2:.0625, -2:2:.0625]
        ax0 = fig0.add_subplot(321, projection='3d')
        ax0.plot_surface(x, y, generated_data[0][0])
        ax1 = fig0.add_subplot(322, projection='3d')
        ax1.scatter(x, y, generated_data[0][0])
        ax2 = fig0.add_subplot(323, projection='3d')
        ax2.plot_surface(x, y, generated_data[1][0])
        ax3 = fig0.add_subplot(324, projection='3d')
        ax3.scatter(x, y, generated_data[1][0])
        ax4 = fig0.add_subplot(325, projection='3d')
        ax4.plot_surface(x, y, generated_data[2][0])
        ax5 = fig0.add_subplot(326, projection='3d')
        ax5.scatter(x, y, generated_data[2][0])

    else:
        ax0 = fig0.add_subplot(311, projection='3d')
        ax0.scatter(generated_data[0][:, 0], generated_data[0][:, 1], generated_data[0][:, 2])
        ax1 = fig0.add_subplot(312, projection='3d')
        ax1.scatter(generated_data[1][:, 0], generated_data[1][:, 1], generated_data[1][:, 2])
        x, y, z = generated_data[1][:, 0], generated_data[1][:, 1], generated_data[1][:, 2]

        ax2 = fig0.add_subplot(313, projection='3d')
        ax2.plot_trisurf(x, y, z, cmap=cm.coolwarm)


    # Plot losses and gradient norms
    fig1, ax1 = plt.subplots(2, 2)

    ax1[0][0].set_title("Generator Loss")
    ax1[0][0].plot(np.linspace(start=0, stop=len(gen_loss), num=len(gen_loss)), gen_loss)

    ax1[0][1].set_title("Discrim Loss")
    ax1[0][1].plot(np.linspace(start=0, stop=len(disc_loss), num=len(disc_loss)), disc_loss)

    ax1[1][0].set_title("Generator Gradient Norms")
    ax1[1][0].plot(np.linspace(start=0, stop=len(gen_grads), num=len(gen_grads)), gen_grads, c='c')

    ax1[1][1].set_title("Discrim. Gradient Norms")
    ax1[1][1].plot(np.linspace(start=0, stop=len(disc_grads), num=len(disc_grads)), disc_grads, c='c')

    plt.show()

    save_training_batch(train_ex, solver)

def plot_data(solver) -> None:
    # Generate Data
    with torch.no_grad():
        noise = solver.get_noise(3)
        generated_data = np.asarray(solver.generator(noise).cpu())  # 3 x 1 x ...

    # Plot Generated data
    fig0 = plt.figure()
    if solver.DC:
        x, y = np.mgrid[-2:2:.0625, -2:2:.0625]
        ax0 = fig0.add_subplot(321, projection='3d')
        ax0.plot_surface(x, y, generated_data[0][0])
        ax1 = fig0.add_subplot(322, projection='3d')
        ax1.scatter(x, y, generated_data[0][0])
        ax2 = fig0.add_subplot(323, projection='3d')
        ax2.plot_surface(x, y, generated_data[1][0])
        ax3 = fig0.add_subplot(324, projection='3d')
        ax3.scatter(x, y, generated_data[1][0])
        ax4 = fig0.add_subplot(325, projection='3d')
        ax4.plot_surface(x, y, generated_data[2][0])
        ax5 = fig0.add_subplot(326, projection='3d')
        ax5.scatter(x, y, generated_data[2][0])

    else:
        ax0 = fig0.add_subplot(311, projection='3d')
        ax0.scatter(generated_data[0][:, 0], generated_data[0][:, 1], generated_data[0][:, 2])
        ax1 = fig0.add_subplot(312, projection='3d')
        ax1.scatter(generated_data[1][:, 0], generated_data[1][:, 1], generated_data[1][:, 2])
        x, y, z = generated_data[1][:, 0], generated_data[1][:, 1], generated_data[1][:, 2]

        ax2 = fig0.add_subplot(313, projection='3d')
        ax2.plot_trisurf(x, y, z, cmap=cm.coolwarm)
    plt.show()
