import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(rewards):
    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Learning Curve (Reward vs Episode)")
    plt.grid()
    plt.show()


def plot_episode_lengths(lengths):
    plt.figure()
    plt.plot(lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length vs Episode")
    plt.grid()
    plt.show()


def plot_confusion_matrix(cm):
    labels = ["Success", "Failure"]
    values = [cm["TP"], cm["FN"]]

    plt.figure()
    plt.bar(labels, values)
    plt.title("Success / Failure Distribution")
    plt.ylabel("Episodes")
    plt.show()
