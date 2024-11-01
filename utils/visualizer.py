import matplotlib.pyplot as plt

class Visualizer:
    @staticmethod
    def plot_loss(losses, title="Training Loss"):
        plt.plot(losses)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(title)
        plt.show()
