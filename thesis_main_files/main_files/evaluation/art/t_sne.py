import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class TSNEVisualizer:
    def __init__(self, perplexity=30, learning_rate=200, n_components=2, random_state=42):
        self.tsne = TSNE(n_components=n_components,
                         perplexity=perplexity,
                         learning_rate=learning_rate,
                         random_state=random_state)

    def visualize(self, audio_embeddings, video_embeddings, draw_lines=True,
                  title="t-SNE of Audio and Video Embeddings", save_path=None, return_fig=False):
        """
        Visualizes aligned audio and video embeddings using t-SNE.

        Parameters:
        - audio_embeddings: np.array of shape (N, D)
        - video_embeddings: np.array of shape (N, D)
        - draw_lines: bool, whether to draw lines between corresponding pairs
        - title: str, plot title
        - save_path: str or None, if provided, saves the plot to this path
        - return_fig: bool, if True, returns the matplotlib Figure object

        Returns:
        - fig (matplotlib.figure.Figure) if return_fig is True, else None
        """

        assert audio_embeddings.shape == video_embeddings.shape, "Audio and video embeddings must have the same shape."

        N = audio_embeddings.shape[0]

        # Stack embeddings
        combined = np.vstack([audio_embeddings, video_embeddings])
        labels = ['audio'] * N + ['video'] * N

        # Run t-SNE
        reduced = self.tsne.fit_transform(combined)

        # Create figure and axes
        fig, ax = plt.subplots(figsize=(8, 6))

        for modality in ['audio', 'video']:
            idxs = [i for i, l in enumerate(labels) if l == modality]
            ax.scatter(reduced[idxs, 0], reduced[idxs, 1], label=modality, alpha=0.7)

        if draw_lines:
            for i in range(N):
                ax.plot([reduced[i, 0], reduced[i + N, 0]],
                        [reduced[i, 1], reduced[i + N, 1]],
                        color='gray', linewidth=0.5, alpha=0.4)

        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300)
            print(f"Plot saved to {save_path}")

        if return_fig:
            return fig

        plt.show()
        plt.close(fig)
