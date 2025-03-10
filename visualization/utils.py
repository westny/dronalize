from typing import Optional
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.collections import LineCollection
from torch_geometric.utils import subgraph


def plot_map_edges(map_pos: torch.Tensor, map_edge_index: torch.Tensor, map_edge_type: torch.Tensor) -> None:
    """Plot the map edges with appropriate styles based on edge type."""
    for i in range(map_edge_index.shape[1]):
        edge = map_edge_index[:, i]
        if map_edge_type[i] == 2:  # Road borders
            plt.plot(map_pos[edge, 0], map_pos[edge, 1],
                     color='gray',
                     linewidth=1,
                     zorder=1,
                     alpha=0.9,
                     linestyle='solid')
        elif map_edge_type[i] in (5, 6, 7, 8, 9):  # Lane markings
            plt.plot(map_pos[edge, 0], map_pos[edge, 1],
                     color='darkgray',
                     linewidth=0.5,
                     zorder=0,
                     alpha=0.6,
                     linestyle=(0, (5, 10)))


def get_agent_color(idx: int, ma_idx: Optional[torch.Tensor] = None) -> str:
    """Determine agent color based on index and type."""
    if ma_idx is None:
        # Random colors for non-ego vehicles when ma_idx not provided
        colors = [
            # Tableau colors
            'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
            'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',

            # Standard matplotlib colors
            'forestgreen', 'crimson', 'darkviolet', 'teal', 'sienna',
            'darkturquoise', 'darkkhaki', 'indianred', 'mediumorchid',

            # CSS-style colors
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD',
            '#D4A5A5', '#9B4F0F', '#C3447A', '#7ECEFD', '#98A8F8',

            # More vibrant colors
            '#FF1E1E', '#00FF7F', '#FF69B4', '#4B0082', '#FFD700',
            '#8B4513', '#FF4500', '#2E8B57', '#9932CC', '#008080',

            # Pastel colors
            '#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA', '#FFB3F7',
            '#C1B3FF', '#B3FFF6', '#FFE4B3', '#B3FFB3', '#FFC9B3'
        ]

        return random.choice(colors)

    if idx == 0:
        return 'tab:blue'  # ego vehicle
    elif idx in ma_idx:
        return 'tab:green'  # main agents
    return 'tab:red'  # other agents


def create_trajectory_fade(pos: torch.Tensor, color: str) -> tuple:
    """Create faded trajectory line segments."""
    linefade = colors.to_rgb(color) + (0.0,)
    color_rgba = colors.to_rgba(color)
    myfade = colors.LinearSegmentedColormap.from_list('my', [linefade, color_rgba])
    alphas = np.clip(np.exp(np.linspace(0, 1, pos.shape[0] - 1)) - 0.6, 0, 1)
    tmp = pos[:, :2][:, None, :]
    segments = np.hstack((tmp[:-1], tmp[1:]))
    return segments, alphas, myfade


def plot_agent_trajectories(ax: plt.Axes,
                            pos: torch.Tensor,
                            gt: torch.Tensor,
                            ma_idx: Optional[torch.Tensor] = None) -> None:
    """Plot agent trajectories with history and ground truth."""
    for i in range(pos.shape[0]):
        color = get_agent_color(i, ma_idx)

        # Plot trajectory history
        segments, alphas, myfade = create_trajectory_fade(pos[i], color)
        lc = LineCollection(segments, array=alphas, cmap=myfade, linewidth=5, zorder=0)
        ax.add_collection(lc)

        # Plot current position
        ax.scatter(pos[i, -1, 0], pos[i, -1, 1],
                   color=color,
                   marker='*',
                   s=100,
                   alpha=1.0)

        # Plot ground truth trajectory
        ax.plot(gt[i, :, 0], gt[i, :, 1],
                color=color,
                marker='.',
                markersize=10,
                linewidth=2,
                alpha=0.3)


def setup_plot_axes(ax: plt.Axes,
                    xlim: Optional[tuple] = None,
                    ylim: Optional[tuple] = None) -> None:

    """Configure plot axes settings."""
    ax.set_aspect('equal')
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    plt.axis('off')
    plt.tight_layout()


def plot_polar_connections(ax: plt.Axes,
                           rho: Optional[torch.Tensor] = None,
                           theta: Optional[torch.Tensor] = None,
                           ma_idx: Optional[torch.Tensor] = None) -> None:
    if rho is None or theta is None:
        return

    """Plot lines connecting positions to their polar coordinate endpoints."""
    for i in range(rho.shape[0]):
        color = get_agent_color(i, ma_idx)

        # For each timestep
        for t in range(rho.shape[1]):
            if not torch.isnan(rho[i, t, 0]):  # Check if position is valid
                # Current position

                # Convert polar to cartesian coordinates
                # theta is assumed to be in radians
                dx = rho[i, t] * torch.cos(theta[i, t])
                dy = rho[i, t] * torch.sin(theta[i, t])


                # Plot connection line
                ax.plot([0., dx.item()], [0., dy.item()],
                        color=color,
                        linestyle=':',
                        linewidth=1,
                        alpha=0.3,
                        zorder=0)


def plot_lane_displacement(ax: plt.Axes,
                           pos: torch.Tensor,
                           lane_disp: torch.Tensor,
                           ma_idx: Optional[torch.Tensor] = None,
                           arrow_scale: float = 10.0) -> None:
    """
    Plot arrows visualizing lane displacement values.

    Args:
        ax: Matplotlib axes
        pos: Agent positions tensor
        lane_disp: Lane displacement values (between -1 and 1)
        ma_idx: Indices of main agents
        arrow_scale: Scale factor for arrow size
    """
    for i in range(pos.shape[0])[:1]:
        color = get_agent_color(i, ma_idx)

        # For each timestep
        for t in range(pos.shape[1]):
            if not torch.isnan(pos[i, t, 0]):  # Check if position is valid
                # Current position
                x = float(pos[i, t, 0])
                y = float(pos[i, t, 1])

                # Lane displacement value (between -1 and 1)
                displacement = float(lane_disp[i, t, 0])

                if not np.isnan(displacement):
                    # We'll draw an arrow perpendicular to the travel direction
                    dx = 0
                    dy = displacement * arrow_scale  # Scale the displacement for visibility

                    # Draw the arrow
                    ax.arrow(x, y, dx, dy,
                             head_width=0.5,
                             head_length=0.3,
                             fc=color,
                             ec='k',
                             alpha=0.7,
                             zorder=100)


def visualize_batch(data: dict, batch_idx: int = 0,
                    use_ma_idx: bool = False, plot_supp: bool = False) -> None:
    """Visualize a complete traffic scene."""
    # Extract agent data for the specified batch
    batch = data['agent']['batch'] == batch_idx
    pos = data['agent']['inp_pos'][batch]
    gt = data['agent']['trg_pos'][batch]
    ma_mask = data['agent']['ma_mask'][batch]
    ma_idx = torch.where(ma_mask[:, 0])[0] if use_ma_idx else None

    # Clean position data
    pos_eq_zero = pos == 0
    pos_eq_zero[0] = False
    pos[pos_eq_zero] = float("nan")
    gt[gt == 0] = float("nan")

    r1: torch.Tensor | None = None
    r2: torch.Tensor | None = None
    if 'inp_r1' in data['agent'] and plot_supp:
        r1 = data['agent']['inp_r1'][batch]  # rho or lane displacement
        r2 = data['agent']['inp_r2'][batch]  # theta or road displacement

        if r1 is not None and r2 is not None:
            r1[pos_eq_zero[..., :1]] = float("nan")
            r2[pos_eq_zero[..., :1]] = float("nan")

    # Extract map data
    map_batch = data['map_point']['batch'] == batch_idx
    map_pos = data['map_point']['position'][map_batch]
    map_edge_index = data['map_point', 'to', 'map_point']['edge_index']
    map_edge_type = data['map_point', 'to', 'map_point']['type']

    # Process map edges
    map_edge_index, map_edge_type = subgraph(map_batch, map_edge_index,
                                             map_edge_type, relabel_nodes=True)

    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot elements
    plot_map_edges(map_pos, map_edge_index, map_edge_type)
    plot_agent_trajectories(ax, pos, gt, ma_idx)

    # Plot lane displacement if available
    if r1 is not None:
        plot_lane_displacement(ax, pos, r1, ma_idx, arrow_scale=10.0)

    # Plot polar connections if available
    if r1 is not None:
        plot_polar_connections(ax, r1, r2, ma_idx)

    setup_plot_axes(ax)

    plt.show()
