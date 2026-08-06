import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import yaml

font_size = 9
fig_width = 3.25
fig_height = 3.25


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'single-static.yml')
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_ideal_position(config):
    ue_variant = 'ue5' if config.get('carla_key', 'carla_ue4') == 'carla_ue5' else 'ue4'
    pos = config['ideal_position'][ue_variant]
    return pos['x'], pos['y'], pos['z']


def plot_initial_pos(run_folder, initial_positions, ideal_x, ideal_y, ideal_z, range_offset):
    initial_x, initial_y, initial_z = zip(*initial_positions)

    plt.figure(figsize=(fig_width, fig_height))
    scatter = plt.scatter(
        initial_x, initial_y,
        c=abs(np.array(initial_z) - ideal_z),
        cmap='Blues',
        s=10,
        edgecolor='black',
        linewidth=0.5,
        label='Initial Positions'
    )

    plt.text(ideal_x, ideal_y, 'H', fontsize=font_size, fontweight='bold',
             color='red', ha='center', va='center', label='Ideal Landing Position')

    cbar = plt.colorbar(scatter, label='Height from Landing Pad (m)', orientation='vertical')
    cbar.ax.set_ylabel('Height from Landing Pad (m)', fontsize=font_size)
    cbar.ax.tick_params(labelsize=9)

    ax = plt.gca()
    filled_rect = Rectangle((ideal_x - 4, ideal_y - 4), 8, 8,
                             linewidth=0, edgecolor='none', facecolor='red', alpha=0.25)
    ax.add_patch(filled_rect)

    plt.title("Initial QuadPlane Positions", fontsize=font_size, color='black')
    plt.xlabel("X Position (m)", fontsize=font_size, color='black')
    plt.ylabel("Y Position (m)", fontsize=font_size, color='black')
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    plt.grid(True, color='black', linestyle='--', linewidth=0.5)
    plt.xlim(ideal_x - range_offset[0], ideal_x + range_offset[0])
    plt.ylim(ideal_y - range_offset[1], ideal_y + range_offset[1])
    plt.tight_layout()

    plt.savefig(f"{run_folder}/initial_positions.PNG", dpi=300)
    plt.close()


def _style_3d_ax(ax, ideal_x, ideal_y, ideal_z, data_x, data_y, data_z, pad_xy=15, pad_z=10):
    """Tight axis limits around the data, fewer ticks, clean panes."""
    xlo, xhi = min(data_x.min(), ideal_x) - pad_xy, max(data_x.max(), ideal_x) + pad_xy
    ylo, yhi = min(data_y.min(), ideal_y) - pad_xy, max(data_y.max(), ideal_y) + pad_xy
    zlo = min(data_z.min(), ideal_z) - pad_z
    zhi = max(data_z.max(), ideal_z) + pad_z
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_zlim(zlo, zhi)

    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.zaxis.set_major_locator(plt.MaxNLocator(5))

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#cccccc')
    ax.yaxis.pane.set_edgecolor('#cccccc')
    ax.zaxis.pane.set_edgecolor('#cccccc')
    ax.grid(True, linewidth=0.3, alpha=0.5)


def plot_initial_pos_3d(run_folder, initial_positions, results, ideal_x, ideal_y, ideal_z, range_offset, z_min_offset=35, z_range=120):
    ix, iy, iz = zip(*initial_positions)
    ix, iy, iz = np.array(ix), np.array(iy), np.array(iz)
    success = np.array([r == 'Success' for r in results])
    n_success = success.sum()
    n_total = len(results)

    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection='3d')

    z_floor = min(iz.min(), ideal_z) - 10

    from matplotlib.patches import FancyBboxPatch
    import mpl_toolkits.mplot3d.art3d as art3d
    pad_size = 6
    pad_rect = Rectangle((ideal_x - pad_size/2, ideal_y - pad_size/2), pad_size, pad_size,
                          facecolor='red', alpha=0.4, edgecolor='darkred', linewidth=1.2)
    ax.add_patch(pad_rect)
    art3d.pathpatch_2d_to_3d(pad_rect, z=z_floor, zdir='z')
    ax.text(ideal_x, ideal_y, z_floor, 'H', fontsize=10, fontweight='bold',
            color='darkred', ha='center', va='center', zorder=11)
    ax.plot([ideal_x, ideal_x], [ideal_y, ideal_y], [z_floor, z_floor],
            'none', label='Landing Pad (H)')

    ax.scatter(ix[~success], iy[~success], iz[~success],
               c='#d62728', s=50, marker='x', linewidths=1.5,
               depthshade=False, alpha=0.85,
               label=f'Fail ({n_total - n_success}/{n_total})')
    ax.scatter(ix[success], iy[success], iz[success],
               c='#2ca02c', s=55, marker='o', edgecolor='black', linewidth=0.5,
               depthshade=False, alpha=0.95,
               label=f'Success ({n_success}/{n_total})')

    for xi, yi, zi, s in zip(ix, iy, iz, success):
        color = '#2ca02c' if s else '#d62728'
        ax.plot([xi, xi], [yi, yi], [z_floor, zi],
                color=color, linewidth=0.4, alpha=0.3, linestyle=':')

    ax.scatter(ix, iy, np.full_like(ix, z_floor),
               c='gray', s=6, alpha=0.2, marker='.')

    ax.set_xlabel('X (m)', fontsize=font_size + 1, labelpad=10)
    ax.set_ylabel('Y (m)', fontsize=font_size + 1, labelpad=10)
    ax.set_zlabel('Z (m)', fontsize=font_size + 1, labelpad=10)
    ax.set_title(f'Initial Positions — {n_success}/{n_total} succeeded ({100*n_success/n_total:.0f}%)',
                 fontsize=font_size + 2, fontweight='bold', pad=15)

    _style_3d_ax(ax, ideal_x, ideal_y, ideal_z, ix, iy, iz)
    ax.tick_params(labelsize=8, pad=3)
    ax.view_init(elev=28, azim=-45)
    ax.legend(fontsize=8, loc='upper left', framealpha=0.9)

    plt.savefig(f"{run_folder}/initial_positions_3d.PNG", dpi=300, bbox_inches='tight')
    plt.close()


def plot_final_pos_3d(run_folder, final_positions, results, ideal_x, ideal_y, ideal_z, range_offset):
    fx, fy, fz = zip(*final_positions)
    fx, fy, fz = np.array(fx), np.array(fy), np.array(fz)
    success = np.array([r == 'Success' for r in results])
    n_success = success.sum()
    n_total = len(results)

    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection='3d')

    # XY distance from pad for each trial
    dist_xy = np.sqrt((fx - ideal_x)**2 + (fy - ideal_y)**2)

    tol_x, tol_y, tol_z = 4, 4, 3

    fig, (ax_side, ax_top) = plt.subplots(1, 2, figsize=(9, 4),
                                           gridspec_kw={'width_ratios': [1.3, 1]})

    # ── Left panel: side view (XY distance vs Altitude) ──
    ax_side.axhspan(ideal_z - tol_z, ideal_z + tol_z, color='green', alpha=0.08)
    ax_side.axhline(ideal_z, color='darkred', linewidth=1.2, linestyle='-', alpha=0.6, label=f'Pad altitude (z={ideal_z}m)')
    ax_side.axhline(ideal_z + tol_z, color='green', linewidth=0.8, linestyle='--', alpha=0.5, label=f'Tolerance (z={ideal_z + tol_z}m)')

    # Stem lines from pad altitude to each point
    for d, zi, s in zip(dist_xy, fz, success):
        color = '#2ca02c' if s else '#d62728'
        ax_side.plot([d, d], [ideal_z, zi], color=color, linewidth=1.5, alpha=0.5)

    ax_side.scatter(dist_xy[~success], fz[~success],
                    c='#d62728', s=70, marker='x', linewidths=2, zorder=5,
                    label=f'Fail ({(~success).sum()}/{n_total})')
    ax_side.scatter(dist_xy[success], fz[success],
                    c='#2ca02c', s=70, marker='o', edgecolor='black', linewidth=0.5, zorder=5,
                    label=f'Success ({n_success}/{n_total})')

    ax_side.set_xlabel('XY distance from pad (m)', fontsize=font_size + 1)
    ax_side.set_ylabel('Final altitude (m)', fontsize=font_size + 1)
    ax_side.set_title(f'Final Positions — {n_success}/{n_total} succeeded ({100*n_success/n_total:.0f}%)',
                      fontsize=font_size + 2, fontweight='bold')
    ax_side.legend(fontsize=7.5, loc='upper right', framealpha=0.9)
    ax_side.grid(True, linewidth=0.3, alpha=0.5)
    ax_side.tick_params(labelsize=8)

    # ── Right panel: top-down XY view ──
    tol_rect = Rectangle((ideal_x - tol_x, ideal_y - tol_y), 2*tol_x, 2*tol_y,
                          facecolor='green', alpha=0.1, edgecolor='green',
                          linewidth=1, linestyle='--', label='XY tolerance')
    ax_top.add_patch(tol_rect)
    pad_rect = Rectangle((ideal_x - 3, ideal_y - 3), 6, 6,
                          facecolor='red', alpha=0.3, edgecolor='darkred', linewidth=1)
    ax_top.add_patch(pad_rect)
    ax_top.text(ideal_x, ideal_y, 'H', fontsize=11, fontweight='bold',
                color='darkred', ha='center', va='center', zorder=11)

    ax_top.scatter(fx[~success], fy[~success],
                   c='#d62728', s=70, marker='x', linewidths=2, zorder=5)
    ax_top.scatter(fx[success], fy[success],
                   c='#2ca02c', s=70, marker='o', edgecolor='black', linewidth=0.5, zorder=5)

    # Annotate z on each point
    for xi, yi, zi, s in zip(fx, fy, fz, success):
        ax_top.annotate(f'{zi:.0f}m', (xi, yi), fontsize=6.5,
                        textcoords='offset points', xytext=(5, 4),
                        color='#2ca02c' if s else '#d62728', fontweight='bold')

    ax_top.set_xlabel('X (m)', fontsize=font_size + 1)
    ax_top.set_ylabel('Y (m)', fontsize=font_size + 1)
    ax_top.set_title('Top-down view (z annotated)', fontsize=font_size + 1, fontweight='bold')
    ax_top.set_aspect('equal')
    ax_top.legend(fontsize=7, loc='upper right', framealpha=0.9)
    ax_top.grid(True, linewidth=0.3, alpha=0.5)
    ax_top.tick_params(labelsize=8)
    margin = max(dist_xy.max(), 10) + 5
    ax_top.set_xlim(ideal_x - margin, ideal_x + margin)
    ax_top.set_ylim(ideal_y - margin, ideal_y + margin)

    plt.tight_layout()

    plt.savefig(f"{run_folder}/final_positions_3d.PNG", dpi=300, bbox_inches='tight')
    plt.close()


def plot_final_pos(run_folder, final_positions, ideal_x, ideal_y, ideal_z, range_offset):
    final_x, final_y, final_z = zip(*final_positions)
    plt.figure(figsize=(fig_width, fig_height))

    plt.text(ideal_x, ideal_y, 'H', fontsize=font_size, color='red',
             fontweight='bold', ha='center', va='center', zorder=2)

    scatter_final = plt.scatter(
        final_x, final_y,
        c=abs(np.array(final_z) - ideal_z),
        cmap='Greens',
        s=10,
        edgecolor='black',
        linewidth=0.5,
        label='Final Positions',
        vmin=0, vmax=10, zorder=3
    )

    cbar = plt.colorbar(scatter_final, label='Height from Landing Pad (m)', orientation='vertical')
    cbar.ax.set_ylabel('Height from Landing Pad (m)', fontsize=font_size)
    cbar.ax.tick_params(labelsize=9)

    ax = plt.gca()
    filled_rect = Rectangle((ideal_x - 4, ideal_y - 4), 8, 8,
                             linewidth=0, edgecolor='none', facecolor='red', alpha=0.25, zorder=1)
    ax.add_patch(filled_rect)

    plt.title("Final Landing Positions", fontsize=font_size, color='black')
    plt.xlabel("X Position (m)", fontsize=font_size, color='black')
    plt.ylabel("Y Position (m)", fontsize=font_size, color='black')
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    plt.grid(True, color='black', linestyle='--', linewidth=0.5)
    plt.xlim(ideal_x - range_offset[0], ideal_x + range_offset[0])
    plt.ylim(ideal_y - range_offset[1], ideal_y + range_offset[1])
    plt.tight_layout()

    plt.savefig(f"{run_folder}/final_positions.PNG", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot initial/final landing positions')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to simulation_results.csv')
    parser.add_argument('--run', type=str, default=None,
                        help='Run folder name (e.g. run_17), looked up under vehicles/jaxguam/script/runs/')
    parser.add_argument('--output', type=str, default='./plots/laplace',
                        help='Output folder for plots')
    args = parser.parse_args()

    config = load_config()
    ideal_x, ideal_y, ideal_z = get_ideal_position(config)
    range_offset = [config['range_offset']['x'], config['range_offset']['y']]

    if args.csv:
        input_csv = args.csv
    elif args.run:
        input_csv = f"./vehicles/jaxguam/script/runs/{args.run}/simulation_results.csv"
    else:
        input_csv = "./vehicles/jaxguam/script/runs/run_17/simulation_results.csv"

    output_folder = args.output
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(input_csv)
    initial_positions = list(zip(df['x_i'], df['y_i'], df['z_i']))
    final_positions = list(zip(df['x_final'], df['y_final'], df['z_final']))
    results = df['Landing Result (Success/Fail)'].tolist()

    n_success = sum(1 for r in results if r == 'Success')
    print(f"Config: ideal=({ideal_x}, {ideal_y}, {ideal_z}), "
          f"range_offset=({range_offset[0]}, {range_offset[1]})")
    print(f"Reading: {input_csv}  ({n_success}/{len(results)} success)")
    print("Generating plots...")
    z_min_offset = config.get('z_min_offset', 35)
    z_range = config['range_offset']['z']
    plot_initial_pos(output_folder, initial_positions, ideal_x, ideal_y, ideal_z, range_offset)
    plot_final_pos(output_folder, final_positions, ideal_x, ideal_y, ideal_z, range_offset)
    plot_initial_pos_3d(output_folder, initial_positions, results, ideal_x, ideal_y, ideal_z, range_offset, z_min_offset, z_range)
    plot_final_pos_3d(output_folder, final_positions, results, ideal_x, ideal_y, ideal_z, range_offset)
    print(f"Plots saved in {output_folder}")


if __name__ == "__main__":
    main()
