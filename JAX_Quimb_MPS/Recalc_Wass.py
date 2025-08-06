# in the Checkpoint_Metrcs_sftp_back directory i want to run through all the trandformed data files and calculate the metrics for each file.
import numpy as np
import os
import sys
import stylized as st
import data_handling as dh
import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import data_handling as dh
import stylized as st
import sys
import argparse
import os
import os
import math
import re
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path
import datetime
import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import os
import seaborn as sns
import JAX_Quimb_CPU_Cechkpoint as jqc
import data_handling as dh
import jax
import jax.numpy as jnp



def process_results_layers(path_master, stride=5, use_new_metrics=True, use_epochs=0, coulors=None, generator_Critic=None, log_option=None):
    """Process the results of the QGAN model and generate analysis images.
    Args:
        path_master (str): Path to the directory containing the data.
        stride (int): Stride for the model.
    """

    if use_new_metrics:
        path_metrics = path_master+'/metrics_new'
    else:
        path_metrics = path_master+'/metrics'
    path_weights = path_master+'/weights'
    file_name = path_master.split('/')[-1]
    bond_dim = int(file_name.split('_')[1])
    layer = int(file_name.split('_')[3])
    time_inc = file_name.split('_')[6]
    n_qubits = int(file_name.split('_')[7])
    print(f"Bond dimension: {bond_dim}, Depth: {layer}, Time increment: {time_inc}, Number of qubits: {n_qubits}")

    # Read metrics directly from files
    loss_wass = np.loadtxt(path_metrics+'/loss_wass.txt')
    loss_acf_abs = np.loadtxt(path_metrics+'/loss_acf_abs.txt')
    loss_acf_nonabs = np.loadtxt(path_metrics+'/loss_acf_nonabs.txt')
    loss_leverage = np.loadtxt(path_metrics+'/loss_leverage.txt')

    if use_epochs == 0:
        epochs_metrics = np.arange(0, len(loss_wass)*50, 50)
    else:
        epochs_metrics = np.arange(0, use_epochs, 50)

    loss_wass_average, loss_wass_err = np.mean(loss_wass), np.std(loss_wass)/np.sqrt(len(loss_wass))
    loss_acf_abs_average, loss_acf_abs_err = np.mean(loss_acf_abs), np.std(loss_acf_abs)/np.sqrt(len(loss_acf_abs))
    loss_acf_nonabs_average, loss_acf_nonabs_err = np.mean(loss_acf_nonabs), np.std(loss_acf_nonabs)/np.sqrt(len(loss_acf_nonabs))
    loss_leverage_average, loss_leverage_err = np.mean(loss_leverage), np.std(loss_leverage)/np.sqrt(len(loss_leverage))

    if coulors is not None:
        pal = plt.cm.cool(np.linspace(0, 1, 4))  # Using 4 colors from the cool palette
    else:
        pal = sns.color_palette()

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('EMD', color=pal[0])
    ax1.tick_params(axis='y', labelcolor=pal[0])

    ax2 = ax1.twinx()
    ax2.plot(epochs_metrics, loss_acf_abs, label=r'$E_{abs}^{ACF}(\theta)$', color=pal[1])
    ax2.fill_between(epochs_metrics, loss_acf_abs-2*loss_acf_abs_err, loss_acf_abs+2*loss_acf_abs_err, alpha=0.3, color=pal[1])
    ax2.plot(epochs_metrics, loss_acf_nonabs, label=r'$E_{id}^{ACF}(\theta)$', color=pal[2])
    ax2.fill_between(epochs_metrics, loss_acf_nonabs-2*loss_acf_nonabs_err, loss_acf_nonabs+2*loss_acf_nonabs_err, alpha=0.3, color=pal[2])
    ax2.plot(epochs_metrics, loss_leverage, label=r'$E_{Lev}(\theta)$', color=pal[3])
    ax2.fill_between(epochs_metrics, loss_leverage-2*loss_leverage_err, loss_leverage+2*loss_leverage_err, alpha=0.3, color=pal[3])

    ax1.plot(epochs_metrics, loss_wass, label='EMD', color=pal[0])
    ax1.fill_between(epochs_metrics, loss_wass-2*loss_wass_err, loss_wass+2*loss_wass_err, alpha=0.3, color=pal[0])
    ax2.set_ylabel('Temporal metric loss')

    # Set artificial y-ticks and labels divided by 10 for ax2
    artificial_ticks = np.linspace(0, 0.6, 7)
    ax2.set_ylim(0, 0.6)
    ax2.set_yticks(artificial_ticks)
    ax2.tick_params(axis='y', labelright=True)
    ax2.set_yticklabels([f"{tick/10:.1f}" for tick in artificial_ticks])

    handles, labels = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
    fig.legend(handles, labels, bbox_to_anchor=(0.84, 0.9))
    #plt.title('Quantitative metrics' + f' for bond dim {bond_dim} and layer {layer}')
    # in the image add the bond dimensoin and layer in the top right corner
    #plt.text(0.95, 0.90, f'$\\chi$: {bond_dim}, Depth: {layer}', transform=ax1.transAxes, fontsize=12,verticalalignment='top', horizontalalignment='right',  bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    plt.tight_layout()
    # log scale
    #ax1.set_yscale('log')
    #ax2.set_yscale('log')
    plt.savefig(path_master+'/images'+'/quant_metrics_good.pdf')
    plt.close(fig)

def make_big_images_grid(root_dir, use_new_metrics=True, anal_image_type="quant_metrics"):
    """
    Create a grid of quant metric plots using GridSpec for fine control.
    Divide into 3 main column groups (col 0, cols 1-2, cols 3-4) with thick grey lines.
    Only bottom row has x-axis label, only leftmost column of each group has EMD ylabel,
    only rightmost column of each group has 'Temporal metric loss' ylabel.
    Add main column titles and super y-labels for each group.
    Only one legend is included at the bottom.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import re
    import seaborn as sns
    from matplotlib.gridspec import GridSpec

    def extract_b_L(path):
        m = re.search(r"b_(\d+)_L_(\d+)", path)
        if m:
            b = int(m.group(1))
            L = int(m.group(2))
            return b, L
        return None, None

    image_info = []
    for entry in os.listdir(root_dir):
        subdir = os.path.join(root_dir, entry)
        if os.path.isdir(subdir) and entry.startswith("b_"):
            b, L = extract_b_L(entry)
            if b is not None and L is not None:
                image_info.append((L, b, subdir))

    if not image_info:
        raise RuntimeError("No quant_metrics folders found in the expected folders.")

    image_info.sort()
    layers = sorted(set(L for L, b, p in image_info))
    bonds = sorted(set(b for L, b, p in image_info))
    info_dict = {(L, b): p for L, b, p in image_info}

    rows = len(layers)
    cols = len(bonds)

    # Define main column groups: [0], [1,2], [3,4] (for up to 5 columns)
    group_indices = [[0], [1,2], [3,4]]
    group_titles = ["Main Col 1", "Main Cols 2-3", "Main Cols 4-5"]

    # Add extra width between main column groups using width_ratios
    width_ratios = []
    for c in range(cols):
        width_ratios.append(1)
        if c < cols - 1 and any(c == idxs[-1] for idxs in group_indices[:-1]):
            width_ratios.append(0.36)  # Increased gap for more space

    fig = plt.figure(figsize=(4.2*cols+1.5+len(width_ratios)*0.5, 3.5*rows+1))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(rows, len(width_ratios), figure=fig, wspace=0.05, hspace=0.2, width_ratios=width_ratios)

    pal = sns.color_palette()
    legend_handles = []
    legend_labels = []

    axes_grid = [[None for _ in range(cols)] for _ in range(rows)]

    for i, L in enumerate(layers):
        for j, b in enumerate(bonds):
            # Compute the correct column index in the GridSpec (skip gap columns)
            gs_col = j
            for g in range(j):
                if any(g == idxs[-1] for idxs in group_indices[:-1]):
                    gs_col += 1
            ax1 = fig.add_subplot(gs[i, gs_col])
            ax1.grid(False)  # Ensure grid lines are disabled for ax1
            axes_grid[i][j] = ax1
            subdir = info_dict.get((L, b))
            if not subdir:
                ax1.axis('off')
                continue
            # Use metrics_new except for layer 1 and bond_dim 1
            if L == 1 and b == 1:
                path_metrics = os.path.join(subdir, 'metrics')
            else:
                if use_new_metrics:
                    path_metrics = os.path.join(subdir, 'metrics_new_short')
                else:
                    path_metrics = os.path.join(subdir, 'metrics')
            try:
                loss_wass = np.loadtxt(os.path.join(path_metrics, 'loss_wass.txt'))
                loss_acf_abs = np.loadtxt(os.path.join(path_metrics, 'loss_acf_abs.txt'))
                loss_acf_nonabs = np.loadtxt(os.path.join(path_metrics, 'loss_acf_nonabs.txt'))
                loss_leverage = np.loadtxt(os.path.join(path_metrics, 'loss_leverage.txt'))
            except Exception as e:
                ax1.text(0.5, 0.5, "Missing data", ha='center', va='center')
                ax1.axis('off')
                continue

            epochs_metrics = None
            # For top-left image (layer 1, bond 1), use 50 epochs per data point
            if L == 1 and b == 1:
                epochs_metrics = np.arange(0, len(loss_wass))
            else:
                epochs_metrics = np.arange(0, len(loss_wass)*50, 50)

            loss_wass_err = np.std(loss_wass)/np.sqrt(len(loss_wass))
            loss_acf_abs_err = np.std(loss_acf_abs)/np.sqrt(len(loss_acf_abs))
            loss_acf_nonabs_err = np.std(loss_acf_nonabs)/np.sqrt(len(loss_acf_nonabs))
            loss_leverage_err = np.std(loss_leverage)/np.sqrt(len(loss_leverage))

            # Only bottom row has x-axis label
            if i == rows-1:
                ax1.set_xlabel('Epoch')
            # else:
            #     ax1.set_xlabel('')
            #     ax1.tick_params(axis='x', labelbottom=False)
            # Instead, always set x-axis label for each subplot
            #ax1.set_xlabel('Epoch')

            # Only leftmost column of each group has EMD ylabel
            if any(j == idxs[0] for idxs in group_indices):
                ax1.set_ylabel('EMD', color=pal[0])
                ax1.tick_params(axis='y', labelcolor=pal[0])
            else:
                ax1.set_ylabel('')
                ax1.tick_params(axis='y', labelleft=False)
                ax1.set_yticklabels([])  # Remove y-tick labels
                ax1.set_yticks([])       # Remove y-ticks

            # Add gridlines to the main axis
            #ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

            ax2 = ax1.twinx()
            ax2.grid(False)  # Ensure grid lines are disabled for ax2
            ax2.plot(epochs_metrics, loss_acf_abs, label=r'$E_{abs}^{ACF}(\theta)$', color=pal[1])
            ax2.fill_between(epochs_metrics, loss_acf_abs-2*loss_acf_abs_err, loss_acf_abs+2*loss_acf_abs_err, alpha=0.3, color=pal[1])
            ax2.plot(epochs_metrics, loss_acf_nonabs, label=r'$E_{id}^{ACF}(\theta)$', color=pal[2])
            ax2.fill_between(epochs_metrics, loss_acf_nonabs-2*loss_acf_nonabs_err, loss_acf_nonabs+2*loss_acf_nonabs_err, alpha=0.3, color=pal[2])
            ax2.plot(epochs_metrics, loss_leverage, label=r'$E_{Lev}(\theta)$', color=pal[3])
            ax2.fill_between(epochs_metrics, loss_leverage-2*loss_leverage_err, loss_leverage+2*loss_leverage_err, alpha=0.3, color=pal[3])

            ax1.plot(epochs_metrics, loss_wass, label='EMD', color=pal[0])
            ax1.fill_between(epochs_metrics, loss_wass-2*loss_wass_err, loss_wass+2*loss_wass_err, alpha=0.3, color=pal[0])

            # Only rightmost column of each group has 'Temporal metric loss' ylabel
            if any(j == idxs[-1] for idxs in group_indices):
                ax2.set_ylabel('Temporal metric loss')
                # Artificially set y-ticks and labels from 0 to 0.6, and fix the axis limits
                '''artificial_ticks = np.linspace(0, 0.6, 7)
                ax2.set_ylim(0, 0.6)
                ax2.set_yticks(artificial_ticks)
                ax2.set_yticklabels([f"{tick:.1f}" for tick in artificial_ticks])'''
                ax2.tick_params(axis='y', labelright=True)
                #if i != 0:  # Apply custom y-tick labels for every row except the first row
                ax2.set_yticklabels([f"{float(label):.1f}" for label in ax2.get_yticks()])
                # remove grid lines from ax2

                ax2.grid(False)  # Ensure grid lines are disabled for ax2
            else:
                ax2.set_ylabel('')
                ax2.tick_params(axis='y', labelright=False)
                ax2.set_yticklabels([])  # Remove right y-tick labels
                ax2.set_yticks([])       # Remove right y-ticks

            if i == 0 and j == 0:
                handles, labels = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
                legend_handles = handles
                legend_labels = labels

            ax1.text(0.95, 0.90, f'$\\chi$: {b}, Depth: {L}', transform=ax1.transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    # Remove main column titles and main y labels for columns
    # (No code for setting titles or main y-labels)
    # ...existing code...

    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=4, fontsize=13, bbox_to_anchor=(0.5, 0.04))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fig.savefig(f"all_quant_metric_images_grid_{date}.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"Saved big image grid as all_quant_metric_images_grid_{date}.pdf")

def full_images(directory):
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            if dir_name.startswith('b_') and 'L_' in dir_name:
                # if it id Checkpoint_Metrcs_sftp_back/b_1_L_1_20250512_210237_SP500_10 dont run
                if dir_name == 'b_1_L_1_20250512_210237_SP500_10':
                    process_results_layers(os.path.join(root, dir_name), stride=5, use_new_metrics=False)
                    continue
                path_master = os.path.join(root, dir_name)
                print(f"Processing directory: {path_master}")
                process_results_layers(path_master, stride=5)

import sys
import argparse

parser = argparse.ArgumentParser(description="Recalculate metrics and generate analysis images.")
parser.add_argument("--metrics_folders", type=str, default="/home/s2334356/data1/Checkpoint_Metrcs", help="Path to the metrics folder.")

args = parser.parse_args()
metrics_folder = args.metrics_folders
 # Process results and generate images for each directory
#full_images(metrics_folder)
# Example usage after processing all images:
make_big_images_grid(metrics_folder, use_new_metrics=True)

'''file = "/home/s2334356/data1/Checkpoint_Metrcs/b_32_L_18_20250612_182811_SP500_10"
process_results_layers(file, stride=5, log_option=None)'''