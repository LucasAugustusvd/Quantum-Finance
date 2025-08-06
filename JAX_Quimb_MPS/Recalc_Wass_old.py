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


def Recalc(metrics_folder, remake_gen_critic_loss=False, long=None):
    #metrics_folder = "/Users/lucas/JAX_Quimb/Checkpoint_Metrcs"
    # Check if the directory exists
    if not os.path.exists(metrics_folder):
        print(f"Warning: The directory '{metrics_folder}' doesn't exist.")
        print("Creating the directory...")

    # Dictionary to store metrics by bond dimension and number of layers
    metrics_data = defaultdict(dict)
    bond_dimensions = set()
    layer_numbers = set()

    # Regular expression to extract parameters from filenames
    pattern = r"b_(\d+)_L_(\d+)_.*"

    # Walk through the metrics folder
    print(f"Scanning directory: {metrics_folder}")
    for root, dirs, files in os.walk(metrics_folder):
        print(f"Found {len(dirs)} directories in {root}")
        for dir_name in dirs:
            if dir_name == 'b_1_L_1_20250512_210237_SP500_10':
                continue
            print(f"Processing directory: {dir_name}")
            match = re.match(pattern, dir_name)

            if long is None:
                metrics_dir = os.path.join(root, dir_name, "metrics_new_short")
            elif long == True:
                metrics_dir = os.path.join(root, dir_name, "metrics_new_long")

            # check if directory already has metrics_new
            if os.path.exists(metrics_dir):
                # Check how many epochs have already been calculated
                loss_wass_path = os.path.join(metrics_dir, 'loss_wass.txt')
                if os.path.exists(loss_wass_path):
                    with open(loss_wass_path, 'r') as f:
                        already_calculated = len(f.readlines())
                else:
                    already_calculated = 0
            else:
                already_calculated = 0
            already_calculated = 0
            if match:
                bond_dim = int(match.group(1))
                n_layers = int(match.group(2))
                n_qubitss = int(dir_name.split('_')[7])
                bond_dimensions.add(bond_dim)
                layer_numbers.add(n_layers)
                print("Bond Dimension:", bond_dim, "Number of Layers:", n_layers)

                transformed_Dats = os.path.join(root, dir_name, "weights/transformed_data")
                if remake_gen_critic_loss:
                    generator_params_path = os.path.join(root, dir_name, "weights/params_generator")
                    critic_params_path = os.path.join(root, dir_name, "weights/params_critic")

                # Loop through all transformed data files in the directory
                files = [
                    f for f in os.listdir(transformed_Dats)
                    if f.startswith('generated_data_') and f.endswith('_generator.npy')
                ]
                if remake_gen_critic_loss:
                    gen_param_files = [
                        f for f in os.listdir(generator_params_path)
                        if f.startswith('generator_') and f.endswith('_generator.npy')
                    ]
                    critic_param_files = [
                        f for f in os.listdir(critic_params_path)
                        if f.startswith('critic_') and f.endswith('_generator.npy')
                    ]
                def extract_epoch(filename):
                    match = re.match(r'generated_data_(\d+)_generator\.npy', filename)
                    return int(match.group(1)) if match else -1

                files_sorted = sorted(files, key=extract_epoch)
                if remake_gen_critic_loss:
                    gen_param_files_sorted = sorted(gen_param_files, key=extract_epoch)
                    critic_param_files_sorted = sorted(critic_param_files, key=extract_epoch)
                # Only process files that have not been calculated yet
                files_to_process = files_sorted[already_calculated:]
                if remake_gen_critic_loss:
                    gen_param_files_to_process = gen_param_files_sorted[already_calculated:]
                    critic_param_files_to_process = critic_param_files_sorted[already_calculated:]

                if not files_to_process:
                    print(f"All epochs already processed for {dir_name}. Skipping.")
                    continue

                # Load previous metrics if they exist, else start fresh
                loss_wass, loss_ACF, loss_ACF_nonabs, loss_leverage = [], [], [], []
                if already_calculated > 0:
                    loss_wass = list(np.loadtxt(os.path.join(metrics_dir, 'loss_wass.txt')))
                    loss_ACF = list(np.loadtxt(os.path.join(metrics_dir, 'loss_acf_abs.txt')))
                    loss_ACF_nonabs = list(np.loadtxt(os.path.join(metrics_dir, 'loss_acf_nonabs.txt')))
                    loss_leverage = list(np.loadtxt(os.path.join(metrics_dir, 'loss_leverage.txt')))
                print("Qubits:", n_qubitss)
                chopsize = (2 * n_qubitss)
                log_returns_sp500 = dh.load_SP500_lr()
                if long == None:
                    benchmarks, benchmark_lags = st.benchmark(log_returns_sp500, chopsize-2)
                elif long == True:
                    benchmarks, benchmark_lags = st.benchmark(log_returns_sp500, chopsize)

                for i in range(len(files_to_process)):
                    file_name = files_to_process[i]
                    if remake_gen_critic_loss:
                        gen_param_file = gen_param_files_to_process[i]
                        critic_param_file = critic_param_files_to_process[i]
                        g_params = np.load(os.path.join(generator_params_path, gen_param_file), allow_pickle=True).item()
                        c_params = np.load(os.path.join(critic_params_path, critic_param_file), allow_pickle=True).item()
                    file_path = os.path.join(transformed_Dats, file_name)
                    print(f"Processing file: {file_path}")

                    # Load the generated data
                    generated_data_transformed = np.load(file_path, allow_pickle=True)

                    metric = st.metrics(
                        generated_data_transformed, 
                        log_returns_sp500,
                        benchmark_lags, 
                        benchmarks, 
                        only_EMD=False
                    )


                    loss_wass.append(metric[3])
                    loss_ACF.append(metric[0])
                    loss_ACF_nonabs.append(metric[1])
                    loss_leverage.append(metric[2])

                    if remake_gen_critic_loss:
                        rng = jax.random.PRNGKey(1234)

                        # 3) We'll replicate your 'noise_cst' for evaluation:
                        rng, rng_eval = jax.random.split(rng)
                        transformed_back_data = dh.transform(generated_data_transformed)
                        critic_model = jqc.Critic(dropout_rate=0.5)
                        moise_dim = n_qubitss
                        log_returns_sp500 = dh.load_SP500_lr()
                        transformed_lr, transform_params = dh.transform(log_returns_sp500)
                        noise_dim = n_qubitss
                        train_time_series = dh.chopchop(transformed_lr, 20, 5).astype('float64')
                        train_time_series = train_time_series.reshape(
                            train_time_series.shape[0], 
                            train_time_series.shape[1]
                        )
                        batch_size = train_time_series.shape[0] // 10

                        # Load the generator and critic parameters
                        #jqc.generator_loss_wgan(gen_params, disc_params, batch_size, rng, critic_model, noise_dim, n_layers, chopsize, bond_dim)
                        g_loss_val = jqc.generator_loss_wgan(
                            g_params, 
                            c_params, 
                            batch_size, 
                            rng, 
                            critic_model, 
                            noise_dim, 
                            n_layers, 
                            chopsize, 
                            bond_dim
                        )
                        
                        '''batch_size = real_batch.shape[0]

                        # 1) Create noise for fake data
                        rng, rng_noise, rng_gp = jax.random.split(rng, 3)
                        noise = jax.random.uniform(
                            rng_noise,
                            shape=(batch_size, noise_dim),
                            minval=0.0, maxval=2*jnp.pi
                        )

                        # 2) Generate fake data
                        fake_out = generator_apply(gen_params, noise, n_layers, bond_dim=bond_dim)
                        fake_out = jnp.reshape(fake_out, (batch_size, chopsize, 1))

                        # 3) Reshape real data
                        real_out = jnp.expand_dims(real_batch, axis=-1)

                        # 4) Critic forward pass
                        variables = {'params': disc_params}
                        real_logits = critic_model.apply(variables, real_out, train=True, rngs={'dropout': rng})
                        fake_logits = critic_model.apply(variables, fake_out, train=True, rngs={'dropout': rng})

                        wgan_loss = jnp.mean(fake_logits) - jnp.mean(real_logits)
'''



                # Save the metrics (overwrite with all, including previous)
                if not os.path.exists(metrics_dir):
                    os.makedirs(metrics_dir)
                np.savetxt(os.path.join(metrics_dir, 'loss_wass.txt'), loss_wass)
                np.savetxt(os.path.join(metrics_dir, 'loss_acf_abs.txt'), loss_ACF)
                np.savetxt(os.path.join(metrics_dir, 'loss_acf_nonabs.txt'), loss_ACF_nonabs)
                np.savetxt(os.path.join(metrics_dir, 'loss_leverage.txt'), loss_leverage)

def process_results_layers(path_master, stride=5, use_new_metrics=True, use_epochs=0, coulors=None, generator_Critic=None, log_option=None, long=None):
    """Process the results of the QGAN model and generate analysis images.
    Args:
        path_master (str): Path to the directory containing the data.
        stride (int): Stride for the model.
    """

    if use_new_metrics:
        if long == None:
            path_metrics = path_master+'/metrics_new_short'
        elif long == True:
            path_metrics = path_master+'/metrics_new_long'
    else:
        path_metrics = path_master+'/metrics'
    path_weights = path_master+'/weights'
    # format of of path_master is checkpoints/b_8_L_1_20250513_072333_SP500_10
    # split the path by '_'
    file_name = path_master.split('/')[-1]
    bond_dim = file_name.split('_')[1]
    bond_dim = int(bond_dim)
    layer = file_name.split('_')[3]
    layer = int(layer)
    time_inc = file_name.split('_')[6]
    n_qubits = file_name.split('_')[7]
    n_qubits = int(n_qubits)
    print(f"Bond dimension: {bond_dim}, Layer: {layer}, Time increment: {time_inc}, Number of qubits: {n_qubits}")
    loss_wass = np.loadtxt(path_metrics+'/loss_wass.txt')

    #length_metrics = np.loadtxt(path_metrics+'/loss_wass.txt').shape[0]
    #length_loss = np.loadtxt(path_metrics+'/loss_gen.txt').shape[0]
    if use_new_metrics and use_epochs == 0:
        # epochs is the lentgh of the loss_wass file where each entey is equal to 50 epochs + the first one being epoch 0
        epochs_metrics = np.arange(0, len(loss_wass)*50, 50)
    elif use_new_metrics and use_epochs > 0:
        epochs_metrics = np.arange(0, use_epochs, 50)
    else:
        epochs_metrics = np.loadtxt(path_metrics+'/loss_epochs.txt')

    chopsize = 2*n_qubits
    generated_data_transformed_dir = path_weights+'/transformed_data'
    #get the file in the directory with the highest number in the following format:generated_data_650_generator.npy
    if use_new_metrics:
        if use_epochs == 0:
            generated_data_transformed = max([f for f in os.listdir(generated_data_transformed_dir) if f.startswith('generated_data_')], key=lambda x: int(x.split('_')[2].split('.')[0]))
        else:
            generated_data_transformed = f"generated_data_{use_epochs}_generator.npy"
        print(f"Using generated data: {generated_data_transformed}")
        generated_data_transformed = os.path.join(generated_data_transformed_dir, generated_data_transformed)
    #generated_data_transformed = "/Users/lucas/JAX_Quimb/Checkpoint_Metrcs_sftp_back/b_8_L_1_20250513_072333_SP500_10/weights/transformed_data/generated_data_2400_generator.npy"
    else:
        generated_data_transformed = "/home/s2334356/data1/Checkpoint_Metrcs/b_1_L_1_20250512_210237_SP500_10/weights/lowest_wass_generator.npy"
        

    if time_inc == "SP500":
        log_returns_sp500 = dh.load_SP500_lr()
    else:
        log_returns_sp500 = dh.load_SP500_lr_own(time_inc)

    if long == None:
        benchmarks, benchmark_lags = st.benchmark(log_returns_sp500, chopsize-2)
    elif long == True:
        benchmarks, benchmark_lags = st.benchmark(log_returns_sp500, chopsize)
    ts = dh.chopchop(log_returns_sp500, chopsize, stride)

    maximum_lag = len(benchmark_lags)+1
    print(f"Maximum lag: {maximum_lag}")


    if use_epochs == 0:
        loss_wass = np.loadtxt(path_metrics+'/loss_wass.txt')
        loss_acf_abs = np.loadtxt(path_metrics+'/loss_acf_abs.txt')*(1/np.sqrt(benchmarks.shape[1]))
        loss_acf_nonabs = np.loadtxt(path_metrics+'/loss_acf_nonabs.txt')*(1/np.sqrt(benchmarks.shape[1]))
        loss_leverage = np.loadtxt(path_metrics+'/loss_leverage.txt')*(1/np.sqrt(benchmarks.shape[1]))
        if generator_Critic is not None:
            loss_gen = np.loadtxt(path_metrics+'/loss_gen.txt')
            loss_disc = np.loadtxt(path_metrics+'/loss_disc.txt')
            length_loss = loss_gen.shape[0]
    elif use_epochs > 0:
        loss_wass = np.loadtxt(path_metrics+'/loss_wass.txt')[:use_epochs//50]
        loss_acf_abs = np.loadtxt(path_metrics+'/loss_acf_abs.txt')[:use_epochs//50]*(1/np.sqrt(benchmarks.shape[1]))
        loss_acf_nonabs = np.loadtxt(path_metrics+'/loss_acf_nonabs.txt')[:use_epochs//50]*(1/np.sqrt(benchmarks.shape[1]))
        loss_leverage = np.loadtxt(path_metrics+'/loss_leverage.txt')[:use_epochs//50]*(1/np.sqrt(benchmarks.shape[1]))
        if generator_Critic is not None:
            loss_gen = np.loadtxt(path_metrics+'/loss_gen.txt')[:use_epochs//50]
            loss_disc = np.loadtxt(path_metrics+'/loss_disc.txt')[:use_epochs//50]
            length_loss = loss_gen.shape[0]

    #generated_data_transformed is a npy file
    if use_new_metrics:
        if os.path.exists(generated_data_transformed):
            try:
                gen_samp = np.load(generated_data_transformed, allow_pickle=True)
            except ValueError as e:
                print(f"Error loading generated data: {e}")
                print("No 3k epoch reached.")
                generated_data_transformed = max([f for f in os.listdir(generated_data_transformed_dir) if f.startswith('generated_data_')], key=lambda x: int(x.split('_')[2].split('.')[0]))
                generated_data_transformed = os.path.join(generated_data_transformed_dir, generated_data_transformed)
                gen_samp = np.load(generated_data_transformed, allow_pickle=True)
        else:
            print(f"File does not exist: {generated_data_transformed}")
            generated_data_transformed = max([f for f in os.listdir(generated_data_transformed_dir) if f.startswith('generated_data_')], key=lambda x: int(x.split('_')[2].split('.')[0]))
            generated_data_transformed = os.path.join(generated_data_transformed_dir, generated_data_transformed)
            gen_samp = np.load(generated_data_transformed, allow_pickle=True)

    else:

        params = np.load(generated_data_transformed, allow_pickle=True).item()
        log_returns_sp500 = dh.load_SP500_lr()
        
        transformed_lr, transform_params = dh.transform(log_returns_sp500)
        noise_dim = 10
        train_time_series = dh.chopchop(transformed_lr, 20, 5).astype('float64')
        train_time_series = train_time_series.reshape(
            train_time_series.shape[0], 
            train_time_series.shape[1]
        )
        rng = jax.random.PRNGKey(1234)

        # 3) We'll replicate your 'noise_cst' for evaluation:
        rng, rng_eval = jax.random.split(rng)
        noise_cst = jax.random.uniform(
            rng_eval,
            shape=(train_time_series.shape[0], noise_dim),
            minval=0.0, maxval=2*jnp.pi
        )
        gen_cst_out = jqc.generator_apply(
                params, 
                noise_cst, 
                1, 
                bond_dim=1
            )
        gen_cst_out_np = np.array(gen_cst_out)
        generated_data_transformed = dh.inverse_transform(
            gen_cst_out_np, 
            transform_params
        )
        gen_samp = generated_data_transformed
    
    layer_ACF_abs = st.auto_corr(np.abs(gen_samp), max_lag = maximum_lag, title = '', double_conf = False, first = False, plot = False)
    layer_ACF_nonabs = st.auto_corr(gen_samp, max_lag = maximum_lag, title = '', double_conf = False, first = False, plot = False)
    layer_leverage = st.leverage_effect(gen_samp, maximum_lag, title = '', plot = False)

    loss_wass_average, loss_wass_err = np.mean(loss_wass), np.std(loss_wass)/np.sqrt(len(loss_wass))
    loss_acf_abs_average, loss_acf_abs_err = np.mean(loss_acf_abs), np.std(loss_acf_abs)/np.sqrt(len(loss_acf_abs))
    loss_acf_nonabs_average, loss_acf_nonabs_err = np.mean(loss_acf_nonabs), np.std(loss_acf_nonabs)/np.sqrt(len(loss_acf_nonabs))
    loss_leverage_average, loss_leverage_err = np.mean(loss_leverage), np.std(loss_leverage)/np.sqrt(len(loss_leverage))
    if generator_Critic is not None:
        loss_gen_average, loss_gen_err = np.mean(loss_gen), np.std(loss_gen)/np.sqrt(len(loss_gen))
        loss_disc_average, loss_disc_err = np.mean(loss_disc), np.std(loss_disc)/np.sqrt(len(loss_disc))
    ACF_abs_average, ACF_abs_err = np.mean(layer_ACF_abs), np.std(layer_ACF_abs)/np.sqrt(len(layer_ACF_abs))
    ACF_nonabs_average, ACF_nonabs_err = np.mean(layer_ACF_nonabs), np.std(layer_ACF_nonabs)/np.sqrt(len(layer_ACF_nonabs))
    leverage_average, leverage_err = np.mean(layer_leverage), np.std(layer_leverage)/np.sqrt(len(layer_leverage))

    st.plot_gen_vs_benchmark(gen_samp, ts, ACF_abs_average, ACF_abs_err, ACF_nonabs_average, ACF_nonabs_err, leverage_average, leverage_err, ts, max_lag = maximum_lag, double_count = 4, path=path_master+'/images')

    if coulors is not None:
        pal = plt.cm.cool(np.linspace(0, 1, 4))  # Using 4 colors from the cool palette
    else:
        pal = sns.color_palette()
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('EMD', color = pal[0])
    ax1.tick_params(axis='y', labelcolor=pal[0])

    ax2 = ax1.twinx()
    try:
        ax2.plot(epochs_metrics,loss_acf_abs, label =  r'$E_{abs}^{ACF}(\theta)$', color = pal[1])
    except ValueError as e:
        epochs_metrics = np.arange(0, len(loss_wass)*50, 50)
        ax2.plot(epochs_metrics,loss_acf_abs, label =  r'$E_{abs}^{ACF}(\theta)$', color = pal[1])
        
    if log_option==None:
        ax2.fill_between(epochs_metrics,loss_acf_abs-2*loss_acf_abs_err,loss_acf_abs+2*loss_acf_abs_err, alpha = 0.3, color = pal[1])
        ax2.plot(epochs_metrics,loss_acf_nonabs, label = r'$E_{id}^{ACF}(\theta)$', color = pal[2])
        ax2.fill_between(epochs_metrics,loss_acf_nonabs-2*loss_acf_nonabs_err,loss_acf_nonabs+2*loss_acf_nonabs_err, alpha = 0.3, color = pal[2])
        ax2.plot(epochs_metrics,loss_leverage, label = r'$E_{Lev}(\theta)$' , color = pal[3])
        ax2.fill_between(epochs_metrics,loss_leverage-2*loss_leverage_err,loss_leverage+2*loss_leverage_err, alpha = 0.3, color = pal[3])
        #ax1.set_ylim(0,0.007)
        #ax2.set_ylim(0,0.5)
        ax1.plot(epochs_metrics,loss_wass, label = 'EMD', color = pal[0])
        ax1.fill_between(epochs_metrics,loss_wass-2*loss_wass_err,loss_wass+2*loss_wass_err, alpha = 0.3, color = pal[0])
        ax2.set_ylabel('Temporal metric loss')
        handles, labels = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
        fig.legend(handles, labels,  bbox_to_anchor=(0.84,0.9))#loc = 'center'
        plt.title('Quantitative metrics' + f' for bond dim {bond_dim} and layer {layer}')
        plt.tight_layout()
        plt.savefig(path_master+'/images'+'/quant_metrics.pdf')
        plt.close(fig)
    else:
        ax2.fill_between(epochs_metrics,loss_acf_abs-2*loss_acf_abs_err,loss_acf_abs+2*loss_acf_abs_err, alpha = 0.3, color = pal[1])
        ax2.plot(epochs_metrics,loss_acf_nonabs, label = r'$E_{id}^{ACF}(\theta)$', color = pal[2])
        ax2.fill_between(epochs_metrics,loss_acf_nonabs-2*loss_acf_nonabs_err,loss_acf_nonabs+2*loss_acf_nonabs_err, alpha = 0.3, color = pal[2])
        ax2.plot(epochs_metrics,loss_leverage, label = r'$E_{Lev}(\theta)$' , color = pal[3])
        ax2.fill_between(epochs_metrics,loss_leverage-2*loss_leverage_err,loss_leverage+2*loss_leverage_err, alpha = 0.3, color = pal[3])
        #ax1.set_ylim(0,0.007)
        #ax2.set_ylim(0,0.5)
        ax1.plot(epochs_metrics,loss_wass, label = 'EMD', color = pal[0])
        ax1.fill_between(epochs_metrics,loss_wass-2*loss_wass_err,loss_wass+2*loss_wass_err, alpha = 0.3, color = pal[0])
        ax2.set_ylabel('Temporal metric loss')
        #log scales for the y-axis
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        handles, labels = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
        fig.legend(handles, labels,  bbox_to_anchor=(0.84,0.9))#loc = 'center'
        plt.title('Quantitative metrics' + f' for bond dim {bond_dim} and layer {layer}')
        plt.tight_layout()
        plt.savefig(path_master+'/images'+'/quant_metrics_log.pdf')
        plt.close(fig)

    if generator_Critic is not None:
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Generator loss', color = pal[0])
        ax1.tick_params(axis='y', labelcolor=pal[0])
        ax2 = ax1.twinx()        
        ax1.plot(loss_gen_average[0], label =  r'Generator loss', color = pal[0])


        ax1.fill_between(np.arange(length_loss),loss_gen_average[0]-2* loss_gen_err[0],loss_gen_average[0]+2*loss_gen_err[0], alpha = 0.3, color = pal[0])
        ax2.plot(loss_disc_average[0], label =  r'Critic loss', color = pal[1])
        ax2.fill_between(np.arange(length_loss),loss_disc_average[0]-2*loss_disc_err[0],loss_disc_average[0]+2*loss_disc_err[0], alpha = 0.3, color = pal[1])
        #plt.xlabel('Epochs')
        ax2.set_ylabel('Critic loss', color = pal[1])
        ax2.tick_params(axis='y', labelcolor=pal[1])
        handles, labels = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
        fig.legend(handles, labels,  bbox_to_anchor=(0.84,0.35))#loc = 'center'
        plt.title('Generator and critic loss')
        #plt.legend()
        plt.tight_layout()
        plt.savefig('gen_critic_loss.pdf')
        plt.show()

def make_big_images(root_dir, anal_image_type="quant_metrics.pdf"):
    # Function to extract bond dimension (b) and layer (L) from folder name
    def extract_b_L(path):
        # Example: b_16_L_1_20250513_182350_SP500_10
        m = re.search(r"b_(\d+)_L_(\d+)", path)
        if m:
            b = int(m.group(1))
            L = int(m.group(2))
            return b, L
        return None, None

    # Collect all quant_metrics.pdf paths in the correct subfolders, with b and L
    image_info = []
    for entry in os.listdir(root_dir):
        subdir = os.path.join(root_dir, entry)
        if os.path.isdir(subdir) and entry.startswith("b_"):
            pdf_path = os.path.join(subdir, "images", anal_image_type)
            if os.path.isfile(pdf_path):
                b, L = extract_b_L(entry)
                if b is not None and L is not None:
                    image_info.append((L, b, pdf_path))

    if not image_info:
        raise RuntimeError("No quant_metrics.pdf files found in the expected folders.")

    # Sort by L (rows), then b (columns)
    image_info.sort()
    layers = sorted(set(L for L, b, p in image_info))
    bonds = sorted(set(b for L, b, p in image_info))

    # Build a lookup for fast access
    info_dict = {(L, b): p for L, b, p in image_info}

    # Load images in sorted grid order
    images = []
    for L in layers:
        for b in bonds:
            p = info_dict.get((L, b))
            if p:
                try:
                    pages = convert_from_path(p, first_page=1, last_page=1)
                    if pages:
                        img = pages[0].convert("RGB")
                        images.append(img)
                    else:
                        print(f"No pages found in {p}")
                except Exception as e:
                    print(f"Could not open {p}: {e}")
            else:
                # Optionally, add a blank image if missing
                images.append(None)

    # Remove trailing None images if any
    images = [img for img in images if img is not None]
    if not images:
        raise RuntimeError("No images could be loaded from PDFs.")

    # Resize all images to the same size (use the size of the first image)
    w, h = images[0].size
    images = [img.resize((w, h)) for img in images]

    rows = len(layers)
    cols = len(bonds)

    # --- Add axis labels ---
    label_font_size = 32
    try:
        font = ImageFont.truetype("arial.ttf", label_font_size)
    except:
        font = ImageFont.load_default()

    # Create a temporary image and draw object for text size calculation
    tmp_img = Image.new('RGB', (10, 10))
    tmp_draw = ImageDraw.Draw(tmp_img)

    def get_text_width_height(text, font):
        # Use textbbox if available (Pillow >=8.0), else fallback to textsize
        if hasattr(tmp_draw, "textbbox"):
            bbox = tmp_draw.textbbox((0, 0), text, font=font)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
        else:
            width, height = tmp_draw.textsize(text, font=font)
        return width, height

    # Calculate label space
    label_w = max(get_text_width_height(str(L), font)[0] for L in layers) + 20
    label_h = max(get_text_width_height(str(b), font)[1] for b in bonds) + 20

    # Create new image with space for labels
    big_img = Image.new('RGB', (cols * w + label_w, rows * h + label_h), color='white')
    draw = ImageDraw.Draw(big_img)

    # Draw column (bond dimension) labels
    for j, b in enumerate(bonds):
        text = str(b)
        tw, th = get_text_width_height(text, font)
        x = label_w + j * w + (w - tw) // 2
        y = 0
        draw.text((x, y), text, fill='black', font=font)
    draw.text((label_w + (cols * w)//2, 0), "Bond dimension", fill='black', font=font, anchor="ms")

    # Draw row (layer) labels
    for i, L in enumerate(layers):
        text = str(L)
        tw, th = get_text_width_height(text, font)
        x = 0
        y = label_h + i * h + (h - th) // 2
        draw.text((x, y), text, fill='black', font=font)
    draw.text((0, label_h + (rows * h)//2), "Layer", fill='black', font=font, anchor="ls")

    # Paste images
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        big_img.paste(img, (label_w + col * w, label_h + row * h))

    # Save the big image
    date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #big_img.save(f"all_quant_metric_images_{date}.png")
    # Optionally, save as a PDF
    big_img.save(f"all_quant_metric_images_{date}_{anal_image_type}.pdf", "PDF", resolution=100.0)
    # Print success message
    print(f"Saved big image with {len(images)} images as all_quant_metric_images.png")
    big_img.close()

def make_analysis(metrics_folder):
    # Check if the directory exists
    if not os.path.exists(metrics_folder):
        print(f"Warning: The directory '{metrics_folder}' doesn't exist.")
        print("Creating the directory...")
        try:
            os.makedirs(metrics_folder)
            print(f"Created directory: {metrics_folder}")
        except Exception as e:
            print(f"Error creating directory: {e}")
            print("Please check the path and permissions.")
            exit(1)

    # Dictionary to store metrics by bond dimension and number of layers
    metrics_data = defaultdict(dict)
    bond_dimensions = set()
    layer_numbers = set()

    # Regular expression to extract parameters from filenames
    pattern = r"b_(\d+)_L_(\d+)_.*"

    # Walk through the metrics folder
    print(f"Scanning directory: {metrics_folder}")
    for root, dirs, files in os.walk(metrics_folder):
        print(f"Found {len(dirs)} directories in {root}")
        for dir_name in dirs:
            print(f"Processing directory: {dir_name}")
            match = re.match(pattern, dir_name)
            
            if match:
                bond_dim = int(match.group(1))
                n_layers = int(match.group(2))
                
                # Track unique values for later plotting
                bond_dimensions.add(bond_dim)
                layer_numbers.add(n_layers)
                
                # Path to the loss_wass.txt file
                loss_file = os.path.join(root, dir_name, "metrics_new/loss_wass.txt")
                
                if os.path.exists(loss_file):
                    try:
                        # Read the last value in the file as the final loss value
                        with open(loss_file, 'r') as f:
                            lines = f.readlines()
                            if lines:
                                # Take the last value in the file
                                #last_value = float(lines[-1].strip())
                                # take the 800th value in the file
                                #last_value = float(lines[800].strip())
                                # take the lowest value in the file
                                last_value = min([float(line.strip()) for line in lines])
                                metrics_data[bond_dim][n_layers] = last_value
                                print(f"Found data: bond_dim={bond_dim}, n_layers={n_layers}, value={last_value}")
                    except Exception as e:
                        print(f"Error reading {loss_file}: {e}")
                else:
                    print(f"File not found: {loss_file}")

    # Convert sets to sorted lists
    bond_dimensions = sorted(list(bond_dimensions))
    layer_numbers = sorted(list(layer_numbers))

    print(f"Found data for bond dimensions: {bond_dimensions}")
    print(f"Found data for layer numbers: {layer_numbers}")

    # Create a matrix representation of the data
    cost_matrix = np.zeros((len(bond_dimensions), len(layer_numbers)))
    cost_matrix.fill(np.nan)  # Fill with NaN for missing data points

    for i, bond_dim in enumerate(bond_dimensions):
        for j, n_layers in enumerate(layer_numbers):
            if n_layers in metrics_data[bond_dim]:
                cost_matrix[i, j] = metrics_data[bond_dim][n_layers]

    # Create output directory if it doesn't exist
    output_dir = os.path.join(metrics_folder, 'analysis_results')
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    # Save the data for later use
    np.save(os.path.join(output_dir, 'cost_matrix.npy'), cost_matrix)
    np.save(os.path.join(output_dir, 'bond_dimensions.npy'), np.array(bond_dimensions))
    np.save(os.path.join(output_dir, 'layer_numbers.npy'), np.array(layer_numbers))

    # Also create a DataFrame for easier analysis
    df = pd.DataFrame(index=bond_dimensions, columns=layer_numbers)
    for bond_dim in bond_dimensions:
        for n_layers in layer_numbers:
            if n_layers in metrics_data[bond_dim]:
                df.loc[bond_dim, n_layers] = metrics_data[bond_dim][n_layers]

    df.to_csv(os.path.join(output_dir, 'metrics_data.csv'))

    print(f"Data has been saved to {output_dir}")
    print("Matrix shape:", cost_matrix.shape)
    print("Sample of the cost matrix:")
    print(cost_matrix[:5, :5])  # Print a small sample

    # Create a simple text file with summary information
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write(f"Bond dimensions: {bond_dimensions}\n")
        f.write(f"Layer numbers: {layer_numbers}\n")
        f.write(f"Matrix shape: {cost_matrix.shape}\n")
        f.write("\nMatrix values:\n")
        f.write(str(df))

def make_3d_plot(metrics_folder):
    # Load the pre-processed data
    cost_matrix = np.load(os.path.join(metrics_folder, 'cost_matrix.npy'))
    bond_dimensions = np.load(os.path.join(metrics_folder, 'bond_dimensions.npy'))
    layer_numbers = np.load(os.path.join(metrics_folder, 'layer_numbers.npy'))

    # Create grid points for plotting
    X, Y = np.meshgrid(bond_dimensions, layer_numbers)
    X = X.T  # Transpose to match the orientation of cost_matrix
    Y = Y.T

    # Convert matrix data to flatten arrays for interpolation
    # Filter out NaN values
    mask = ~np.isnan(cost_matrix)
    x_points = X[mask]
    y_points = Y[mask]
    z_points = cost_matrix[mask]

    # Create a finer mesh grid for smoother plotting
    x_grid = np.linspace(min(bond_dimensions), max(bond_dimensions), 100)
    y_grid = np.linspace(min(layer_numbers), max(layer_numbers), 100)
    X_fine, Y_fine = np.meshgrid(x_grid, y_grid)
    X_fine = X_fine.T
    Y_fine = Y_fine.T

    # Interpolate for a smooth surface
    Z_fine = griddata((x_points, y_points), z_points, (X_fine, Y_fine), method='cubic')

    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surface = ax.plot_surface(X_fine, Y_fine, Z_fine, cmap='cool', edgecolor='none', alpha=0.9)

    # Add scatter points of actual data points
    ax.scatter(x_points, y_points, z_points, c='cyan', s=50, alpha=1, label='Data points')

    ax.set_xlabel('Bond Dimension')
    ax.set_ylabel('Layer Depth')
    ax.set_zlabel('Cost Function')
    ax.set_title('3D Surface Plot of Cost Function by Bond Dimension and Layer Depth')

    # Color bar to map values to colors
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5, label='Cost Function Value')

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_folder, 'cost_function_3d_plot.pdf'), dpi=300)
    #plt.show()

    # also make a 2d plot with layers vs cost function and  another bondimension vs cost function
    # Create 2D plots
    from cycler import cycler
    cool_colors = plt.cm.cool(np.linspace(0, 1, max(len(bond_dimensions), len(layer_numbers))))
    plt.rcParams['axes.prop_cycle'] = cycler(color=cool_colors)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # Plot Layer Depth vs Cost Function
    for i, bond_dim in enumerate(bond_dimensions):
        ax[0].plot(layer_numbers, cost_matrix[i], label=f'Bond Dim: {bond_dim}')
    ax[0].set_xlabel('Layer Depth')
    ax[0].set_ylabel('Cost Function')
    ax[0].set_title('Layer Depth vs Cost Function')
    ax[0].legend()
    # Set the x-ticks to be the layer numbers
    ax[0].set_xticks(layer_numbers)
    ax[0].set_xticklabels(layer_numbers)
    ax[0].set_xlim(min(layer_numbers), max(layer_numbers))

    # Plot Bond Dimension vs Cost Function
    for j, layer_num in enumerate(layer_numbers):
        ax[1].plot(bond_dimensions, cost_matrix[:, j], label=f'Layer: {layer_num}')
    ax[1].set_xlabel('Bond Dimension')
    ax[1].set_ylabel('Cost Function')
    ax[1].set_title('Bond Dimension vs Cost Function')
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_folder, 'cost_function_2d_plots.pdf'), dpi=300)
    plt.close(fig)


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

'''file = "Checkpoint_Metrcs/b_1_L_5_20250517_152253_SP500_10"
process_results_layers(file, stride=5, use_new_metrics=False)'''

'''file = "/home/s2334356/data1/Checkpoint_Metrcs/b_24_L_5_20250518_041048_SP500_10"
process_results_layers(file, stride=5, use_new_metrics=True, use_epochs=6000)
'''
# Define the metrics folder path
#metrics_folder = "/home/s2334356/data1/Checkpoint_Metrcs"
import sys
import argparse

parser = argparse.ArgumentParser(description="Recalculate metrics and generate analysis images.")
parser.add_argument("--metrics_folders", type=str, default="/home/s2334356/data1/Checkpoint_Metrcs", help="Path to the metrics folder.")

args = parser.parse_args()
metrics_folder = args.metrics_folders

analysis = metrics_folder + "/analysis_results"

# Recalculate metrics for all directories in the metrics folder
Recalc(metrics_folder)

'''file = "/home/s2334356/data1/Checkpoint_Metrcs/b_32_L_18_20250612_182811_SP500_10"
process_results_layers(file, stride=5, log_option=None)'''

# Process results and generate images for each directory
full_images(metrics_folder)

# Make a big image with all the quant_metrics.pdf files in the metrics folder
'''make_big_images(metrics_folder, "quant_metrics.pdf")
make_big_images(metrics_folder, "ACF_abs_gen.pdf")
make_big_images(metrics_folder, "leverage_gen.pdf")'''
# Make analysis of the metrics folder
#make_analysis(metrics_folder)

# Make a 3D plot of the cost function
#make_3d_plot(analysis)

