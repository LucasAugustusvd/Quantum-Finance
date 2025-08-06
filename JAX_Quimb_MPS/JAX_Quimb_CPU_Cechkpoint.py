"""
Created on Thu Oct 12 11:23:45 2023

Name: JAX_QUIMB_CODE_CPU.py
Author: Lucas van Drooge
Location: Leiden, Netherlands
Email: lucasaugustusvd@gmail.com
GitHub: https://github.com/LucasAugustusvd
Description:
    This code is a JAX implementation of a quantum GAN model
    using the quimb library for quantum circuits and JAX for optimization.
    The model is designed to generate synthetic time series data that mimics
    the statistical properties of real financial data, specifically the S&P 500 index.
    The code includes functions for initializing parameters, applying the generator,
    and training the model using WGAN-GP loss functions.
"""

# Import necessary libraries
import quimb
import quimb.tensor as qtn
import data_handling as dh
import stylized as st
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime as dt
# Set the JAX backend to use float64 for higher precision
# This is important for numerical stability in quantum simulations
os.environ["JAX_ENABLE_X64"] = "true"
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax  # JAX's optimizer library
import time
from functools import partial
from quimb.tensor import CircuitMPS
from flax.training import checkpoints


def init_generator_params(key, n_layers: int, n_qubits: int):
    """
    Create a dictionary of trainable parameters for the generator:
      - 'circuit_params': shape (n_layers+1, n_qubits, 3)
      - 'alternating_weights': shape (2*n_qubits,)
      - 'lambdas': shape (n_qubits*n_layers,)

    Args:
        key: jax random key, used for parameter initialization
        n_layers: int, number of layers in the generator
        n_qubits: int, number of qubits in the circuit
    Returns:
        params: dict with
                - 'circuit_params': shape (n_layers+1, n_qubits, 3)
                - 'alternating_weights': shape (2*n_qubits,)
                - 'lambdas': shape (n_qubits*n_layers,)
    """
    # Split key for circuit vs. alternating vs. lambdas
    key_circ, key_alt, key_lambdas = jax.random.split(key, 3)
    
    # (a) Circuit parameters ~ N(0,0.01) 
    circuit_params = 0.01 * jax.random.normal(
        key_circ, 
        shape=(n_layers + 1, n_qubits, 3),
        dtype=jnp.float64
    )

    # (b) Alternating weights (initialized to 1)
    #     shape is (2*n_qubits,) so it can broadcast across the circuit output
    alt_weights = jnp.ones((2*n_qubits,), dtype=jnp.float64)
    
    # (c) Lambda scaling factors (initialized to 1)
    lambdas = jnp.ones((n_qubits * n_layers,), dtype=jnp.float64)

    return {
        'circuit_params': circuit_params,
        'alternating_weights': alt_weights,
        'lambdas': lambdas
    }

def generator_apply(params_gen, x_batch, n_layers: int, bond_dim=64):
    """
    Applies:
      1) The qubit circuit (via batched_quimb_circuit) with trainable lambdas.
      2) The 'Alternating' layer which multiplies 
         each output dimension by a learned weight.
      3) Reshapes to match your old final shape 
         (batch, 2*n_qubits, 1).
    
    Args:
      params_gen: dict with
         - 'circuit_params': shape (n_layers+1, n_qubits, 3)
         - 'alternating_weights': shape (2*n_qubits,)
         - 'lambdas': shape (n_qubits * n_layers,)
      x_batch: shape (batch, n_qubits)
      n_layers: int, number of layers in the generator
      bond_dim: int (optional) for MPS circuit contraction, default=64
    
    Returns:
      jnp array of shape (batch, 2*n_qubits, 1)
    """
    # 1) Evaluate the quantum circuit:
    #    shape = (batch, 2*n_qubits)
    circuit_out = batched_quimb_circuit(
        params_gen['circuit_params'], 
        x_batch, 
        params_gen['lambdas'],
        n_layers, 
        bond_dim
    )  # jnp.float64

    # 2) "Alternating" step = multiply each dimension by a learned weight
    scaled_out = circuit_out * params_gen['alternating_weights']

    # 3) Expand dims at the end: (batch, 2*n_qubits) -> (batch, 2*n_qubits, 1)
    final_out = jnp.expand_dims(scaled_out, axis=-1)

    return final_out


def single_quimb_circuit(params, x, lambdas, n_layers: int, bond_dim: int):
    """
    Applies a single qubit circuit to a single input x.
    Args:
      params: jnp array, shape (n_layers+1, n_qubits, 3)
      x: jnp array, shape (n_qubits,)
      lambdas: jnp array, shape (n_qubits * n_layers,)
      n_layers: int, number of layers in the circuit
      bond_dim: int, bond dimension for MPS circuit contraction
    Returns:
      jnp array, shape (2*n_qubits,)
    """
    n_qubits = x.shape[0]
    # Create an MPS-based circuit with JAX autodiff enabled:
    circ = CircuitMPS(
        n_qubits,
        psi0=None,
        gate_contract='auto-mps',
        gate_opts = {'method': 'svd', 'max_bond': bond_dim, 'cutoff':   0}, # cutoff=0 because we use bond_dim
        to_backend=jnp.array                 # ← critical for JAX gradients
        #convert_eager=True
    )

    # Apply the initial rotation layer:
    for layer in range(n_layers):
        for q in range(n_qubits):
            circ.rx(params[layer, q, 0], q)
            circ.ry(params[layer, q, 1], q)
            circ.rz(params[layer, q, 2], q)
        # Apply CNOT gates between qubits
        for q in range(n_qubits - 1):
            circ.cnot(q, q + 1)
        
        # Apply scaled data input, using lambdas to scale x
        for q in range(n_qubits):
            lambda_idx = layer * n_qubits + q
            scaled_x = lambdas[lambda_idx] * x[q]
            circ.rx(scaled_x, q)

    # Final rotation layer:
    for q in range(n_qubits):
        circ.rx(params[n_layers, q, 0], q)
        circ.ry(params[n_layers, q, 1], q)
        circ.rz(params[n_layers, q, 2], q)

    # Now measure:
    expecs = []
    for q in range(n_qubits):
        expecs.append(circ.local_expectation(quimb.pauli('Z'), q))
        expecs.append(circ.local_expectation(quimb.pauli('X'), q))

    return jnp.real(jnp.stack(expecs))


def batched_quimb_circuit(params, x_batch, lambdas, n_layers, bond_dim=64):
    """
    Vectorizes 'single_quimb_circuit' over a batch of classical inputs.
    
    Args:
      params: jnp array, shape (n_layers+1, n_qubits, 3), 
              parameters for the circuit
      x_batch: jnp array, shape (batch_size, n_qubits), 
               batch of classical inputs
      lambdas: jnp array, shape (n_qubits * n_layers,), 
               scaling factors for the circuit
      n_layers: int, number of layers in the circuit
      bond_dim: int, bond dimension for MPS circuit contraction
    
    Returns: 
      jnp array, shape (batch_size, 2*n_qubits)
    """
    return jax.vmap(
        single_quimb_circuit, 
        in_axes=(None, 0, None, None, None)
    )(params, x_batch, lambdas, n_layers, bond_dim)


class Critic(nn.Module):
    """
    1D CNN Critic for WGAN-GP.
    This is a Flax Module, so we can use the 'nn' module to define layers.

    """
    dropout_rate: float = 0.5

    # Define the model architecture
    @nn.compact
    def __call__(self, x, train: bool = True):
        """
        x shape: (batch_size, chopsize, 1)
        If train=True, use dropout; if False, no dropout.
        Returns shape: (batch_size, 1)
        """
        # 1) Convolution1D(64, kernel=10, padding='same')
        #    Flax conv wants kernel_size as an int tuple: (10,)
        x = nn.Conv(features=64, kernel_size=(10,), padding='SAME')(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        # 2) Convolution1D(128, kernel=10, padding='same')
        x = nn.Conv(features=128, kernel_size=(10,), padding='SAME')(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        # 3) Another Convolution1D(128, kernel=10, padding='same')
        x = nn.Conv(features=128, kernel_size=(10,), padding='SAME')(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        # Flatten
        x = x.reshape((x.shape[0], -1))  # shape [batch_size, ... everything else...]

        # Dense(32)
        x = nn.Dense(features=32)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        # Dropout(0.5)
        #   Flax dropout requires an RNG key from the 'apply' method 
        #   and a bool to say if we're training or not
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        # Dense(1)
        x = nn.Dense(features=1)(x)  # shape: (batch_size, 1)

        return x


def critic_loss_wgan_gp(disc_params, gen_params, real_batch, rng, 
                        critic_model, noise_dim, n_layers, chopsize, 
                        bond_dim, gradient_penalty_weight):
    """
    WGAN-GP critic loss = mean(D(fake)) - mean(D(real))
    + gradient penalty term

    Args:
        disc_params: Critic parameters
        gen_params: Generator parameters
        real_batch: Real data batch
        rng: JAX random key
        critic_model: Critic model
        noise_dim: Dimension of the noise input
        n_layers: Number of layers in the generator
        chopsize: Size of the input data
        bond_dim: Bond dimension for the MPS circuit
        gradient_penalty_weight: Weight for the gradient penalty term
    Returns:
        wgan_loss: WGAN loss value
    """
    batch_size = real_batch.shape[0]

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

    # 5) Gradient Penalty
    alpha = jax.random.uniform(rng_gp, shape=(batch_size,1,1), minval=0., maxval=1.)
    interpolates = real_out + alpha * (fake_out - real_out)

    # Split rng for critic application in gradient penalty
    rng_critic_gp = jax.random.fold_in(rng_gp, 0)
    
    def critic_apply(x):
        # Add rngs parameter for dropout
        return critic_model.apply({'params': disc_params}, x, train=True, rngs={'dropout': rng_critic_gp})

    def single_grad(inter_sample):
        # Compute the gradient of the critic output w.r.t. the input
        grad_fn = jax.grad(lambda inp: jnp.mean(critic_apply(inp[None, ...])))
        return grad_fn(inter_sample)

    # Compute the gradient for each sample in the batch
    # and reshape to (batch_size, chopsize, 1)

    grads = jax.vmap(single_grad, in_axes=0)(interpolates)
    norm_grads = jnp.sqrt(jnp.sum(grads**2, axis=(1,2)))
    gp = jnp.mean((norm_grads - 1.0)**2)
    gp_term = gradient_penalty_weight * gp

    return wgan_loss, wgan_loss + gp_term


def generator_loss_wgan(gen_params, disc_params, batch_size, rng,
                        critic_model, noise_dim, n_layers, chopsize, bond_dim):
    """
    WGAN generator loss = -mean(D(fake))
    Args:
        gen_params: Generator parameters
        disc_params: Critic parameters
        batch_size: Size of the batch
        rng: JAX random key
        critic_model: Critic model
        noise_dim: Dimension of the noise input
        n_layers: Number of layers in the generator
        chopsize: Size of the input data
        bond_dim: Bond dimension for the MPS circuit
    Returns:
        generator_loss: Generator loss value
    """
    # 1) Create noise for fake data
    rng, rng_noise = jax.random.split(rng)
    noise = jax.random.uniform(
        rng_noise,
        shape=(batch_size, noise_dim),
        minval=0.0, maxval=2*jnp.pi
    )
    # 2) Generate fake data
    fake_out = generator_apply(gen_params, noise, n_layers, bond_dim=bond_dim)
    fake_out = jnp.reshape(fake_out, (batch_size, chopsize, 1))
    # 3) Reshape fake data
    variables = {'params': disc_params}
    fake_logits = critic_model.apply(variables, fake_out, train=False)
    # 4) Compute generator loss
    return -jnp.mean(fake_logits)


@partial(jax.jit, static_argnames=("critic_model", "disc_optimizer", "noise_dim", "n_layers", "chopsize", "bond_dim", "gradient_penalty_weight"))
def train_critic_jax(disc_params, disc_opt_state, gen_params, real_batch, rng,
                     critic_model, noise_dim, n_layers, chopsize, bond_dim,
                     gradient_penalty_weight, disc_optimizer):
    """
    Single critic update in JAX, WGAN-GP
    Args:
        disc_params: Critic parameters
        disc_opt_state: Optimizer state for the critic
        gen_params: Generator parameters
        real_batch: Real data batch
        rng: JAX random key
        critic_model: Critic model
        noise_dim: Dimension of the noise input
        n_layers: Number of layers in the generator
        chopsize: Size of the input data
        bond_dim: Bond dimension for the MPS circuit
        gradient_penalty_weight: Weight for the gradient penalty term
        disc_optimizer: Optimizer for the critic
    Returns:
        disc_params: Updated critic parameters
        disc_opt_state: Updated optimizer state for the critic
        d_loss_val: Critic loss value
    """
    
    def loss_fn(d_p):
        # Compute the WGAN-GP loss
        wgan_loss, wgan_loss_full = critic_loss_wgan_gp(
            d_p, gen_params, real_batch, rng,
            critic_model, noise_dim, n_layers, chopsize, bond_dim,
            gradient_penalty_weight
        )
        return wgan_loss_full  # Return the full loss for optimization

    # Compute the loss and gradients
    d_loss_val, d_grads = jax.value_and_grad(loss_fn)(disc_params)
    updates, disc_opt_state = disc_optimizer.update(d_grads, disc_opt_state)
    disc_params = optax.apply_updates(disc_params, updates)
    # Return updated parameters and loss value
    return disc_params, disc_opt_state, d_loss_val


@partial(jax.jit, static_argnames=("critic_model", "gen_optimizer", "noise_dim", "n_layers", "chopsize", "bond_dim", "batch_size"))
def train_generator_jax(gen_params, gen_opt_state, disc_params,
                        batch_size, rng, critic_model, noise_dim,
                        n_layers, chopsize, bond_dim, gen_optimizer):
    """
    Single generator update in JAX, WGAN-GP
    Args:
        gen_params: Generator parameters
        gen_opt_state: Optimizer state for the generator
        disc_params: Critic parameters
        batch_size: Size of the batch
        rng: JAX random key
        critic_model: Critic model
        noise_dim: Dimension of the noise input
        n_layers: Number of layers in the generator
        chopsize: Size of the input data
        bond_dim: Bond dimension for the MPS circuit
        gen_optimizer: Optimizer for the generator
    Returns:
        gen_params: Updated generator parameters
        gen_opt_state: Updated optimizer state for the generator
        g_loss_val: Generator loss value
    """
    def loss_fn(g_p):
        return generator_loss_wgan(
            g_p, disc_params, batch_size, rng,
            critic_model, noise_dim, n_layers, chopsize, bond_dim
        )

    g_loss_val, g_grads = jax.value_and_grad(loss_fn)(gen_params)
    updates, gen_opt_state = gen_optimizer.update(g_grads, gen_opt_state)
    gen_params = optax.apply_updates(gen_params, updates)
    return gen_params, gen_opt_state, g_loss_val



class quantum_GAN(object):
    """
    Quantum GAN class for training a quantum generative adversarial network
    on time series data.
    """
    def __init__(
        self, 
        chopsize, 
        stride, 
        n_qubits, 
        n_layers, 
        epochs, 
        save=False,
        path='./', 
        time_inc="SP500.csv", 
        bond_dim=5,
        ckpt_dir: str = "",
        resume_from: bool = False
    ):
        """
        Initialize the Quantum GAN model.
        Args:
            chopsize: int, size of the input data
            stride: int, stride for data chopping
            n_qubits: int, number of qubits in the circuit
            n_layers: int, number of layers in the generator
            epochs: int, number of training epochs
            save: bool, whether to save the model weights and metrics
        """
        lr = 1e-3
        # current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print("Current directory:", current_dir)
        self.ckpt_dir = "Checkpoint_Metrcs/" + ckpt_dir
        self.resume_from = resume_from
        self.chopsize = chopsize
        self.stride = stride

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.epochs = int(epochs)
        self.save = save
        if time_inc == "SP500":
            self.log_returns_sp500 = dh.load_SP500_lr()
        elif time_inc == "weekly_data.csv":
            self.log_returns_sp500 = dh.load_SP500_lr_week_long()
        else:
            self.log_returns_sp500 = dh.load_SP500_lr_own(time_inc)
        self.lr = lr
        
        # Optax
        self.generator_optimizer = optax.adam(lr)
        self.discriminator_optimizer = optax.adam(lr)
        
        self.transformed_lr, self.transform_params = dh.transform(self.log_returns_sp500)
        self.noise_dim = n_qubits

        self.train_time_series = dh.chopchop(self.transformed_lr, chopsize, stride).astype('float64')
        self.train_time_series = self.train_time_series.reshape(
            self.train_time_series.shape[0], 
            self.train_time_series.shape[1]
        )
        self.BUFFER_SIZE = self.train_time_series.shape[0]
        self.BATCH_SIZE = self.train_time_series.shape[0] // 10

        self.dimension_image_alpha = None
        self.gradient_penalty_weight = 10.0
        self.nb_steps_update_critic = 5
        self.loss_gen = []
        self.loss_gen, self.loss_disc, self.loss_wass, self.loss_ACF, \
            self.loss_ACF_nonabs, self.loss_leverage, self.loss_epochs = \
                [], [], [], [], [], [], []
        
        self.benchmarks, self.benchmark_lags = st.benchmark(self.log_returns_sp500, chopsize - 2)
        
        if save == True and resume_from == False:
            timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
            self.path = path + '/{}'.format("b_" + str(bond_dim) + "_L_" + str(n_layers) 
                                            + "_" + timestamp + "_" + time_inc 
                                            + "_" + str(n_qubits))
            self.weight_path = self.path + '/weights'
            self.plot_path = self.path + '/plots'
            self.metrics_path = self.path + '/metrics'
            self.checkpoint_path = self.path + '/checkpoints'
            self.checkpoint_dup =  self.path + '/checkpoints_duplicate'
            # Create directories for saving the params of generator and critic and the transformed data in the the weights folder
            self.params_path_gen = self.weight_path + '/params_generator'
            self.params_path_critic = self.weight_path + '/params_critic'
            self.transformed_data_path = self.weight_path + '/transformed_data'


            os.mkdir(self.path)
            os.mkdir(self.checkpoint_path)
            os.mkdir(self.checkpoint_dup)
            os.mkdir(self.path + '/images')
            os.mkdir(self.weight_path)
            os.mkdir(self.plot_path)
            os.mkdir(self.metrics_path)
            os.mkdir(self.params_path_gen)
            os.mkdir(self.params_path_critic)
            os.mkdir(self.transformed_data_path)
        elif resume_from == True:
            self.path = self.ckpt_dir
            self.weight_path = self.ckpt_dir + '/weights'
            self.plot_path = self.ckpt_dir + '/plots'
            self.metrics_path = self.ckpt_dir + '/metrics'
            self.checkpoint_path = self.ckpt_dir + '/checkpoints'
            self.checkpoint_dup =  self.ckpt_dir + '/checkpoints_duplicate'
            self.params_path_gen = self.weight_path + '/params_generator'
            self.params_path_critic = self.weight_path + '/params_critic'
            self.transformed_data_path = self.weight_path + '/transformed_data'

        
        
        # We'll define these in train_model or externally
        self.critic_model = None
        self.critic_params = None
        self.generator_params = None
        self.bond_dim = bond_dim

    def train(self):
        # 2) If not yet initialized, do so
        if self.generator_params is None:
            key_init = jax.random.PRNGKey(0)
            self.generator_params = init_generator_params(key_init, self.n_layers, self.n_qubits)
        if self.critic_params is None:
            dummy_inp = jnp.ones((1, self.chopsize, 1), dtype=jnp.float32)
            vars_ = self.critic_model.init(jax.random.PRNGKey(42), dummy_inp, train=True)
            self.critic_params = vars_['params']

        gen_opt_state = self.generator_optimizer.init(self.generator_params)
        disc_opt_state = self.discriminator_optimizer.init(self.critic_params)
        state = {
            "gen_params": self.generator_params,
            "gen_opt":    gen_opt_state,  # optimizer state placeholder
            "disc_params": self.critic_params,
            "disc_opt":   disc_opt_state,
            "step": 0
            }
        
        # 1) Clear old logs
        self.loss_gen = []
        self.loss_disc = []
        self.loss_wass = []
        self.loss_ACF = []
        self.loss_ACF_nonabs = []
        self.loss_leverage = []
        self.loss_epochs = []

        # For random number generation in the loop:
        rng = jax.random.PRNGKey(1234)

        # 3) We'll replicate your 'noise_cst' for evaluation:
        rng, rng_eval = jax.random.split(rng)
        noise_cst = jax.random.uniform(
            rng_eval,
            shape=(self.train_time_series.shape[0], self.noise_dim),
            minval=0.0, maxval=2*jnp.pi
        )

        # 4) Build dataset approach
        total_size = self.train_time_series.shape[0]
        num_steps_epoch = total_size // (self.BATCH_SIZE * self.nb_steps_update_critic)

        ckpt_dir_boem = os.path.abspath(self.checkpoint_path)
        ckpt_duplictae_dir_boem = os.path.abspath(self.checkpoint_dup)

        # (Optional) print for debug so you can see what it's looking at:
        print("Restoring from checkpoint directory:", ckpt_dir_boem)
        # If we asked to resume, pull in the latest checkpoint
        if self.resume_from==True:
            state = checkpoints.restore_checkpoint(
                ckpt_dir=ckpt_dir_boem,
                target=state
            )

            print("Restored state keys:", state.keys())
            print("Step value in state:", state.get("step", "not found"))
            # Unpack back into your attributes
            self.generator_params = state["gen_params"]
            gen_opt_state         = state["gen_opt"]
            self.critic_params    = state["disc_params"]
            disc_opt_state        = state["disc_opt"]
            start_epoch = state["step"]
            print(f"Resuming from epoch {start_epoch}...")

            # If resuming, load the best metrics from the last run
            try:
                best_metrics = np.loadtxt(self.metrics_path + '/best_metrics.txt')
                lowest_wass = best_metrics[0]
                lowest_wass_epoch = int(best_metrics[1])
            except:
                print("No previous best metrics found, starting fresh.")
                lowest_wass = 1e5
                lowest_wass_epoch = 0
            try:
                lowest_acf_abs = np.loadtxt(self.metrics_path + '/loss_acf_abs.txt')[-1]
                lowest_acf_abs_epoch = int(np.loadtxt(self.metrics_path + '/loss_epochs.txt')[-1])
            except:
                print("No previous ACF abs metrics found, starting fresh.")
                lowest_acf_abs = 1e5
                lowest_acf_abs_epoch = 0
            try:
                lowest_acf_nonabs = np.loadtxt(self.metrics_path + '/loss_acf_nonabs.txt')[-1]
                lowest_acf_nonabs_epoch = int(np.loadtxt(self.metrics_path + '/loss_epochs.txt')[-1])
            except:
                print("No previous ACF non-abs metrics found, starting fresh.")
                lowest_acf_nonabs = 1e5
                lowest_acf_nonabs_epoch = 0
            try:
                lowest_leverage = np.loadtxt(self.metrics_path + '/loss_leverage.txt')[-1]
                lowest_leverage_epoch = int(np.loadtxt(self.metrics_path + '/loss_epochs.txt')[-1])
            except:
                print("No previous leverage metrics found, starting fresh.")
                lowest_leverage = 1e5
                lowest_leverage_epoch = 0

        else:
            start_epoch = 0
            lowest_wass = 1e5
            lowest_acf_abs = 1e5
            lowest_acf_nonabs = 1e5
            lowest_leverage = 1e5
            lowest_wass_epoch = 0
            lowest_acf_abs_epoch = 0
            lowest_acf_nonabs_epoch = 0
            lowest_leverage_epoch = 0



        for epoch in range(start_epoch, self.epochs):
            start = time.time()
            print(f'Epoch {epoch+1}/{self.epochs}')
            rng, rng_perm = jax.random.split(rng)
            perm = jax.random.permutation(rng_perm, total_size)
            train_shuffled = self.train_time_series[perm]

            step_idx = 0
            for step in range(num_steps_epoch):
                start_i = step * self.BATCH_SIZE * self.nb_steps_update_critic
                end_i   = (step+1) * self.BATCH_SIZE * self.nb_steps_update_critic
                big_batch = train_shuffled[start_i:end_i]

                n_sub = self.nb_steps_update_critic
                sub_batch_size = self.BATCH_SIZE
                #print len (big_batch), len(big_batch)/n_sub
                #print big_batch.shape, sub_batch_size, n_sub
                for sub_step in range(n_sub):
                    sub_start = sub_step * sub_batch_size
                    sub_end   = (sub_step+1)* sub_batch_size
                    real_sub_batch = big_batch[sub_start:sub_end]

                    # Critic update
                    self.critic_params, disc_opt_state, d_loss_val = train_critic_jax(
                        self.critic_params,
                        disc_opt_state,
                        self.generator_params,
                        real_sub_batch,
                        rng,
                        self.critic_model,
                        self.noise_dim,
                        self.n_layers,
                        self.chopsize,
                        self.bond_dim,
                        self.gradient_penalty_weight,
                        self.discriminator_optimizer
                    )

                # Generator update
                self.generator_params, gen_opt_state, g_loss_val = train_generator_jax(
                    self.generator_params,
                    gen_opt_state,
                    self.critic_params,
                    sub_batch_size,
                    rng,
                    self.critic_model,
                    self.noise_dim,
                    self.n_layers,
                    self.chopsize,
                    self.bond_dim,
                    self.generator_optimizer
                )

                step_idx += 1

            # -- End epoch, do metrics
            gen_cst_out = generator_apply(
                self.generator_params, 
                noise_cst, 
                self.n_layers, 
                bond_dim=self.bond_dim
            )
            gen_cst_out_np = np.array(gen_cst_out)
            generated_data_transformed = dh.inverse_transform(
                gen_cst_out_np, 
                self.transform_params
            )

            # measure stylized metrics
            
            metric = st.metrics(
                generated_data_transformed, 
                self.log_returns_sp500,
                self.benchmark_lags, 
                self.benchmarks, 
                only_EMD=False
            )
            # metric => [acf_abs, acf_nonabs, leverage, EMD]
            self.loss_wass.append(metric[3])
            self.loss_ACF.append(metric[0])
            self.loss_ACF_nonabs.append(metric[1])
            self.loss_leverage.append(metric[2])
            self.loss_epochs.append(epoch)
            self.loss_gen.append(g_loss_val)
            self.loss_disc.append(d_loss_val)

            print(f"  Metrics => EMD: {metric[3]:.6f}, ACF: {metric[0]:.6f}, Lev: {metric[2]:.6f}")

            # check best
            if metric[3] < lowest_wass:
                lowest_wass = metric[3]
                lowest_wass_epoch = epoch
                if self.save:
                    np.save(self.weight_path + f"/lowest_wass_{epoch}_generator.npy", 
                            np.array(self.generator_params, dtype=object))
            if metric[0] < lowest_acf_abs:
                lowest_acf_abs = metric[0]
                lowest_acf_abs_epoch = epoch
                if self.save:
                    np.save(self.weight_path + f"/lowest_acf_abs_{epoch}_generator.npy", 
                            np.array(self.generator_params, dtype=object))
            if metric[1] < lowest_acf_nonabs:
                lowest_acf_nonabs = metric[1]
                lowest_acf_nonabs_epoch = epoch
                if self.save:
                    np.save(self.weight_path + f"/lowest_acf_nonabs_{epoch}_generator.npy", 
                            np.array(self.generator_params, dtype=object))
            if metric[2] < lowest_leverage:
                lowest_leverage = metric[2]
                lowest_leverage_epoch = epoch
                if self.save:
                    np.save(self.weight_path + f"/lowest_leverage_{epoch}_generator.npy", 
                            np.array(self.generator_params, dtype=object))

            if epoch%250 == 0:
                st.QQ_plot(generated_data_transformed, self.log_returns_sp500, 'QQ plot at epoch {}'.format(str(epoch)), xlabel = 'Quantiles generated', ylabel = 'Quantiles SP 500', limit = [-0.04,0.04], show = False, path = self.plot_path+'/QQ_plot_epoch_{}.pdf'.format(str(epoch)))

            if epoch%50 == 0:
                # save geneator and critic weights
                np.save(self.params_path_gen + f"/generator_{epoch}_generator.npy", 
                        np.array(self.generator_params, dtype=object))
                np.save(self.params_path_critic + f"/critic_{epoch}_generator.npy",
                        np.array(self.critic_params, dtype=object))
                # save generated data
                np.save(self.transformed_data_path + f"/generated_data_{epoch}_generator.npy", 
                        np.array(generated_data_transformed, dtype=object))
            # Save log
            
            print(f"Epoch {epoch+1}/{self.epochs} took {time.time() - start:.2f} seconds")
            state = {
                "gen_params": self.generator_params,
                "gen_opt":    gen_opt_state,
                "disc_params": self.critic_params,
                "disc_opt":   disc_opt_state,
                # also record which epoch we just finished
                "step": epoch,
                }
            # Save the state to a checkpoint
            # Save a checkpoint every 50 epochs and keep all such checkpoints
            try:
                if epoch % 50 == 0:
                    checkpoints.save_checkpoint(
                        ckpt_dir=ckpt_duplictae_dir_boem,
                        target=state,
                        step=epoch,
                        keep=200,  # Keep the last 5 checkpoints
                        overwrite=False
                    )
            except Exception as e:
                print(f"Error saving checkpoint at epoch {epoch}: {e}")
            

            checkpoints.save_checkpoint(
                ckpt_dir=ckpt_dir_boem,
                target=state,
                step=epoch,
                keep=5,
                overwrite=True
            )
        

            if self.save and self.resume_from == False:
                np.savetxt(self.metrics_path+'/loss_wass.txt', np.array(self.loss_wass))
                np.savetxt(self.metrics_path+'/loss_acf_abs.txt', np.array(self.loss_ACF))
                np.savetxt(self.metrics_path+'/loss_acf_nonabs.txt', np.array(self.loss_ACF_nonabs))
                np.savetxt(self.metrics_path+'/loss_leverage.txt', np.array(self.loss_leverage))
                np.savetxt(self.metrics_path+'/loss_epochs.txt', np.array(self.loss_epochs))
                np.savetxt(self.metrics_path+'/best_metrics.txt', np.array([lowest_wass, lowest_wass_epoch]))
                try:
                    np.savetxt(self.metrics_path+'/loss_gen.txt', np.array(self.loss_gen))
                    np.savetxt(self.metrics_path+'/loss_disc.txt', np.array(self.loss_disc))
                except Exception as e:
                    print(f"Error saving loss_gen or loss_disc: {e}")
            
            if self.save and self.resume_from == True:
                with open(self.metrics_path+'/loss_wass.txt', 'a') as f:
                    np.savetxt(f, np.array(self.loss_wass))
                with open(self.metrics_path+'/loss_acf_abs.txt', 'a') as f:
                    np.savetxt(f, np.array(self.loss_ACF))
                with open(self.metrics_path+'/loss_acf_nonabs.txt', 'a') as f:
                    np.savetxt(f, np.array(self.loss_ACF_nonabs))
                with open(self.metrics_path+'/loss_leverage.txt', 'a') as f:
                    np.savetxt(f, np.array(self.loss_leverage))
                with open(self.metrics_path+'/loss_epochs.txt', 'a') as f:
                    np.savetxt(f, np.array(self.loss_epochs))
                try:
                    with open(self.metrics_path+'/loss_gen.txt', 'a') as f:
                        np.savetxt(f, np.array(self.loss_gen))
                    with open(self.metrics_path+'/loss_disc.txt', 'a') as f:
                        np.savetxt(f, np.array(self.loss_disc))
                except Exception as e:
                    print(f"Error appending loss_gen or loss_disc: {e}")

                np.savetxt(self.metrics_path+'/best_metrics.txt', np.array([lowest_wass, lowest_wass_epoch]))

        print("Training complete.")
        print("Best EMD:", lowest_wass, "at epoch", lowest_wass_epoch)
        print("Best ACF_abs:", lowest_acf_abs, "at epoch", lowest_acf_abs_epoch)
        print("Best ACF_nonabs:", lowest_acf_nonabs, "at epoch", lowest_acf_nonabs_epoch)
        print("Best leverage:", lowest_leverage, "at epoch", lowest_leverage_epoch)
        
