import JAX_Quimb_CPU_Cechkpoint as jmm
import jax
import jax.numpy as jnp
import sys
import argparse
import os

parser = argparse.ArgumentParser(description="Train a quantum GAN model.")
parser.add_argument("--bond_dim", type=int, required=True, help="Bond dimension for the model.")
parser.add_argument("--layer", type=int, required=True, help="Number of layers for the model.")
parser.add_argument("--stride", type=int, default=5, help="Stride for the model.")
parser.add_argument("--n_qubits", type=int, default=10, help="Number of qubits.")
parser.add_argument("--epochs", type=int, default=3001, help="Number of training epochs.")
parser.add_argument("--save", type=bool, default=True, help="Whether to save the model.")
parser.add_argument("--path", type=str, default="/home/s2334356/data1/Checkpoint_Metrcs", help="Path to save metrics and weights.")
parser.add_argument("--time_inc", type=str, default="SP500", help="Time increment for the model.")
parser.add_argument("--ckpt_dir", type=str, default="", help="Checkpoint directory for the model.")
parser.add_argument("--resume_from", type=bool, default=False, help="Whether to resume training from a checkpoint.")

args = parser.parse_args()

gan_params = {
    "chopsize": 2*args.n_qubits,
    "stride": args.stride,
    "n_qubits": args.n_qubits,
    "n_layers": args.layer,
    "epochs": args.epochs,
    "save": args.save,
    "path": args.path,
    "time_inc": args.time_inc,
    "bond_dim": args.bond_dim,
    "ckpt_dir": args.ckpt_dir,
    "resume_from": args.resume_from
}

bond_dim = args.bond_dim
layer = args.layer

print(f"Running with bond_dim {bond_dim} and layer {layer}")
if args.resume_from == True:
    print(f"Resuming from checkpoint in {args.ckpt_dir}")
    print("Overwriting bond_dim and layer with checkpoint values.")
    # Load the checkpoint to verify bond_dim and layer
    # checkpoint format is b_1_L_5_20250517_152253_SP500_10
    # pattern = r"b_(\d+)_L_(\d+)_.*"
    pattern = r"b_(\d+)_L_(\d+)_.*"
    import re
    match = re.search(pattern, args.ckpt_dir)
    if not match:
        raise ValueError(f"Checkpoint directory {args.ckpt_dir} does not match expected format.")
    checkpoint_bond_dim = int(match.group(1))
    print(f"Checkpoint bond_dim: {checkpoint_bond_dim}")
    checkpoint_layer = int(match.group(2))
    print(f"Checkpoint layer: {checkpoint_layer}")
    bond_dim = checkpoint_bond_dim
    layer = checkpoint_layer
    if checkpoint_bond_dim != bond_dim or checkpoint_layer != layer:
        raise ValueError(f"Checkpoint bond_dim {checkpoint_bond_dim} and layer {checkpoint_layer} do not match provided bond_dim {bond_dim} and layer {layer}.")

def train_model(layers, bond_dim, gan_params):
    gan_params["n_layers"] = layers
    gan_params["bond_dim"] = bond_dim
    gan_instance = jmm.quantum_GAN(**gan_params)
    # init the critic model & params
    gan_instance.critic_model = jmm.Critic(dropout_rate=0.5)
    dummy = jnp.ones((1, gan_instance.chopsize, 1), dtype=jnp.float32)
    vars_ = gan_instance.critic_model.init(jax.random.PRNGKey(42), dummy, train=True)
    gan_instance.critic_params = vars_['params']
    # then run
    gan_instance.train()

train_model(layer, bond_dim, gan_params)