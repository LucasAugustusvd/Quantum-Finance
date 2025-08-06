import JAX_QUIMB_Composite_Critic as jmm
import jax
import jax.numpy as jnp
import sys
import argparse

parser = argparse.ArgumentParser(description="Train a quantum GAN model.")
parser.add_argument("--bond_dim", type=int, required=True, help="Bond dimension for the model.")
parser.add_argument("--layer", type=int, required=True, help="Number of layers for the model.")
parser.add_argument("--stride", type=int, default=5, help="Stride for the model.")
parser.add_argument("--n_qubits", type=int, default=10, help="Number of qubits.")
parser.add_argument("--epochs", type=int, default=8001, help="Number of training epochs.")
parser.add_argument("--save", type=bool, default=True, help="Whether to save the model.")
parser.add_argument("--path", type=str, default="/home/s2334356/data1/Checkpoint_Composites", help="Path to save metrics and weights.")
parser.add_argument("--time_inc", type=str, default="SP500", help="Time increment for the model.")
parser.add_argument("--ckpt_dir", type=str, default="", help="Checkpoint directory for the model.")
parser.add_argument("--resume_from", type=bool, default=False, help="Whether to resume training from a checkpoint.")
parser.add_argument("--alpha_acf", type=float, default=0.0, help="Weight for ACF loss.")
parser.add_argument("--alpha_leverage", type=float, default=0.0, help="Weight for leverage loss.")
parser.add_argument("--alpha_emd", type=float, default=1.0, help="Weight for EMD loss.")

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
    "resume_from": args.resume_from,
    "alpha_acf": args.alpha_acf,
    "alpha_leverage": args.alpha_leverage,
    "alpha_emd": args.alpha_emd
}

bond_dim = args.bond_dim
layer = args.layer
print(f"Running with bond_dim {bond_dim} and layer {layer}")
if args.resume_from == True:
    print(f"Resuming from checkpoint in {args.ckpt_dir}")



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