import datetime
import os
import pickle
import time
from functools import partial
import orbax.checkpoint as ocp
import jax
import jax.numpy as jnp
import mctx
from flax import nnx
import optax
from typing import NamedTuple
import external.pgx.pgx as pgx
from external.pgx.pgx.experimental import auto_reset
from external.pgx.pgx import State
from pydantic import BaseModel
from omegaconf import OmegaConf
import wandb


class Config(BaseModel):
    env_id: pgx.EnvId = "connect_five"
    seed: int = 0
    max_num_iters: int = 1000

    selfplay_batch_size: int = 2048
    num_simulations: int = 400
    max_num_step: int = 81

    training_batch_size: int = 256
    learing_rate: float = 0.001
    eval_interval: int = 5


    class Config:
        extra = "forbid"

config_dict = OmegaConf.from_cli()
config: Config = Config(**config_dict)
print(config)

class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray

class AZNet(nnx.Module):
  def __init__(self, num_actions: int, din: int, dmid: int, nlayers: int, rngs: nnx.Rngs):
    self.din = din
    self.layers = [
        nnx.Linear(din, dmid, rngs=rngs),
        *[nnx.Linear(dmid, dmid, rngs=rngs) for _ in range(nlayers - 1)]
    ]
    self.linear_out = nnx.Linear(dmid, num_actions + 1, rngs=rngs)

  def __call__(self, x):
      shape = x.shape
      x = jnp.reshape(x, (*shape[:-3], self.din))
      for layer in self.layers:
          x = nnx.gelu(layer(x))
      x = self.linear_out(x)
      return x[:, :-1], nnx.tanh(x[:, -1])  # logits, value


def forward_fn(x):
   net = AZNet(num_actions=9, din=81, dmid=81, nlayers=2)
   policy_out, value_out = net(x)
   return policy_out, value_out

env = pgx.make("connect_five")

forward = jax.jit(forward_fn)
optimizer = optax.adam(learning_rate=0.001)


def loss_fn(model, data: Sample):
    logits, value = model(data.obs)
    policy_loss = optax.softmax_cross_entropy(logits=logits, labels=data.policy_tgt).mean()
    value_loss = jnp.mean((value - data.value_tgt) ** 2).mean()
    loss = policy_loss + value_loss
    return loss, (logits, value, policy_loss, value_loss)

def train(model, opt_state, data: Sample):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, value, policy_loss, value_loss)), grads = grad_fn(model, data)
    opt_state.update(grads)
    return opt_state, policy_loss, value_loss

class SelfplayOutput(NamedTuple):
    obs: jnp.ndarray
    reward: jnp.ndarray 
    terminated: jnp.ndarray
    actions_weights: jnp.ndarray



def make_recurent_fn(baseline = False):
    def recurent_fn(model, rng_key, action, state):
        del rng_key
        current_player = state.current_player
        state = jax.vmap(env.step)(state, action)

        logits, value = model(state.observation)
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        
        discount = -1.0 * jnp.ones_like(value)
        discount = jnp.where(state.terminated, 0.0, discount)

        rewards = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
        value = jnp.where(state.terminated, 0.0, value)
        logits = jnp.where(baseline, state.legal_action_mask.astype(jnp.float32), logits)
        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=rewards,
            discount=discount,
            prior_logits=logits,
            value=value
        )
        return recurrent_fn_output, state
    
    return recurent_fn

def self_play(model, rng_key: jnp.ndarray):
    batch_size = config.selfplay_batch_size

    def step_fn(state : State, key) -> SelfplayOutput:
        key1, key2 = jax.random.split(key)
        observations = state.observation
        logits, value = model(observations)

        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

        policy_output = mctx.gumbel_muzero_policy(
            params=model,
            rng_key=key1,
            root=root,
            recurrent_fn=make_recurent_fn(),
            num_simulations=config.num_simulations,
            max_depth=81,
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )

        actor = state.current_player
        keys = jax.random.split(key2, batch_size)
        state = jax.vmap(auto_reset(env.step, env.init))(state, policy_output.action, keys)
        return state, SelfplayOutput(
            obs=state.observation,
            actions_weights=policy_output.action_weights,
            reward=state.rewards[jnp.arange(state.rewards.shape[0]), actor],
            terminated=state.terminated
        )

    rng_key, sub_key =  jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    state = jax.vmap(env.init)(keys)
    key_seq = jax.random.split(rng_key, config.max_num_step)
    _, data = jax.lax.scan(step_fn, state, key_seq)


    return data

def compute_loss_input(data: SelfplayOutput) -> Sample:
    batch_size = config.selfplay_batch_size
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1
    def body_fn(carry, i):
        ix = config.max_num_step - i - 1
        v = data.reward[ix] + carry
        return v, v
    
    _, value_tgt = jax.lax.scan(
        body_fn,
        jnp.zeros(batch_size),
        jnp.arange(config.max_num_step)
    )
    value_tgt = value_tgt[::-1, :]

    return Sample(
        obs=data.obs,
        policy_tgt=data.actions_weights,
        value_tgt=value_tgt,
        mask=value_mask
    )



def baseline(state, rng_key):
    root = mctx.RootFnOutput(prior_logits=state.legal_action_mask.astype(jnp.float32), value=jnp.ones(state.legal_action_mask.shape[0]), embedding=state)
    
    policy_output = mctx.gumbel_muzero_policy(
        params=model,
        root=root,
        recurrent_fn=make_recurent_fn(baseline=True),
        num_simulations=config.num_simulations,
        max_depth=81,
        invalid_actions=~state.legal_action_mask,
        qtransform=mctx.qtransform_completed_by_mix_value,
        gumbel_scale=1.0,
        rng_key=rng_key,
    )
    return policy_output.action



def evaluate(rng_key, my_model):
    my_player = 0
    opponent = 1
    key, sub_key = jax.random.split(rng_key)
    batch_size = config.selfplay_batch_size
    keys = jax.random.split(sub_key, batch_size)
    state = jax.vmap(env.init)(keys)

    def body_fn(val):
        key, state, R = val
        (my_logits, _) = my_model(state.observation)
        my_logits = my_logits - jnp.max(my_logits, axis=-1, keepdims=True)
        my_logits = jnp.where(state.legal_action_mask, my_logits, jnp.finfo(my_logits.dtype).min)
        key, sub_key = jax.random.split(key)
        my_actions = jax.random.categorical(sub_key, my_logits, axis=-1)

        key, sub_key = jax.random.split(key)
        opp_action = baseline(state, sub_key)
        
        is_my_turn = (state.current_player == my_player)
        
        actions = jnp.where(is_my_turn, my_actions, opp_action)
        key, sub_key = jax.random.split(key)
        keys = jax.random.split(sub_key, batch_size)
        state = jax.vmap(env.step)(state, actions, keys)

        R = R + state.rewards[jnp.arange(batch_size), my_player]

        return (key, state, R)
    

    _, _, R = jax.lax.while_loop(
        lambda x: ~(x[1].terminated.all()),
        body_fn,
        (key, state, jnp.zeros(batch_size))
    )
    return ((1 + R) / 2).mean()



if __name__ == "__main__":
    wandb.init(project="connect_five", config=config.model_dump())
    rng = nnx.Rngs(config.seed)
    rng_key = jax.random.key(config.seed)
    # Initialize the model and optimizer
    model = AZNet(num_actions=9, din=81 * 2, dmid=81, nlayers=2, rngs=rng)
    opt_state = nnx.Optimizer(model, optimizer)

    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    now = now.strftime("%Y%m%d%H%M%S")
    ckpt_dir = ocp.test_utils.erase_and_create_empty('/tmp/checkpoints', now)

    iteration: int = 0
    frames: int = 0
    log = {"iteration": iteration, "frames": frames}

    
    while True:
        if iteration >= config.max_num_iters:
            break

        st = time.time()


        # Eval against random agent
        if iteration % config.eval_interval == 0:
            env = pgx.make(config.env_id)
            key, sub_key = jax.random.split(rng_key)
            R = evaluate(sub_key, model)
            log.update({
                "winrate": R.item(),
            })
            _, state = nnx.split(model)
            checkpointer = ocp.StandardCheckpointer()
            checkpointer.save(ckpt_dir / f'state_{iteration}', state)

        # Self-play
        rng_key, sub_key = jax.random.split(rng_key)
        data : SelfplayOutput = self_play(model, sub_key)
        samples : Sample = compute_loss_input(data) 

        samples = jax.device_get(samples)
        frames += samples.obs.shape[0] * samples.obs.shape[1]
        samples = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), samples)
        rng_key, sub_key = jax.random.split(rng_key)

        ixs = jax.random.permutation(sub_key, jnp.arange(samples.obs.shape[0]))


        samples = jax.tree_util.tree_map(lambda x: x[ixs], samples)

        num_update = samples.obs.shape[0] // config.training_batch_size
        minibatches =  jax.tree_util.tree_map(
            lambda x: x.reshape((num_update, -1, *x.shape[1:])),
            samples
        )



        # Training

        policy_losses, value_losses = [], []

        for i in range(num_update):
            minibatch: Sample = jax.tree_util.tree_map(lambda x: x[i], minibatches)
            opt_state, policy_loss, value_loss = train(model, opt_state, minibatch)
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

        policy_loss = jnp.mean(jnp.array(policy_losses))
        value_loss = jnp.mean(jnp.array(value_losses))
        log.update({
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "frames": frames,
            "iteration": iteration,
        })
        wandb.log(log)
        iteration += 1