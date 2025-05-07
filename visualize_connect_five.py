#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import pgx
import streamlit as st
import streamlit.components.v1 as components

# 3a) Configure wide layout
st.set_page_config(layout="wide")

# 1) Build env + jit'd fns
env = pgx.make("connect_five")
init = jax.jit(env.init)
step = jax.jit(env.step)

@jax.jit
def act_randomly_single(rng, legal_mask):
    logits = jnp.log(legal_mask.astype(jnp.float32))
    return jax.random.categorical(rng, logits, axis=-1)

# 2) Rollout & collect SVGs
key = jax.random.PRNGKey(42)
state = init(key)
states = [state]
while not (state.terminated or state.truncated):
    key, sub = jax.random.split(key)
    a = act_randomly_single(sub, state.legal_action_mask)
    state = step(state, a)
    states.append(state)

svgs = [s.to_svg() for s in states]

# 3b) Streamlit UI
st.title("Connect-Five Replay")
step_index = st.slider("Step", 0, len(svgs) - 1)

# wrap SVG in a div to allow scrolling if too large
svg_html = f'<div style="width:100%; height:calc(100vh - 150px); overflow:auto;">{svgs[step_index]}</div>'
components.html(svg_html, height=900, width=0, scrolling=True)