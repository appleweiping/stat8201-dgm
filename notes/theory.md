# Theory notes — STAT8201 Deep Generative Models

Concise derivations for the concepts implemented in this repo. STAT8201 is a PhD seminar
organized around paper readings; these notes cover the core theory behind each model
family we built, following the papers on the course reading list.

---

## 1. Variational Autoencoders and the ELBO (Kingma & Welling 2014)

We want to fit a latent-variable model `p_θ(x) = ∫ p_θ(x|z) p(z) dz` but the marginal
likelihood is intractable. Introduce an approximate posterior `q_φ(z|x)` and note, for any
`q`,

```
log p_θ(x) = E_{q_φ(z|x)}[ log p_θ(x,z) - log q_φ(z|x) ]  +  KL( q_φ(z|x) || p_θ(z|x) )
           = ELBO(x)                                       +  KL(·||·) ≥ ELBO(x)
```

because the KL term is ≥ 0. Maximizing the ELBO both fits the model and drives `q_φ`
toward the true posterior. The ELBO splits into a reconstruction term and a prior-matching
term:

```
ELBO(x) = E_{q_φ(z|x)}[ log p_θ(x|z) ] - KL( q_φ(z|x) || p(z) ).
```

**Reparameterization trick.** To get low-variance gradients through the sampling of
`z ~ q_φ(z|x) = N(μ, σ²)`, write `z = μ + σ ⊙ ε`, `ε ~ N(0, I)`. Now the randomness is
independent of `φ`, so `∇_φ` passes through `μ, σ` deterministically. For Gaussian
`q` and prior, the KL is closed form:

```
KL( N(μ,σ²) || N(0,I) ) = -½ Σ_j ( 1 + log σ_j² - μ_j² - σ_j² ).
```

Implemented in `a1_vae/model.py` (`reparameterize`, `elbo`).

## 2. Tighter bounds: the IWAE estimator (Burda et al. 2016)

The ELBO is loose when `q` is a poor posterior. The importance-weighted bound uses `k`
samples:

```
log p(x) ≥ E[ log (1/k) Σ_i w_i ],   w_i = p(x|z_i)p(z_i) / q(z_i|x),  z_i ~ q(z|x),
```

which is monotonically tighter in `k` and → `log p(x)` as `k → ∞`. We use it only as an
**evaluation** metric (`iwae_log_likelihood` in `a1_vae/train.py`), reporting a tighter
estimate of the test log-likelihood than the ELBO. This is the "Improving the ELBO: better
bounds" topic of the course.

## 3. Normalizing flows and change of variables (Dinh et al. 2017, RealNVP)

If `f` is an invertible, differentiable map and `z = f(x)` with a simple base density
`p_z`, the change-of-variables formula gives the **exact** likelihood:

```
log p_x(x) = log p_z(f(x)) + log |det ∂f/∂x|.
```

RealNVP builds `f` from **affine coupling layers**. Split `x = (x_a, x_b)` via a binary
mask; leave `x_a` unchanged and set `y_b = x_b ⊙ exp(s(x_a)) + t(x_a)`. The Jacobian is
triangular, so `log|det| = Σ s(x_a)` — cheap to compute, and the layer is trivially
invertible: `x_b = (y_b - t(x_a)) ⊙ exp(-s(x_a))`. Stacking layers with alternating masks
lets every coordinate be transformed. We use a 2D spatial checkerboard mask so the flow
respects image structure.

**Image preprocessing.** Pixel values are discrete; a continuous density would place
infinite mass on them. We *dequantize* (add uniform noise) and apply a `logit` transform to
`(0,1)→R`, tracking the transform's Jacobian so reported bits/dim refer to pixel space.
(Note: the logit Jacobian is large on near-binary MNIST, which can drive the reported
bits/dim below zero — an artifact of continuous-density modeling of nearly-binary data, not
an error; the Jacobian is verified against autograd in `tests/`.)

## 4. Autoregressive models: PixelCNN (van den Oord et al. 2016)

Any joint factorizes exactly by the chain rule in a fixed (raster) order:

```
p(x) = Π_i p(x_i | x_{<i}).
```

PixelCNN parameterizes each conditional with a CNN whose receptive field is restricted to
already-seen pixels via **masked convolutions**. A "type A" mask (first layer) excludes the
current pixel; "type B" masks (deeper layers) include it. This yields an exact, tractable
likelihood computed in one forward pass; sampling is sequential (one pixel at a time). We
verify the masking is correct by checking that `∂ output_{ij} / ∂ input_{ij and future} = 0`
(`tests/test_ar_diffusion.py`).

## 5. Generative Adversarial Networks (Goodfellow et al. 2014; Radford et al. 2016)

A generator `G` maps noise `z ~ p(z)` to samples; a discriminator `D` estimates
`P(real)`. The minimax game is

```
min_G max_D  E_{x~data}[log D(x)] + E_{z}[log(1 - D(G(z)))].
```

For a fixed `G`, the optimal discriminator is `D*(x) = p_data(x) / (p_data(x) + p_G(x))`,
and plugging it back shows the generator minimizes the **Jensen–Shannon divergence** between
`p_data` and `p_G`. In practice we use the non-saturating generator loss
`max_G E[log D(G(z))]` to avoid vanishing gradients early in training. Implemented as the
`bce` objective in `a3_gan`.

## 6. Wasserstein GAN + gradient penalty (Arjovsky et al. 2017; Gulrajani et al. 2017)

JS divergence gives poor gradients when supports don't overlap. WGAN instead minimizes the
Earth-Mover (Wasserstein-1) distance, whose Kantorovich–Rubinstein dual is

```
W(p_data, p_G) = sup_{||f||_L ≤ 1}  E_{x~data}[f(x)] - E_{x~G}[f(x)],
```

where `f` (the "critic") must be 1-Lipschitz. WGAN-GP enforces the Lipschitz constraint
softly by penalizing the critic's gradient norm on interpolates between real and fake
points:

```
L_critic = E_G[f] - E_data[f] + λ · E_{x̂}[ (||∇_{x̂} f(x̂)||₂ - 1)² ].
```

The critic loss `E_data[f] - E_G[f]` is an estimate of the Wasserstein distance (a
meaningful, non-saturating training signal). This is the "Wasserstein GAN" week of the
course; implemented as the `wgan_gp` objective in `a3_gan`.

## 7. Denoising Diffusion Probabilistic Models (Ho et al. 2020)

A fixed forward process adds Gaussian noise over `T` steps; the marginal has a closed form

```
q(x_t | x_0) = N( √ᾱ_t · x_0 , (1 - ᾱ_t) I ),    ᾱ_t = Π_{s≤t}(1 - β_s).
```

The reverse process `p_θ(x_{t-1}|x_t)` is learned. Ho et al. reparameterize the reverse
mean in terms of the noise `ε` and show the variational bound reduces (up to weighting) to
the **simple denoising objective**

```
L = E_{x_0, t, ε} || ε - ε_θ(x_t, t) ||² ,   x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε.
```

Sampling starts from `x_T ~ N(0, I)` and iterates the learned reverse step. Our
`ε_θ` is a small U-Net with sinusoidal timestep embeddings (`a5_diffusion/model.py`).
Diffusion post-dates the 2019 offering of the course but is the natural continuation of the
deep-generative-models lineage the seminar traces, so it is included per the repo brief.

---

### References (from / adjacent to the STAT8201 reading list)
- Kingma & Welling, *Auto-Encoding Variational Bayes*, 2014.
- Burda, Grosse & Salakhutdinov, *Importance Weighted Autoencoders*, 2016.
- Dinh, Sohl-Dickstein & Bengio, *Density estimation using Real NVP*, 2017.
- van den Oord et al., *Pixel Recurrent Neural Networks / Conditional PixelCNN*, 2016.
- Goodfellow et al., *Generative Adversarial Nets*, 2014.
- Radford, Metz & Chintala, *Unsupervised Representation Learning with DCGANs*, 2016.
- Arjovsky, Chintala & Bottou, *Wasserstein GAN*, 2017.
- Gulrajani et al., *Improved Training of Wasserstein GANs*, 2017.
- Ho, Jain & Abbeel, *Denoising Diffusion Probabilistic Models*, 2020.
