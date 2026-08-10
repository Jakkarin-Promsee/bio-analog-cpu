# 12 — Recurrent Visual Attention

> **Prerequisites**: [[1-transformer]], [[7-vit-foundation]], [[11-perciever-and-more]] **Pairs with**: [[13-latent-space-and-shortcuts]] **Feeds into**: [[s2-opened-topic-ideas]]

---

## 0. The question this whole line of work is answering

> _If you cannot afford to look at everything, how do you decide what to look at — and how do you learn that decision?_

This note builds the full mechanism from scratch: the sensor, the networks, the gradient problem, the four families of solutions, and where the line stands today.

---

## 1. Why the problem exists at all

### 1.1 The scaling problem

For an image of size $H \times W$, a standard CNN or ViT costs $O(HW)$. A $1000 \times 1000$ image costs 100× a $100 \times 100$ image — even when the object of interest occupies 5% of the frame.

Different architectures answer this differently:

| Architecture                              | Strategy                                                                            | Compute                               |
| ----------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------- |
| CNN / ViT                                 | process every pixel/patch                                                           | $O(N)$                                |
| **Perceiver** ([[11-perciever-and-more]]) | read everything **once** through cross-attention into a latent bottleneck $M \ll N$ | $O(NM)$, but depth decouples from $N$ |
| **RAM (Mnih 2014)**                       | **never read everything** — read $T$ small patches                                  | $O(T k g^2)$ — **independent of $N$** |

RAM's claim is the radical one: **compute is constant with respect to image size**. That, not accuracy, is the paper's actual selling point.

### 1.2 Biological motivation

The human fovea covers roughly 2° of visual angle at high acuity; the periphery is coarse. We compensate with saccades — 3–4 eye movements per second. Evolution chose _"read little, read often"_ over _"read everything at once"_.

---

## 2. Formal setup: a POMDP, not a supervised problem

This is the structural break from everything in [[1-transformer]] and [[7-vit-foundation]]: RAM is a **control problem**.

| Term                  | Meaning                                                              |
| --------------------- | -------------------------------------------------------------------- |
| Environment           | the static image $x$                                                 |
| Agent                 | the model                                                            |
| Partial observability | the agent **never sees $x$ in full** — only what it chose to look at |

At time $t$ the true state is

$$s_t = (x,\ l_1, \rho_1,\ \ldots,\ l_{t-1}, \rho_{t-1})$$

| Symbol   | Meaning                                       | Type                                     |
| -------- | --------------------------------------------- | ---------------------------------------- |
| $x$      | the image                                     | $\mathbb{R}^{H \times W \times C}$       |
| $l_t$    | **location** — where to look next, normalised | $[-1,1]^2$, where $(0,0)$ = image centre |
| $\rho_t$ | what was actually observed at that location   | $\mathbb{R}^{k g_w^2 C}$                 |
| $T$      | number of glimpses before answering           | integer, typically 4–8                   |

Everything the agent knows must be compressed into a single hidden state $h_t$. This is _why_ an RNN is needed — not aesthetics, but because **it is the only memory available**.

---

## 3. The glimpse sensor $\rho(x, l)$ — the read-bandwidth limit

This is the physical heart of the model.

$$\rho(x, l_{t-1}) = \Big[\ \text{resize}\big(\text{crop}(x, l_{t-1}, g_w)\big),\ \ \text{resize}\big(\text{crop}(x, l_{t-1}, 2 g_w)\big),\ \ \ldots,\ \ \text{resize}\big(\text{crop}(x, l_{t-1}, 2^{k-1} g_w)\big)\ \Big]$$

### Every symbol

| Symbol                 | Meaning                                         | Typical value                      |
| ---------------------- | ----------------------------------------------- | ---------------------------------- |
| $x$                    | input image                                     | 60×60 (cluttered translated MNIST) |
| $l_{t-1}$              | crop centre in normalised coords                | e.g. $(0.3, -0.5)$                 |
| $g_w$                  | width of the **finest** patch                   | 8                                  |
| $k$                    | number of scales                                | 3                                  |
| $C$                    | channels                                        | 1 (grayscale) or 3                 |
| $\text{crop}(x,l,s)$   | extract an $s \times s$ square centred at $l$   | —                                  |
| $\text{resize}(\cdot)$ | downsample every patch back to $g_w \times g_w$ | —                                  |

### The intuition to lock in

All $k$ patches share the same centre but double in width each time. After resizing them all to the same size:

- the small patch preserves **fine detail**
- the large patch preserves **coarse context**

This is a crude log-polar / foveal encoding. The output dimension is

$$\dim(\rho) = k \cdot g_w^2 \cdot C = 3 \times 64 \times 1 = 192$$

**192 dimensions, constant, whether the image is 60×60 or 6000×6000.** This is the _read-bandwidth limit_ — a deliberate design constraint, not an accident.

> **Note for later**: the coarse peripheral patch is not decoration. It is the mechanism that tells the model where to jump next. See §11.3.

---

## 4. Glimpse network $f_g$ — merging "what" and "where"

$$g_t = f_g(x, l_{t-1}; \theta_g)$$

Expanded:

$$h^g = \mathrm{ReLU}\big(W_g,\rho(x, l_{t-1}) + b_g\big)$$ $$h^l = \mathrm{ReLU}\big(W_l, l_{t-1} + b_l\big)$$ $$g_t = \mathrm{ReLU}\big(W_{g2} h^g + W_{l2} h^l + b\big)$$

| Symbol | Meaning                                 | Shape              |
| ------ | --------------------------------------- | ------------------ |
| $h^g$  | **"what" pathway** — what was seen      | $\mathbb{R}^{128}$ |
| $h^l$  | **"where" pathway** — where it was seen | $\mathbb{R}^{128}$ |
| $W_g$  | retina projection                       | $128 \times 192$   |
| $W_l$  | coordinate projection                   | $128 \times 2$     |
| $g_t$  | glimpse feature vector                  | $\mathbb{R}^{256}$ |

### Why feed $l_{t-1}$ in at all

The same patch means different things depending on where it came from. A curve at the top-left corner and the identical curve at the centre carry different evidence about which digit is present.

This is **positional information, exactly analogous to positional encoding** in [[8-position-transformer]] — except here it is a _continuous coordinate the agent chose itself_, not a fixed index.

_(Variant: some implementations use $g_t = h^g \odot h^l$, the multiplicative form. The original paper uses the additive form. Both work.)_

---

## 5. Core network $f_h$ — the only place information accumulates

$$h_t = f_h(h_{t-1}, g_t; \theta_h) = \mathrm{ReLU}\big(W_{hh} h_{t-1} + W_{gh} g_t\big)$$

| Symbol                     | Meaning                                           |
| -------------------------- | ------------------------------------------------- |
| $h_t \in \mathbb{R}^{256}$ | the agent's accumulated belief after $t$ glimpses |
| $W_{hh}$                   | recurrent weight — remember                       |
| $W_{gh}$                   | input weight — absorb                             |

$h_t$ must do **two jobs with one vector**:

1. accumulate evidence ("this is probably a 7")
2. track where it has already looked and what remains suspicious

Job 2 is the hard one, and it is why vanilla-RNN RAM only handles easy tasks. Harder variants swap in an LSTM. See §14 for why this bottleneck is ultimately fatal.

---

## 6. The two heads — and why one is called "action"

### 6.1 Code level: they are tiny

```python
self.action_net   = nn.Linear(256, 10)   # h_t -> class logits
self.location_net = nn.Linear(256, 2)    # h_t -> (x, y)
```

That is the entirety of both networks. All the difficulty of the paper lies in **how to obtain a gradient for `location_net`**, because its output never appears in the loss directly.

### 6.2 Conceptual level: two kinds of action

The paper is written in RL language. An agent takes **actions**, and RAM has two fundamentally different kinds:

|                   | **Location action** $l_t$        | **Environment action** $a_t$ |
| ----------------- | -------------------------------- | ---------------------------- |
| Changes           | what the agent **will see next** | the **world** / the answer   |
| Paper's term      | sensor action                    | environment action           |
| In classification | move the gaze                    | **predict the digit**        |
| In a game         | move the gaze                    | press a button               |
| Effect on reward  | **indirect** (better evidence)   | **direct**                   |

RAM was not designed as an image classifier. It was designed as **an agent with limited vision**, then tested on classification. In the paper's dynamic-environment experiment, `action_net` controls a real actuator.

> When reading the paper, every time you see the word _action_, ask: **sensor action or environment action?** This single habit makes the paper readable.

### 6.3 The asymmetry that defines everything

$$\text{action} \xrightarrow{\ \text{straight into loss}\ } \mathcal{L} \qquad\qquad \text{location} \xrightarrow{\ \text{via what gets seen}\ } \cdots \xrightarrow{\quad} \mathcal{L}$$

**Action network:**

- feeds cross-entropy immediately: $\mathcal{L}_{cls} = -\log p(y \mid h_T)$
- has **ground truth** (we know the label)
- → clean backprop, low variance

**Location network:**

- outputs $\mu_t$ → sampled to $l_t$ → tells the sensor where to crop → affects $\rho_t$ → accumulates into $h_t$ → eventually affects loss
- has **no ground truth at all** — "where should you look" is not in the dataset
- the backprop path is **broken** (§7)
- → only REINFORCE remains

> `action_net` learns from **the correct answer**. `location_net` learns only from **a good outcome**. That is the whole difference between supervised learning and RL.

### 6.4 Details people get wrong

**(a) `location_net` does not output a location — it outputs a policy mean.**

$$\underbrace{\mu_t = \tanh(W_{loc} h_t + b_{loc})}_{\text{what the network computes}} \qquad \underbrace{l_t \sim \mathcal{N}(\mu_t, \sigma^2 I_2)}_{\text{the actual location, sampled}}$$

The policy is the **whole distribution**, not a number. At **test time the sampling is dropped** and $\mu_t$ is used directly.

$\sigma$ is a **fixed hyperparameter** (~0.15), not learned. It is the single most sensitive knob in the model.

**(b) `action_net` is also a stochastic policy in principle.** In pure RL it would be $a_T \sim \mathrm{softmax}(\cdot)$ trained by REINFORCE. The paper _cheats_ by using cross-entropy because labels exist. This is the "hybrid loss" and it matters enormously — it pulls the vast majority of parameters out of the high-variance regime.

**(c) `action_net` fires once at $t=T$; `location_net` fires every step.** With $T=6$: 6 location decisions, 1 answer. This is why variance scales with $T$.

**(d) `h_t.detach()` into `location_net`.** The original paper does _not_ detach. Many implementations do, because noisy REINFORCE gradients otherwise corrupt the representation $h_t$ is trying to learn. **Start detached**; it trains far more easily.

---

## 7. Where the gradient dies — two separate layers

A common but imprecise summary is _"choosing where to look is a discrete choice, therefore non-differentiable."_ This is wrong in a way that leads to misunderstanding STN.

$l_t$ is **continuous** ($\mathbb{R}^2$). The real problem has two independent layers.

### Layer 1: `crop` is not differentiable with respect to $l$

```python
cx = int((l[0] + 1) / 2 * W)          # <-- here
patch = x[cy-g//2 : cy+g//2, cx-g//2 : cx+g//2]
```

Indexing an array requires rounding to integers. The map $l \mapsto \rho(x, l)$ is a **step function**: nudge $l$ by 0.1 px and the patch does not change at all (derivative 0); cross a pixel boundary and it jumps (derivative undefined).

$$\frac{\partial \rho(x,l)}{\partial l} = 0 \quad \text{almost everywhere}$$

**This is the real killer** — discreteness of the _pixel grid_, not of the _choice_.

### Layer 2: sampling from a distribution

Even with layer 1 fixed, $l_t \sim \mathcal{N}(\mu_t, \sigma^2)$ is a sampling operation with no naive gradient from sample back to mean.

### Why the distinction matters

- **STN fixes layer 1** (bilinear interpolation)
- **Reparameterisation / Gumbel fix layer 2**
- **RAM fixes neither** → REINFORCE, which sidesteps both at once

**Crucially**: the Gaussian in RAM was _always_ reparameterisable — $l = \mu + \sigma\varepsilon$ is trivially differentiable. It was useless only because the downstream $\partial f / \partial l = 0$. **Layer 1 is the sole blocker for RAM.** Layer 2 becomes a genuine barrier only for truly _discrete_ choices, where no smooth reparameterisation exists.

---

## 8. The objective and REINFORCE, derived from scratch

### 8.1 Reward and objective

$$R = \sum_{t=1}^{T} r_t, \qquad r_t = \begin{cases} 1 & t = T \text{ and the prediction is correct} \ 0 & \text{otherwise}\end{cases}$$

The policy induces a distribution over **trajectories** $\tau = (l_1, \rho_1, \ldots, l_T, a_T)$:

$$p(\tau; \theta) = \prod_{t=1}^{T} \pi(l_t \mid h_t; \theta)\ p(\rho_t \mid x, l_t)$$

| Symbol                      | Meaning                                                         |
| --------------------------- | --------------------------------------------------------------- |
| $\tau$                      | one full sequence of looks                                      |
| $\pi(l_t \mid h_t; \theta)$ | the **policy**                                                  |
| $p(\rho_t \mid x, l_t)$     | the environment — deterministic and **independent of $\theta$** |

$$J(\theta) = \mathbb{E}_{\tau \sim p(\tau;\theta)}\big[R(\tau)\big] = \int p(\tau;\theta), R(\tau), d\tau$$

**$\theta$ lives inside the distribution, not inside a differentiable path to $R$.** Ordinary backprop requires $\theta$ inside the _function computing the value_, not the _function assigning probability_.

### 8.2 The log-derivative trick

$$\nabla_\theta J = \int \nabla_\theta p(\tau;\theta), R(\tau), d\tau$$

This is not an expectation, so Monte Carlo cannot touch it. Apply the identity

$$\nabla_\theta \log p = \frac{\nabla_\theta p}{p} \quad\Longrightarrow\quad \nabla_\theta p = p, \nabla_\theta \log p$$

$$\nabla_\theta J = \int p(\tau;\theta), \nabla_\theta \log p(\tau;\theta), R(\tau), d\tau = \boxed{\ \mathbb{E}_\tau\big[R(\tau), \nabla_\theta \log p(\tau;\theta)\big]\ }$$

Expanding the log:

$$\log p(\tau;\theta) = \sum_{t=1}^{T} \log \pi(l_t \mid h_t;\theta) + \underbrace{\sum_t \log p(\rho_t \mid x, l_t)}_{\theta\text{-independent} \to \text{vanishes}}$$

**The environment drops out of the equation entirely.** We never differentiate through `crop`. Layer 1 is simply bypassed.

$$\nabla_\theta J \approx \frac{1}{M} \sum_{i=1}^{M} \sum_{t=1}^{T} \nabla_\theta \log \pi(l_t^i \mid h_t^i; \theta), R^i$$

$M$ = Monte Carlo samples, obtained by **replicating the same image $M$ times** in the batch and letting each replica sample its own trajectory.

### 8.3 The Gaussian policy gradient — the concrete formula

$$\pi(l_t \mid \mu_t) = \frac{1}{2\pi\sigma^2}\exp\left(-\frac{\lVert l_t - \mu_t \rVert^2}{2\sigma^2}\right)$$

$$\log \pi = -\frac{\lVert l_t - \mu_t \rVert^2}{2\sigma^2} - \log(2\pi\sigma^2)$$

$$\nabla_{\mu_t} \log \pi = \frac{l_t - \mu_t}{\sigma^2}$$

then chain-rule through $\mu_t = \tanh(W_{loc} h_t)$ as usual.

**Read it in plain language:**

$$\Delta \mu_t \ \propto\ \underbrace{(l_t - \mu_t)}_{\text{the noise that was sampled}} \times \underbrace{R}_{\text{did it work?}}$$

- noise pushed right **and the answer was correct** → move $\mu$ right
- answer wrong ($R = 0$) → **no update at all**

This is pure trial-and-error. The model never learns _where it should look_; it learns only _that looking roughly there happened to work_.

### 8.4 Numeric proof that REINFORCE is correct

Smallest possible problem: two choices, $\pi(z{=}1) = \sigma(\theta)$, rewards $f(1)=1$, $f(0)=0$, evaluated at $\theta = 0$ so $p = 0.5$.

**Ground truth:** $$J(\theta) = \sigma(\theta), \qquad \frac{dJ}{d\theta}\Big|_{0} = \sigma(1-\sigma) = 0.25$$

**REINFORCE:** $\nabla_\theta \log\pi(1) = 1-\sigma = 0.5$, $\nabla_\theta \log\pi(0) = -\sigma = -0.5$

| Sampled | Prob | $f(z)$ | $f \cdot \nabla\log\pi$ |
| ------- | ---- | ------ | ----------------------- |
| $z=1$   | 0.5  | 1      | $+0.5$                  |
| $z=0$   | 0.5  | 0      | $0$                     |

$$\mathbb{E}[\hat g] = 0.5(0.5) + 0.5(0) = 0.25 \quad\checkmark$$

**Exactly correct, without ever differentiating $f$.** Note the per-sample values are ${+0.5, 0}$ — every individual estimate is wrong; only the mean is right.

**Now add a baseline $b = 0.5$:**

| Sampled | $(f - b)\cdot\nabla\log\pi$ |
| ------- | --------------------------- |
| $z=1$   | $(1-0.5)(0.5) = 0.25$       |
| $z=0$   | $(0-0.5)(-0.5) = 0.25$      |

$$\mathbb{E} = 0.25 \ \checkmark \qquad \mathrm{Var} = \mathbf{0}$$

Same mean, variance eliminated. Note that the $z=0$ case went from _learning nothing_ to _"this option is below average — reduce its probability."_ Both outcomes now carry information.

### 8.5 Why one random direction is enough

This is the deepest point about REINFORCE. With small $\sigma$, Taylor-expand:

$$R(\mu + \sigma\varepsilon) \approx R(\mu) + \sigma, \underbrace{g^\top \varepsilon}_{\text{a scalar}}, \qquad g \equiv \nabla_\mu R$$

The baseline-corrected estimator becomes

$$\hat g = \frac{R(\mu + \sigma\varepsilon) - R(\mu)}{\sigma},\varepsilon \approx (g^\top \varepsilon),\varepsilon$$

and since $\varepsilon \sim \mathcal{N}(0, I)$ gives $\mathbb{E}[\varepsilon\varepsilon^\top] = I$:

$$\mathbb{E}_\varepsilon\big[(g^\top\varepsilon)\varepsilon\big] = \mathbb{E}[\varepsilon\varepsilon^\top],g = \boxed{,g,}$$

> Measure the slope **along one random direction**, multiply that scalar back onto the direction — **on average this reconstructs the true gradient**.

No need to try every direction: symmetric random sampling _performs the averaging for you_. Each sample gives a rank-1 estimate; averaging across thousands of SGD steps assembles the full gradient. This formula is exactly **SPSA / Evolution Strategies** — REINFORCE and ES are close relatives.

### 8.6 Worked example of the averaging

Target lies to the right ($+x$); model is at $\mu = (0,0)$ and knows nothing.

| #   | $\varepsilon$  | Outcome | $\varepsilon \cdot R$ |
| --- | -------------- | ------- | --------------------- |
| 1   | $(+1.2, +0.3)$ | correct | $(+1.2, +0.3)$        |
| 2   | $(-0.8, +0.5)$ | wrong   | $(0,0)$               |
| 3   | $(+0.4, -1.1)$ | correct | $(+0.4, -1.1)$        |
| 4   | $(-1.5, -0.2)$ | wrong   | $(0,0)$               |
| 5   | $(+0.9, +0.9)$ | correct | $(+0.9, +0.9)$        |
| 6   | $(+0.2, +1.4)$ | wrong   | $(0,0)$               |

$$\hat g = \tfrac{1}{6}(2.5,\ 0.1) = (0.42,\ 0.017)$$

Points right; the $y$ component correctly cancels. Six binary signals, no derivative of $f$ anywhere.

---

## 9. Variance: why this is hard

### 9.1 The scaling law

Writing $l_t - \mu_t = \sigma\varepsilon_t$:

$$\hat g = \frac{1}{\sigma}\sum_{t=1}^{T}\varepsilon_t R \qquad\Longrightarrow\qquad \mathbb{E}\lVert\hat g\rVert^2 \approx \frac{T, d}{\sigma^2},\mathbb{E}[R^2]$$

| Factor                 | Effect on variance                            |
| ---------------------- | --------------------------------------------- |
| $T$ (glimpses)         | **linear** — more looks, harder training      |
| $d$ (action dimension) | **linear** — RAM is lucky, $d = 2$            |
| $\sigma$               | **$1/\sigma^2$ — explodes as $\sigma \to 0$** |
| $\mathbb{E}[R^2]$      | larger/noisier reward is worse                |

$$\text{samples needed} \sim O(d) \quad\text{— linear in dimension, NOT exponential in } T$$

| Task              | $d$       | REINFORCE viable?                            |
| ----------------- | --------- | -------------------------------------------- |
| RAM (coordinates) | **2**     | comfortably                                  |
| robot control     | 20–50     | with many tricks                             |
| LLM weights       | $10^{11}$ | **impossible** — this is why backprop exists |

### 9.2 The $\sigma$ trap is two-sided

- $\sigma$ **too small** → variance $\propto 1/\sigma^2$ explodes, and no exploration → collapses to a local optimum (usually "always look at the centre")
- $\sigma$ **too large** → garbage patches → the classifier cannot learn

Practical range 0.1–0.2. **This is the first knob to tune.**

### 9.3 The hidden credit-assignment problem

The paper uses future return $R_t = \sum_{t' \geq t} r_{t'}$, but since reward exists only at $T$:

$$R_1 = R_2 = \cdots = R_T = R$$

**Every glimpse receives an identical signal.** If glimpse 3 was the only useful one, the other five useless glimpses are reinforced equally. This — not the $1/\sigma^2$ factor — is the true source of the difficulty.

### 9.4 Baselines

$$\nabla_\theta J \approx \frac{1}{M}\sum_i \sum_t \nabla_\theta \log\pi(l_t^i \mid h_t^i),\big(R_t^i - b_t\big)$$

**Unbiasedness proof** (this is why you can subtract anything action-independent):

$$\mathbb{E}_\tau\big[b,\nabla_\theta \log p(\tau;\theta)\big] = b\int p,\frac{\nabla_\theta p}{p},d\tau = b,\nabla_\theta\int p,d\tau = b,\nabla_\theta(1) = 0$$

**How much it helps.** With accuracy $p$ (Bernoulli reward):

- no baseline: $\mathbb{E}[R^2] = p$
- optimal baseline $b = \mathbb{E}[R] = p$: $\mathbb{E}[(R-b)^2] = p(1-p)$

At $p = 0.9$: from $0.9$ to $0.09$ — **10× variance reduction**, and it improves as the model improves. This is not a minor trick; it is the difference between trainable and not.

The paper uses a learned per-timestep baseline:

$$b_t = w_b^\top \mathrm{sg}(h_t) + c_b, \qquad \mathcal{L}_{base} = \sum_t (b_t - R_t)^2$$

($\mathrm{sg}$ = stop-gradient; the baseline should not shape the representation.)

---

## 10. The full hybrid loss

$$\mathcal{L}_{total} = \underbrace{-\log p(y \mid h_T)}_{\text{(1) cross-entropy}} \underbrace{- \sum_t \log\pi(l_t \mid h_t)\ \mathrm{sg}(R_t - b_t)}_{\text{(2) REINFORCE surrogate}} + \underbrace{\sum_t \big(b_t - \mathrm{sg}(R_t)\big)^2}_{\text{(3) baseline regression}}$$

| Part | Trains                                                  | Gradient type                   |
| ---- | ------------------------------------------------------- | ------------------------------- |
| (1)  | $\theta_a, \theta_h, \theta_g$ — **~99% of parameters** | **true backprop**, low variance |
| (2)  | $\theta_{loc}$ — a $2\times256$ matrix, **~1%**         | REINFORCE, high variance        |
| (3)  | $w_b, c_b$                                              | ordinary regression             |

> **The pattern to carry forward**: RL is not a whole-system choice. It is a **patch over the one hole backprop cannot cross — made as small as possible.** RLHF does the same thing: pretrain everything with backprop, apply RL only in a thin layer at the end.

```python
mu = torch.tanh(loc_net(h.detach()))        # detach is a stabiliser, optional
l  = mu + sigma * torch.randn_like(mu)
logpi = Normal(mu, sigma).log_prob(l.detach()).sum(-1)

adv  = (R - b).detach()
loss = F.cross_entropy(logits, y) - (logpi * adv).mean() + F.mse_loss(b, R.detach())
```

---

## 11. Making it differentiable: the interpolation view

### 11.1 Images as continuous functions

Rather than treating an image as an array, treat it as a continuous function via convolution with a kernel:

$$\tilde{U}(u) = \sum_m U_m \cdot k(u - m)$$

| Symbol     | Meaning                          |
| ---------- | -------------------------------- |
| $U_m$      | pixel value at integer index $m$ |
| $u$        | a **fractional** coordinate      |
| $k(\cdot)$ | the **kernel** — a design choice |

The condition that brightness is preserved has a name — **partition of unity**:

$$\sum_m k(u - m) = 1 \quad \forall u$$

### 11.2 Kernel choice _is_ gradient choice

| Kernel $k(u)$                          | Known as               | $\tilde U$        | $d\tilde U/du$                  |
| -------------------------------------- | ---------------------- | ----------------- | ------------------------------- |
| box $\mathbb{1}[\lvert u\rvert < 0.5]$ | **naive crop**         | staircase         | **0 almost everywhere**         |
| hat $\max(0, 1-\lvert u\rvert)$        | **bilinear (STN)**     | piecewise linear  | piecewise constant, **nonzero** |
| Gaussian $e^{-u^2/2\sigma^2}$          | **DRAW**               | smooth $C^\infty$ | smooth, wide support            |
| sinc                                   | Shannon reconstruction | the "true" image  | unusable (infinite support)     |

**Naive `crop` is just interpolation with the worst possible kernel.** Nothing sacred about it.

### 11.3 Numeric example: 1D spike at index 2, $U = [0,0,1,0,0]$

**Box kernel:**

| $l$            | 0.5 | 1.4 | 1.6      | 2.3 |
| -------------- | --- | --- | -------- | --- |
| $\tilde U$     | 0   | 0   | 1        | 1   |
| $d\tilde U/dl$ | 0   | 0   | $\infty$ | 0   |

Useless everywhere.

**Hat kernel:** $\tilde U(l) = \max(0, 1 - \lvert l-2\rvert)$

| $l$            | 0.5   | 1.4    | 2.3  | 2.9  |
| -------------- | ----- | ------ | ---- | ---- |
| $\tilde U$     | 0     | 0.4    | 0.7  | 0.1  |
| $d\tilde U/dl$ | **0** | **+1** | $-1$ | $-1$ |

At $l=1.4$ the gradient says _"move right"_ — usable. At $l=0.5$ it is **still zero**: beyond 1 pixel there is nothing.

**Gaussian $\sigma = 1.5$:** gradient at $l=0.5$ is $+0.13$ — findable from far away, but the value is smeared.

### 11.4 The trap: differentiable ≠ optimisable

$$\frac{\partial \tilde U}{\partial l} = \text{the local image gradient}$$

**The location gradient knows only about texture immediately under the current patch.** It has no idea a digit sits in the opposite corner.

- crop starting on **empty background** → gradient exactly 0 → **stuck forever**
- the loss landscape over $l$ is **as rugged as the image itself**
- this is the **aperture problem** from optical flow, identically

And there is an unavoidable trade-off:

$$\text{wider kernel} \Rightarrow \text{gradient reaches further} \Rightarrow \text{image is blurrier} \Rightarrow \text{less information read}$$

**This is why DRAW learns $\sigma$**: blur while searching, sharpen while reading.

---

## 12. The wall — and how it is actually solved

### 12.1 The wall is real

Task: "find an eye." A face has two → $f(l)$ has **two local minima** separated by the bridge of the nose.

**Failure mode 1 — ridge.** Standing on the right eye, moving left makes $f$ worse, so $\partial f/\partial l$ **points back**. Gradient descent actively resists crossing.

**Failure mode 2 — plateau.** If the region between is smooth skin, $\partial\rho/\partial l = 0$ exactly. No resistance, but **no direction either**. Worse, because the optimiser cannot even detect it is stuck.

Formally: the loss landscape over $l$ is **non-convex with separated basins of attraction**. Not a bug — natural images are full of high-frequency content.

### 12.2 Reach costs compute — quadratically

To have a nonzero gradient at distance $d$, the kernel support must reach $d$, so you must read:

$$\text{read cost} = (S + 2d)^2$$

|                                 | reach $d$ | pixels read      |
| ------------------------------- | --------- | ---------------- |
| plain bilinear                  | 1         | $10^2 = 100$     |
| kernel width 8                  | 8         | $24^2 = 576$     |
| kernel width 16                 | 16        | $40^2 = 1{,}600$ |
| reach across half a 60×60 image | 30        | $68^2 = 4{,}624$ |
| just read the whole image       | —         | $3{,}600$        |

**Once you want image-spanning reach, it costs more than reading everything.** There is no way to have both global reach and $O(1)$ compute from a local differentiable kernel. This is a closed trade-off.

Contrast with RL: $\sigma = 1.0$ jumps anywhere in the image while still reading only $(S+1)^2 = 100$ pixels.

$$\boxed{\ \text{RL decouples \textbf{search radius} from \textbf{read bandwidth}. Relaxation cannot.}\ }$$

### 12.3 There is a further difference: averaging vs visiting

$$\text{wide kernel} = \text{the \textbf{average} of a region} \qquad \text{sample} = \text{the \textbf{actual value} at a point}$$

A wide kernel melts the two eyes into one blob — they become indistinguishable. A sample at distance $d$ sees the left eye at full sharpness. And in gradient terms: bilinear **extrapolates linearly**, valid only over short distances; RL **evaluates for real**, correct at any distance. On a rugged $f$, linear extrapolation is wrong no matter how large you make $d$.

### 12.4 The five real solutions

| Strategy                             | Mechanism                                                                                                                                                                                                                                                                                                                                            |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Coarse-to-fine**                | blur first → landscape becomes near-convex → anneal $\sigma$ down. Known as **graduated non-convexity / scale-space continuation**; identical to pyramidal Lucas–Kanade. This is why DRAW learning $\sigma$ matters.                                                                                                                                 |
| **2. Amortise, don't optimise**      | STN's localisation net sees the **whole image** and regresses coordinates in one shot — it **jumps, it does not walk**. Gradients are local _per image_, but the learned mapping is global _across the dataset_. This is why STN works despite the theory. (Corollary: initialise $A_\theta$ to identity so at least one image gives useful signal.) |
| **3. Read the periphery, then jump** | **RAM's own answer** — the coarse 32×32 patch shows the far object faintly. That information enters through the **forward path**, not the gradient. The model never crosses the wall; it _sees over it_.                                                                                                                                             |
| **4. Randomise**                     | $\sigma$ acts like temperature in simulated annealing. RL does not fear walls because it **teleports**.                                                                                                                                                                                                                                              |
| **5. Enumerate**                     | soft attention computes $f$ everywhere at once. No search → no local minima. Costs $O(N)$.                                                                                                                                                                                                                                                           |

> **The generalisation**: nobody actually solves "where to look" by gradient descent on $l$. $\partial f/\partial l$ is _always_ used for **local refinement only**; region selection always comes from another mechanism.

| Model               | Selects region via                       | Refines via                              |
| ------------------- | ---------------------------------------- | ---------------------------------------- |
| **RAM**             | peripheral patch + random jumps          | nothing (REINFORCE only)                 |
| **STN**             | localisation net on full image           | $\partial f/\partial l$ through bilinear |
| **DRAW**            | wide blur first (learned $\sigma$)       | shrink $\sigma$                          |
| **Deformable DETR** | **soft attention finds reference point** | bilinear offsets                         |
| **ViT**             | sees all patches at once                 | n/a                                      |

---

## 13. The differentiable machinery in full

### 13.1 Spatial Transformer Networks (Jaderberg 2015)

**Three components.**

**(1) Localisation net** — predicts transform parameters: $\theta = f_{loc}(U) \in \mathbb{R}^6$

**(2) Grid generator:**

$$\begin{pmatrix} x_i^s \ y_i^s \end{pmatrix} = A_\theta \begin{pmatrix} x_i^t \ y_i^t \ 1 \end{pmatrix} = \begin{bmatrix}\theta_{11} & \theta_{12} & \theta_{13} \ \theta_{21} & \theta_{22} & \theta_{23}\end{bmatrix}\begin{pmatrix} x_i^t \ y_i^t \ 1\end{pmatrix}$$

| Symbol                     | Meaning                                                 |
| -------------------------- | ------------------------------------------------------- |
| $(x_i^t, y_i^t)$           | output-pixel coordinate — a **fixed** grid              |
| $(x_i^s, y_i^s)$           | where to sample in the input — **fractional**           |
| $\theta_{13}, \theta_{23}$ | translation = **"where to look"** ← this is RAM's $l_t$ |
| $\theta_{11}, \theta_{22}$ | scale = **"how much to zoom"**                          |

The crop-and-zoom special case is $A_\theta = \begin{bmatrix} s & 0 & t_x \ 0 & s & t_y \end{bmatrix}$ — **exactly RAM's glimpse sensor with a learnable scale**.

**(3) Bilinear sampler — the actual hero:**

$$V_i = \sum_{n=1}^{H}\sum_{m=1}^{W} U_{nm}\ \max(0,\ 1 - \lvert x_i^s - m\rvert)\ \max(0,\ 1 - \lvert y_i^s - n\rvert)$$

and the derivative that unlocks everything:

$$\frac{\partial V_i}{\partial x_i^s} = \sum_n\sum_m U_{nm}\max(0, 1-\lvert y_i^s - n\rvert)\cdot\begin{cases}0 & \lvert m - x_i^s\rvert \geq 1 \ +1 & m \geq x_i^s \ -1 & m < x_i^s\end{cases}$$

**Why it works**: with rounding, moving $x^s$ by 0.1 px changes nothing. With bilinear, the _mixing weights_ shift by 0.1 → the output moves proportionally → the gradient carries direction. Subpixel interpolation is what makes a coordinate a genuinely continuous variable.

**Weaknesses**: gradient support is ±1 pixel (very local), and the localisation net reads the full image, so **it does not scale to megapixel inputs** and gives up RAM's compute advantage.

### 13.2 DRAW (Gregor 2015) — the middle ground

A grid of Gaussian filters instead of a hard crop:

$$F_X[i,a] = \frac{1}{Z_X}\exp\left(-\frac{(a - \mu_X^i)^2}{2\sigma^2}\right)$$

| Parameter    | Meaning                              |
| ------------ | ------------------------------------ |
| $(g_X, g_Y)$ | grid centre                          |
| $\delta$     | stride between filter centres (zoom) |
| $\sigma$     | filter width (blur) — **learned**    |
| $\gamma$     | intensity                            |

Fully differentiable, and because $\sigma$ is learned, the model controls its own wall height. **This is the point where RAM and STN meet**, and the direct ancestor of "let the latent choose its own resolution" (see [[s2-opened-topic-ideas]]).

### 13.3 Gumbel-Softmax — reparameterising the discrete

**Gumbel-max trick** (exact):

$$z = \arg\max_i\big[\log\pi_i + g_i\big], \qquad g_i = -\log(-\log u_i),\ u_i \sim U(0,1)$$

$g_i$ is $\theta$-independent → reparameterisation succeeded. But `argmax` is a step function — **the same problem as `crop` returns**. Relax it:

$$y_i = \frac{\exp\big((\log\pi_i + g_i)/\tau\big)}{\sum_j \exp\big((\log\pi_j + g_j)/\tau\big)}$$

| Symbol  | Meaning                                  |
| ------- | ---------------------------------------- |
| $\pi_i$ | probability of option $i$                |
| $g_i$   | Gumbel noise                             |
| $\tau$  | **temperature** — the bias/variance knob |

$\tau \to 0$: one-hot but exploding gradients. $\tau \to \infty$: smooth but meaningless. In practice **anneal** 1.0 → 0.1.

**Straight-through:**

```python
y_hard = onehot(y_soft.argmax(-1))
y = y_hard - y_soft.detach() + y_soft   # forward: hard | backward: soft
```

A deliberately **biased** gradient, traded for low variance.

### 13.4 Show, Attend and Tell (Xu 2015) — both worlds in one paper

**Hard attention:**

$$s_t \sim \mathrm{Multinoulli}\big({\alpha_{t,i}}_{i=1}^{L}\big), \qquad \hat z_t = \sum_{i} s_{t,i}, a_i$$

| Symbol         | Meaning                                     |
| -------------- | ------------------------------------------- |
| $a_i$          | CNN feature at location $i$ (of $L$)        |
| $\alpha_{t,i}$ | attention weight, $\sum_i \alpha_{t,i} = 1$ |
| $s_t$          | **one-hot** — a single location             |

Objective is a variational lower bound $\mathcal{L}_s = \sum_s p(s\mid a)\log p(y\mid s,a) \leq \log p(y \mid a)$, whose gradient has **two terms**:

$$\frac{\partial\mathcal{L}_s}{\partial W} = \sum_s p(s\mid a)\Big[\underbrace{\tfrac{\partial \log p(y\mid s,a)}{\partial W}}_{\text{pathwise}} + \underbrace{\log p(y\mid s,a)\tfrac{\partial\log p(s\mid a)}{\partial W}}_{\text{score function}}\Big]$$

Needed three stabilisers to train: moving-average baseline, **entropy bonus** $\lambda H[s]$, and 50% mixing with the argmax sample.

**Soft attention:**

$$\hat z_t = \mathbb{E}_{p(s_t\mid a)}[\hat z_t] = \sum_{i=1}^{L}\alpha_{t,i}, a_i$$

**Soft attention is the expectation of hard attention pushed inside the model.** Exact-but-noisy is traded for approximate-but-smooth — at the cost of computing all $L$ features, which destroys the only advantage hard attention had.

### 13.5 The estimator taxonomy — memorise this

| Method                         | Bias                    | Variance      | Requires                            | Use when                                             |
| ------------------------------ | ----------------------- | ------------- | ----------------------------------- | ---------------------------------------------------- |
| **Score function (REINFORCE)** | 0                       | **very high** | only the ability to evaluate reward | environment is a black box, fully non-differentiable |
| **Pathwise (reparam / STN)**   | 0                       | **very low**  | a differentiable path               | a continuous relaxation exists                       |
| **Gumbel-ST**                  | **yes**                 | low           | finite option set $K$               | genuinely discrete, bias acceptable                  |
| **Soft / expectation**         | 0 (different objective) | 0             | evaluating all options              | $K$ small enough to afford                           |

$$\text{If a continuous path can be built, build it. REINFORCE is the \textbf{last} resort.}$$

### 13.6 The unifying pattern: relaxation appears everywhere

| Hard operation              | Relaxation                 | Name               |
| --------------------------- | -------------------------- | ------------------ |
| `crop` by integer index     | bilinear / Gaussian filter | **STN, DRAW**      |
| $\arg\max$ over $K$ options | Gumbel-softmax             | **Gumbel-ST**      |
| attend to one patch         | weight all patches         | **soft attention** |
| pick one expert             | top-$k$ softmax gating     | **MoE router**     |

All four are the same idea: **replace a hard selection with a weighted blend.** The Transformer is line 3, and it won because GPUs make dense matmuls nearly free.

**RL is the escape hatch that requires no relaxation at all** — it is how you skip softmax's denominator, which is where the $O(N)$ comes from:

$$\text{softmax} = \text{"score everything, then weight"} \qquad \text{RL} = \text{"guess one, then check"}$$

---

## 14. Why RAM lost — honestly

The idea won; the mechanism lost. And it did **not** lose primarily because of REINFORCE.

### Reason 1 — hard sequential reads destroy parallelism

$$\text{glimpse } t \text{ must wait for } h_{t-1} \Longrightarrow \text{no parallelism at all}$$

|                  | FLOPs  | **Wall-clock on GPU**   |
| ---------------- | ------ | ----------------------- |
| RAM, 6 glimpses  | ~1,300 | **6 sequential rounds** |
| ViT, whole image | ~3,600 | **1 matmul**            |

**3× fewer FLOPs and still slower.** RAM saved the cheap resource and paid with the expensive one. **This is the real cause of death.**

### Reason 2 — it simply reads too little

192 dims × 6 glimpses ≈ 1,150 dimensions for an entire image. Fine for MNIST digits; hopeless for ImageNet textures spread across the frame.

### Reason 3 — $h_t$ is an unfixed bottleneck

One 256-d vector must hold accumulated evidence + a map of where it has looked + the plan for where to go. **This is the exact bottleneck attention was invented to remove.** RAM escaped the input bottleneck and ran straight into the memory bottleneck.

### Reason 4 — hardware lottery

From 2014–2020 GPUs grew faster than anyone expected, so **saving FLOPs had no value**. Everyone who traded parallelism for FLOPs lost, not just RAM.

### Reason 5 — the benchmark never demanded sequential reasoning

Cluttered MNIST is _"find the thing that isn't noise"_ — pure saliency. No step's target depends on what a previous step saw.

$$\text{RAM proved \textbf{selecting where to look} helps. It never proved \textbf{looking in sequence} helps.}$$

This is a crucial and under-stated limitation — and it is picked up in detail in [[13-latent-space-and-shortcuts]] §7.

---

## 15. Where the idea actually survived

The mechanism died; the **pointer** survived.

$$\text{content-based retrieval } O(N) \qquad\text{vs}\qquad \text{address-based retrieval } O(1)$$

|                                                                                                    | Query form                                         | Cost   |
| -------------------------------------------------------------------------------------------------- | -------------------------------------------------- | ------ |
| **Hopfield / attention** ([[4-hopfield-internal]], [[6-fixed-point-and-what-transformer-chasing]]) | "who resembles this?" — compare against everything | $O(N)$ |
| **RAM**                                                                                            | "fetch what is at address $l$"                     | $O(1)$ |

$$\text{attention} = \textbf{associative memory} \qquad\qquad \text{RAM} = \textbf{pointer}$$

**RAM predicts an address instead of searching content.** Address prediction costs the same no matter how large the memory is.

### Modern descendants

**Deformable attention (Zhu et al. 2021)** — the version that won:

$$\mathrm{DeformAttn}(z_q, \hat p_q, x) = \sum_{m=1}^{M} W_m\Big[\sum_{k=1}^{K} A_{mqk}\cdot W'_m, x(\hat p_q + \Delta p_{mqk})\Big]$$

with offsets and weights predicted linearly from the query, $x(\cdot)$ bilinear, complexity $O(NK)$, $K \ll N$.

| Component                 | RAM equivalent           |
| ------------------------- | ------------------------ |
| $z_q$                     | $h_t$ (query from state) |
| $\hat p_q$                | reference point          |
| $\Delta p_{mqk}$          | location network         |
| $x(\hat p + \Delta p)$    | **glimpse sensor**       |
| bilinear                  | makes it differentiable  |
| $K$ points instead of $N$ | $O(1)$ read              |

$$\text{Deformable attention} = \text{RAM minus the RNN, minus the RL, with } K \text{ reads in parallel}$$

**Attention Sampling (Katharopoulos & Fleuret, ICML 2019)** — mathematically the most elegant. Sample locations from an attention distribution computed on a **downsampled view**; because features are aggregated as an attention-weighted average, one obtains an **unbiased, minimum-variance estimator of the full model's gradient**, trainable with ordinary SGD and **explicitly without reinforcement learning or variational methods**. The same paper notes STN's localisation net operates on the full image and therefore does not scale to megapixels.

**Iterative Patch Selection (Bergner et al., ICLR 2023)** — latent cross-attention scores patches in no-gradient mode, keeps a top-$M$ buffer, re-embeds the survivors with gradients, aggregates via cross-attention. Fine-tuned on whole-slide images of up to 250k patches (>16 gigapixels) with **5 GB of VRAM at batch size 16**.

**LookWhere (2025)**, **LTRP** — patch ranking learned by self-supervision (measure reconstruction change when a patch is removed): finding _where to look_ **without any labels**.

Others: NTM/DNC (address- and content-based memory together), RAG and tool use (the LLM predicts a query then retrieves only what it needs — RAM at token granularity), KV-cache eviction, agents running `grep` instead of reading a repo.

$$\textbf{Every time a system predicts where to read and then reads only there, that is RAM.}$$

---

## 16. Relationship to Transformer, Perceiver, Hopfield

**RAM is cross-attention, not self-attention.** This distinction is structural, not pedantic.

|                 | Query from             | Key/Value from               |
| --------------- | ---------------------- | ---------------------------- |
| Self-attention  | the sequence itself    | **the same thing**           |
| Cross-attention | one place              | **another place**            |
| **RAM**         | $h_t$ (internal state) | **the image $x$ (external)** |

$$\mu_t = \tanh(W h_t) \ \longrightarrow\ \rho(x, l_t)$$

State queries image; image answers. **No pixel ever talks to another pixel in RAM.**

|               | Query         | Selection                  | Cost/read |
| ------------- | ------------- | -------------------------- | --------- |
| **RAM**       | $h_t$         | **RL** (hard, 1 point)     | $O(1)$    |
| **Perceiver** | latent $Z$    | softmax (soft, all points) | $O(N)$    |
| Cross-attn    | decoder state | softmax                    | $O(N)$    |
| **Self-attn** | every token   | softmax                    | $O(N^2)$  |

Self-attention is quadratic precisely because it does what RAM cannot: **let every part of the input talk to every other part**. RAM must compare A and B by looking at A, then looking at B, and _hoping $h_t$ remembered A_.

$$\text{self-attention: compare in parallel within one layer} \quad|\quad \text{RAM: compare across time through memory}$$

**Connection to [[2-hopfield]] / [[6-fixed-point-and-what-transformer-chasing]]**: modern Hopfield update $\xi^{new} = X,\mathrm{softmax}(\beta X^\top \xi)$ is soft attention — retrieval from all of memory at once. RAM retrieves **one at a time along a learned query trajectory** $\mu_1 \to \mu_2 \to \cdots$ instead of following energy descent. RAM is _"Hopfield where the agent chooses its own next probe."_

**Connection to [[11-perciever-and-more]]**: Perceiver bottlenecks on **number of latents**; RAM bottlenecks on **bits readable per step**. Both are learned read bandwidth, on different axes.

**Connection to Forward-Forward / Hebbian**: two families exist in the world —

| Family            | Signal source           | Examples                                    |
| ----------------- | ----------------------- | ------------------------------------------- |
| **Structural**    | exploit known structure | backprop, STN, soft attention               |
| **Correlational** | perturb and correlate   | **REINFORCE, ES, Hebbian, Forward-Forward** |

FF perturbs the **input**; REINFORCE perturbs the **action**. Same learning mechanism, same price (far less information per update), same benefit (no global differentiable path required).

---

## 17. Practical notes if implementing

**Order of work:**

1. Build the sensor as **bilinear from day one**, train with pure backprop. Get it working. Then you know later problems come from RL, not from bugs.
2. Only then swap in REINFORCE and watch the variance yourself.

**Datasets in order:** MNIST 28×28 (should exceed 98%; if not, it's a bug, not RL difficulty) → translated MNIST 60×60 → cluttered translated MNIST 60×60.

**What to monitor — accuracy is not enough:**

| Watch                                  | Danger sign                                                                                |
| -------------------------------------- | ------------------------------------------------------------------------------------------ |
| mean $\lVert\mu_t\rVert$               | → 0 means **location collapse to centre** (failure mode #1)                                |
| $\mathrm{Var}_t[\mu_t]$                | flat means it looks at the same place every step — the RNN is unused                       |
| $\lVert\hat g_{loc}\rVert$             | swinging 100× between batches means variance is exploding; raise $M$, don't lower $\sigma$ |
| $b_t$ vs $R_t$                         | if the baseline can't track, variance reduction isn't working                              |
| **plot $\mu_t$ overlaid on the image** | **do this from epoch 1 — it reveals problems faster than any metric**                      |

**Gotchas that have cost people weeks:**

- forgetting `detach()` on reward/advantage
- advantage normalisation per batch `(A - A.mean())/A.std()` — slightly biased, helps a lot, use it
- $M \geq 10$ MC samples per image during training; at test time use $\mu_t$ deterministically
- gradient clipping is mandatory, not optional
- start with $T = 2$ and grow (variance $\propto T$)

---

## 18. Status: Open / Pending / Closed

### 🔴 Closed

| Topic                                                  | Closed by                         | Why                                             |
| ------------------------------------------------------ | --------------------------------- | ----------------------------------------------- |
| hard glimpse + REINFORCE for classification            | STN, deformable attention         | strictly dominated                              |
| patch selection on gigapixel images                    | IPS, Attention Sampling           | 16 gigapixels on 5 GB already achieved          |
| sparse/deformable attention as an efficiency mechanism | Deformable DETR                   | now standard equipment, not a contribution      |
| zoom-in visual search on high-res images               | V\*/SEAL, DyFo, ZoomEye, CVSearch | training-free versions already exist            |
| "is hard attention cheaper in FLOPs"                   | —                                 | yes, and irrelevant; wall-clock is what matters |

### 🟡 Pending

| Topic                                                            | State                                                       |
| ---------------------------------------------------------------- | ----------------------------------------------------------- |
| Gumbel/relaxation for discrete reads inside LLM-scale systems    | active — Latent-SFT, Latent-GRPO, SofT-GRPO                 |
| active perception as sequential Bayesian experimental design     | emerging — "perceptual bandwidth bottleneck" framing, FOVEA |
| self-supervised learning of _where to look_                      | LTRP, LookWhere — early                                     |
| deformable/pointer reads outside 2D grids (video, 3D, documents) | thin                                                        |
| learned **scale/resolution** (not just position)                 | **DRAW did it in 2015; almost nobody followed**             |

### 🟢 Open

1. **Perceiver IO + pointer reads as a general-purpose backbone.** Deformable lives in detection; IPS lives in high-res classification. No well-known model unifies latent bottleneck + address-based reads as a _general_ architecture.
2. **Letting each latent choose its own resolution.** Everyone uses fixed scales. Learned, per-query zoom is RAM's multi-scale retina made adaptive — see [[s2-opened-topic-ideas]].
3. **Benchmarks that require sequential looking.** Almost all vision benchmarks are _wide-hard_ (more objects, bigger images), not _deep-hard_ (step $t$ depends on step $t-1$). This is discussed at length in [[13-latent-space-and-shortcuts]].
4. **Adaptive halting** — learning _when to stop looking_. Most work fixes $T$.
5. **Pointer reads for non-grid modalities.**

---

## 19. Reading order

**Core:**

1. Mnih et al. 2014, _Recurrent Models of Visual Attention_ — §3 "Training" is the real content; the architecture equations can be skimmed
2. Jaderberg et al. 2015, _Spatial Transformer Networks_ — the bilinear sampler and its derivative
3. Xu et al. 2015, _Show, Attend and Tell_ — soft vs hard in one controlled comparison

**Then:** 4. Gregor et al. 2015, _DRAW_ — learned $\sigma$, the missing middle ground 5. Katharopoulos & Fleuret 2019, _Processing Megapixel Images with Deep Attention-Sampling Models_ (arXiv 1905.03711) — **short, elegant, answers most open questions above** 6. Zhu et al. 2021, _Deformable DETR_ — the version that won 7. Bergner et al. 2023, _Iterative Patch Selection_ (arXiv 2210.13007) — code at github.com/benbergner/ips

---

## 20. The three sentences to remember

1. **RAM's real contribution is the framing**: _what to read_ is a decision that must be learned, not an architecture that is fixed.
2. **It lost because it optimised the wrong resource for its era** — it saved FLOPs (getting cheaper every year) and spent parallelism (the thing that mattered).
3. **The pointer survived everything.** Whenever a system predicts an address and reads only there — deformable attention, RAG, tool use, agents running `grep` — that is this idea, still running.
