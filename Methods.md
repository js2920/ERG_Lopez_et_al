# Mathematical & Methodological Specifications

### Research Code Companion for Lopez et al., *Nature Cancer* (2026)
**Single-cell dissection of stemness programs and transcriptional plasticity in pediatric acute myeloid leukemia relapse**

---

## 1. Multi-Cohort Preprocessing & Feature Selection

### 1.1 Hierarchical Batch Allocation Rationale

Let cell $i$ belong to dataset $d_i \in \{\text{Healthy-HSPC}, \text{Fetal-Liver}, \text{pAML}\}$. The categorical batch covariate $b_i$ is assigned hierarchically:

```math
b_i = \begin{cases}
\text{library-id}_i, & \text{if } d_i \in \{\text{Healthy-HSPC}, \text{Fetal-Liver}\} \\
\text{aml-id}_i, & \text{if } d_i = \text{pAML}
\end{cases}
```

**Mathematical Rationale**: Standard batch correction techniques treat the batch covariate $b_i$ as a nuisance parameter whose variance must be projected out. In pediatric acute myeloid leukemia (pAML), clinical libraries are sequenced per collection timepoint ($\mathbf{x}_{\text{DX}}$ at diagnosis and $\mathbf{x}_{\text{REL}}$ at relapse). If the sequencing library $b_i = \text{library-id}_i$ were assigned as the batch key for disease cells, the indicator vector of clinical relapse would lie entirely within the linear span of the categorical batch indicators:

```math
\mathbf{1}_{\text{Relapse}} \in \text{span}(\mathbf{B})
```

Under this condition, removing batch variance mathematically eliminates the biological axis of temporal disease progression. 

By contrast, batching pAML cells on the patient identifier $b_i = \text{aml-id}_i$ preserves both biological contrasts of interest within each batch:
1. **Temporal relapse dynamics**: Diagnosis versus matched Relapse;
2. **Leukemic transformation**: Residual Normal bridge cells versus Malignant blasts.

Because both conditions are represented within every patient batch, variational inference removes inter-patient technical variation while strictly preserving the longitudinal trajectory of leukemic recurrence.

### 1.2 Multi-Compartment Rank-Aggregated Variance

Consensus features are determined across four biological compartments $\mathcal{C} = \{c_1, c_2, c_3, c_4\}$:
1. Circulating CD34+ HSPCs across the human lifespan (Furer/Shlush et al., *Nature Medicine* 2025; GEO: GSE285943);
2. Human fetal liver hematopoietic progenitors (Suo et al., *Science* 2022);
3. Pediatric AML normal bone marrow bridge cells (Lambo et al., *Cancer Cell* 2023; GEO: GSE235063);
4. Pediatric AML malignant progenitor blasts (Lambo et al., GEO: GSE235063).

Within each compartment $c$, raw counts are modeled using Seurat v3 variance-stabilizing transformation. Each gene $g$ is assigned an integer rank $\text{Rank}_c(g) \in [1, K]$, where $K = 3,000$. The consensus rank score is defined as:

```math
\text{Score}(g) = \sum_{c \in \mathcal{C}} \frac{K - \text{Rank}_c(g)}{K} \cdot \mathbb{I}(g \in \mathcal{V}_c)
```

where $\mathcal{V}_c$ denotes the set of detected transcripts in compartment $c$, and $\mathbb{I}(\cdot)$ is the indicator function. 

Technical artifacts (mitochondrial transcripts `MT-`, ribosomal structural proteins `RPL`/`RPS`, and nuclear non-coding RNAs `MALAT1`, `NEAT1`, `XIST`) are filtered out. Key hematopoietic regulatory factors (`ERG`, `CD34`, `GATA2`, `RUNX1`, `CEBPA`, `SPI1`) are explicitly retained, defining a consensus 4,000-gene space $\mathcal{G}_{\text{consensus}}$.

---

## 2. Deep Generative Latent Space Modeling (scVI & scANVI)

### 2.1 Generative Model Formulation

For cell $n \in \{1, \dots, N\}$ and gene $g \in \{1, \dots, G\}$:

1. **Latent cell state**:
```math
\mathbf{z}_n \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d), \quad d = 30
```

2. **Library size factor**:
```math
l_n \sim \text{LogNormal}(\mathbf{l}_\mu^\top \mathbf{b}_n, \, \mathbf{l}_\sigma^\top \mathbf{b}_n)
```
where $\mathbf{l}_\mu, \mathbf{l}_\sigma$ represent empirical batch-specific library scaling parameters.

3. **Neural network decoders** parameterize normalized relative expression proportions $\mathbf{\rho}_n \in \Delta^{G-1}$ and gene-specific dropout probabilities $\mathbf{\pi}_n \in [0, 1]^G$:
```math
\mathbf{\rho}_n = \text{softmax}(f_\rho(\mathbf{z}_n, \mathbf{b}_n))
```
```math
\mathbf{\pi}_n = \text{sigmoid}(f_\pi(\mathbf{z}_n, \mathbf{b}_n))
```

4. **Expected count rate**:
```math
\mu_{ng} = l_n \cdot \rho_{ng}
```

5. **Observed count**:
```math
x_{ng} \sim \text{ZINB}(\mu_{ng}, \theta_g, \pi_{ng})
```
where $\theta_g > 0$ is a gene-specific inverse dispersion parameter learned during training.

---

### 2.2 Zero-Inflated Negative Binomial (ZINB) Likelihood

The Probability Mass Function (PMF) of the ZINB distribution is defined as:

```math
p(x \mid \mu, \theta, \pi) = \pi \delta_0(x) + (1 - \pi) \frac{\Gamma(x + \theta)}{\Gamma(\theta) x!} \left( \frac{\theta}{\theta + \mu} \right)^\theta \left( \frac{\mu}{\theta + \mu} \right)^x
```

where $\delta_0(x) = 1$ if $x = 0$ and $0$ otherwise.

#### Piecewise Log-Likelihood Formulation

For numerical stability, let $\eta = \text{logit}(\pi) = \log\left(\frac{\pi}{1 - \pi}\right)$:

**Case 1: Observed Zero Count ($x = 0$)**
```math
\log p(0 \mid \mu, \theta, \pi) = -\text{softplus}(\eta) + \text{logsumexp}\left( \eta, \, \theta \log\left( \frac{\theta}{\theta + \mu} \right) \right)
```

**Case 2: Observed Non-Zero Count ($x > 0$)**
```math
\begin{aligned}
\log p(x \mid \mu, \theta, \pi) &= -\text{softplus}(\eta) + \log \Gamma(x + \theta) - \log \Gamma(\theta) - \sum_{k=1}^x \log(k) \\
&\quad + \theta \log\left( \frac{\theta}{\theta + \mu} \right) + x \log\left( \frac{\mu}{\theta + \mu} \right)
\end{aligned}
```

---

### 2.3 Evidence Lower Bound (ELBO) Derivation

The marginal log-likelihood for cell $n$ given batch covariate $\mathbf{b}_n$ is:

```math
\log p(\mathbf{x}_n \mid \mathbf{b}_n) = \log \iint p(\mathbf{x}_n, \mathbf{z}_n, l_n \mid \mathbf{b}_n) \, d\mathbf{z}_n \, dl_n
```

We introduce a factorized variational posterior parameterized by encoder neural networks $\phi$:

```math
q_\phi(\mathbf{z}_n, l_n \mid \mathbf{x}_n, \mathbf{b}_n) = q_\phi(\mathbf{z}_n \mid \mathbf{x}_n, \mathbf{b}_n) \cdot q_\phi(l_n \mid \mathbf{x}_n, \mathbf{b}_n)
```

Applying Jensen's inequality:

```math
\begin{aligned}
\log p(\mathbf{x}_n \mid \mathbf{b}_n) &= \log \mathbb{E}_{q_\phi} \left[ \frac{p(\mathbf{x}_n, \mathbf{z}_n, l_n \mid \mathbf{b}_n)}{q_\phi(\mathbf{z}_n, l_n \mid \mathbf{x}_n, \mathbf{b}_n)} \right] \\
&\ge \mathbb{E}_{q_\phi} \left[ \log \frac{p(\mathbf{x}_n, \mathbf{z}_n, l_n \mid \mathbf{b}_n)}{q_\phi(\mathbf{z}_n, l_n \mid \mathbf{x}_n, \mathbf{b}_n)} \right] \equiv \mathcal{L}_{\text{ELBO}}(\mathbf{x}_n)
\end{aligned}
```

Decomposition into reconstruction and divergence terms:

```math
\begin{aligned}
\mathcal{L}_{\text{ELBO}}(\mathbf{x}_n) &= \mathbb{E}_{q_\phi} \left[ \sum_{g=1}^G \log p(x_{ng} \mid \mathbf{z}_n, l_n, \mathbf{b}_n) \right] \\
&\quad - D_{\text{KL}}\left( q_\phi(\mathbf{z}_n \mid \mathbf{x}_n, \mathbf{b}_n) \parallel p(\mathbf{z}_n) \right) \\
&\quad - D_{\text{KL}}\left( q_\phi(l_n \mid \mathbf{x}_n, \mathbf{b}_n) \parallel p(l_n \mid \mathbf{b}_n) \right)
\end{aligned}
```

---

### 2.4 Analytical Kullback-Leibler Expressions

#### Latent Space Prior Divergence
```math
D_{\text{KL}}\left( \mathcal{N}(\mathbf{\mu}_z, \mathbf{\Sigma}_z) \parallel \mathcal{N}(\mathbf{0}, \mathbf{I}_d) \right) = \frac{1}{2} \sum_{j=1}^d \left( \sigma_{z,j}^2 + \mu_{z,j}^2 - 1 - \log(\sigma_{z,j}^2) \right)
```

#### Library Size Prior Divergence
```math
D_{\text{KL}}\left( \text{LogNormal}(\mu_l, \sigma_l^2) \parallel \text{LogNormal}(l_\mu, l_\sigma^2) \right) = \log\left(\frac{l_\sigma}{\sigma_l}\right) + \frac{\sigma_l^2 + (\mu_l - l_\mu)^2}{2 l_\sigma^2} - \frac{1}{2}
```

---

### 2.5 Semi-Supervised Cell-State Transfer (scANVI)

In scANVI, a categorical cell state $y_n \in \{1, \dots, K\}$ is introduced as an additional latent variable. 
- For reference cells (healthy circulating HSPCs and normal pAML bridge cells), $y_n$ is observed;
- For malignant AML blasts and validation holdouts, $y_n$ is unobserved ($y_n = \text{Unknown}$).

**Labelled Objective**:
```math
\begin{aligned}
\mathcal{L}_{\text{labelled}}(\mathbf{x}_n, y_n) &= \mathbb{E}_{q_\phi} \left[ \log p(\mathbf{x}_n \mid \mathbf{z}_n, l_n, \mathbf{b}_n) + \log p(\mathbf{z}_n \mid y_n) + \log p(l_n \mid \mathbf{b}_n) \right. \\
&\quad \left. - \log q_\phi(\mathbf{z}_n, l_n \mid \mathbf{x}_n, y_n, \mathbf{b}_n) \right] + \log p(y_n) + \alpha \log q_\psi(y_n \mid \mathbf{x}_n, \mathbf{b}_n)
\end{aligned}
```

**Unlabelled Objective**:
```math
\mathcal{U}_{\text{unlabelled}}(\mathbf{x}_n) = \sum_{k=1}^K q_\psi(y_n = k \mid \mathbf{x}_n, \mathbf{b}_n) \mathcal{L}_{\text{labelled}}(\mathbf{x}_n, k) + \mathcal{H}\left( q_\psi(y_n \mid \mathbf{x}_n, \mathbf{b}_n) \right)
```

---

## 3. Markov Affinity Graph Diffusion (MAGIC Operators)

### 3.1 Adaptive Distance Metric & Affinity Matrix

Let $\mathbf{Z} \in \mathbb{R}^{N \times d_{\text{pca}}}$ denote normalized coordinates. The distance between cells $i$ and $j$ is Euclidean: $d(\mathbf{z}_i, \mathbf{z}_j) = \|\mathbf{z}_i - \mathbf{z}_j\|_2$.

Adaptive Gaussian bandwidth $\sigma_i$ is computed as the distance to the $k$-th nearest neighbor ($k = 15$):

```math
\sigma_i = \text{k-NN}(\mathbf{z}_i, k)
```

The directed affinity kernel is:

```math
W_{ij} = \exp\left( -\left( \frac{d(\mathbf{z}_i, \mathbf{z}_j)}{\sigma_i} \right)^\alpha \right), \quad \alpha = 1
```

Affinities are symmetrized: $W^{\text{sym}}_{ij} = \frac{W_{ij} + W_{ji}}{2}$.

### 3.2 Markov Transition Operator & Low-Pass Filtering

The continuous diffusion operator $\mathbf{P} \in \mathbb{R}^{N \times N}$ is constructed via degree normalization:

```math
D_{ii} = \sum_{j=1}^N W^{\text{sym}}_{ij}, \quad \mathbf{P} = \mathbf{D}^{-1} \mathbf{W}^{\text{sym}}
```

Raising $\mathbf{P}$ to diffusion step $t = 3$:

```math
\mathbf{P}^t = \mathbf{\Phi} \mathbf{\Lambda}^t \mathbf{\Psi}^\top = \sum_{r=1}^N \lambda_r^t \mathbf{\phi}_r \mathbf{\psi}_r^\top
```

The imputed continuous expression profile $\hat{\mathbf{E}} \in \mathbb{R}^{N \times G}$ is:

```math
\hat{\mathbf{E}} = \mathbf{P}^t \mathbf{E}
```

### 3.3 Proof of Timepoint-Isolated Imputation Stability

Let the single-cell graph partition into Diagnosis ($D$) and Relapse ($R$) subsets: $V = V_D \cup V_R$.

By partitioning the affinity matrix such that $\mathbf{P}_{DR} = \mathbf{P}_{RD} = \mathbf{0}$:

```math
\mathbf{P}_{\text{isolated}} = \begin{bmatrix} \mathbf{P}_D & \mathbf{0} \\ \mathbf{0} & \mathbf{P}_R \end{bmatrix} \implies \mathbf{P}_{\text{isolated}}^t = \begin{bmatrix} \mathbf{P}_D^t & \mathbf{0} \\ \mathbf{0} & \mathbf{P}_R^t \end{bmatrix}
```

For all $i \in V_R$, $\hat{E}_{i, g} = \sum_{j \in V_R} (\mathbf{P}_R^t)_{ij} E_{j, g}$, proving that imputed *ERG* expression at relapse reflects purely relapse-intrinsic manifold geodesics without cross-stage information leakage.

---

## 4. Single-Cell Rank-Based Stemness Scoring (pyUCell)

### 4.1 Rank Truncation & Mann-Whitney U Derivation

For cell $i$, let raw count expression across all $G$ genes be $\mathbf{x}_i = (x_{i1}, \dots, x_{iG})$.
1. Genes are ranked in descending order:
```math
R_{ig} = \text{Rank}(x_{ig}) \in \{1, \dots, G\}
```

2. Ranks are truncated at threshold $r_{\text{max}} = 1,500$:
```math
r'_{ig} = \begin{cases} R_{ig}, & \text{if } R_{ig} \le r_{\text{max}} \\ r_{\text{max}} + 1, & \text{if } R_{ig} > r_{\text{max}} \end{cases}
```

3. For gene signature set $\mathcal{S}$ of size $K = |\mathcal{S}|$, the Mann-Whitney rank sum is:
```math
U_i(\mathcal{S}) = \sum_{g \in \mathcal{S}} r'_{ig} - \frac{K(K + 1)}{2}
```

4. The normalized UCell activity score is:
```math
\text{UCell}_i(\mathcal{S}) = 1 - \frac{U_i(\mathcal{S})}{K \cdot r_{\text{max}}}
```

### 4.2 Direction-Aware LSC17 Formulation

We define the direction-aware signed stemness score as:

```math
\text{LSC17}_{\text{signed}, i} = \text{UCell}_i(\mathcal{S}^+) - \text{UCell}_i(\mathcal{S}^-)
```

where:
- $\mathcal{S}^+ = \{\text{DNMT3B}, \text{NYNRIN}, \text{LAPTM4B}, \text{MMRN1}, \text{DPYSL3}, \text{FAM30A}, \text{SOCS2}, \text{EMP1}, \text{BEX3}, \text{CD34}, \text{ADGRG1}\}$
- $\mathcal{S}^- = \{\text{ZBTB46}, \text{ARHGAP22}, \text{CDK6}, \text{CPXM1}, \text{SMIM24}, \text{AKR1C3}\}$

---

## 5. Paired Non-Parametric Clonal Inference (Wilcoxon Signed-Rank)

Let $k \in \{1, \dots, n\}$ represent the $n = 20$ matched patient cohorts. Let $D_k$ and $R_k$ represent the percentage of $ERG$-positive malignant blasts ($\log_{1p}(\text{CP10k}) > 0$) at Diagnosis and Relapse, respectively. 

The paired difference is $d_k = R_k - D_k$.
1. Absolute differences $|d_k|$ are sorted in ascending order and assigned integer ranks:
```math
\text{Rank}(|d_k|) \in \{1, \dots, N_r\}
```

2. Test statistics $W^+$ and $W^-$ are computed:
```math
W^+ = \sum_{k: d_k > 0} \text{Rank}(|d_k|), \quad W^- = \sum_{k: d_k < 0} \text{Rank}(|d_k|)
```
The test statistic is $W = \min(W^+, W^-)$.

3. Under the null hypothesis $H_0: \mathbb{P}(R_k > D_k) = \mathbb{P}(R_k < D_k)$:
```math
\mathbb{E}[W] = \frac{N_r (N_r + 1)}{4} = \frac{20 \cdot 21}{4} = 105.0
```
```math
\sigma_W^2 = \frac{N_r (N_r + 1)(2 N_r + 1)}{24} - \sum_t \frac{t^3 - t}{48}
```

4. Results in our cohort:
- $15 / 20$ cohorts show positive expansion ($d_k > 0$, $75.0\%$);
- Exact two-sided permutation test yields:
```math
p = 0.002712 \quad (p < 0.01)
```
- Cohort mean shift: $\bar{D} = 11.12\% \to \bar{R} = 19.63\%$.

---

## 6. References

1. Lopez et al., *Single-cell dissection of stemness programs and transcriptional plasticity in pediatric acute myeloid leukemia relapse*. *Nature Cancer* (2026).
2. Lopez, R. et al. Deep generative modeling for single-cell transcriptomics. *Nat. Methods* 15, 1053–1058 (2018).
3. Xu, C. et al. Probabilistic harmonization and annotation of single-cell transcriptomics data with scANVI. *Mol. Syst. Biol.* 17, e10082 (2021).
4. van Dijk, D. et al. Recovering Gene Interactions from Single-Cell Data Using Data Diffusion. *Cell* 174, 716–729 (2018).
5. Andreatta, M. & Carmona, S. J. UCell: Robust and scalable single-cell gene signature scoring. *Comp. Struct. Biotechnol. J.* 19, 3796–3798 (2021).
6. Ng, S. W. et al. A 17-gene stemness score for rapid determination of risk in acute leukaemia. *Nature* 540, 433–437 (2016).
7. Lambo, R. et al. The genomic and transcriptomic landscape of pediatric AML relapse. *Cancer Cell* 41, 1989–2005 (2023).
8. Furer, N. et al. A reference model of circulating hematopoietic stem cells across the lifespan with applications to diagnostics. *Nat. Med.* 31, 2442–2451 (2025).
9. Suo, C. et al. Mapping the developing human immune system across development. *Science* 376, eabo0510 (2022).
