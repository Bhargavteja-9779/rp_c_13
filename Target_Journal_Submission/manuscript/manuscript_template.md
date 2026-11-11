---
title: "Prevalence-adaptive F-measure thresholding for imbalanced tabular classification: a guarded decision layer and its limits under controlled and real distribution shift"
---

**Abstract**

Classifiers for imbalanced problems are deployed with a decision threshold tuned for the F-measure on validation data. That threshold is optimal only for the validation prevalence, yet prevalence drifts after deployment. We characterise the F~β~-optimal threshold under prior-probability (label) shift and propose PAFT, a model-agnostic decision layer that infers the deployment prevalence from unlabelled scores with a mixture likelihood and a weak prior, selects the threshold maximising the posterior-expected metric on a validation-derived surface, and applies a goodness-of-fit guard that reverts to the static threshold when the label-shift assumption is rejected. We compare it with seven baselines on ⟦n_ds⟧ public tabular datasets (five seeds, four base learners, ⟦n_batches_main⟧ simulated deployment batches) and on five real temporal or cross-site shifts. Under controlled label shift PAFT improves mean F1 over the static threshold on all ⟦n_ds⟧ datasets (⟦gain_static⟧, 95% CI ⟦gain_static_ci⟧; Holm-adjusted *p* ⟦p_static⟧) and is statistically indistinguishable from the strongest existing rule, expected-F maximisation on EM-corrected posteriors (⟦gain_dta⟧; *p* = ⟦p_dta⟧), while also covering metrics which that rule does not. On the real shifts, label-free adaptation by any method gives no reliable gain and significantly lowers F1 on two suites because prevalence estimation breaks; the guard detects this and restores static performance. Prevalence-adaptive thresholding is therefore effective when label shift holds and should be guarded otherwise. A small labelled audit sample used only to estimate prevalence recovers part of the gain where prevalence, rather than concept, has changed.

**Keywords** Class imbalance · Decision threshold · F-measure · Label shift · Quantification · Dataset shift

# 1 Introduction

Rare-event classification is the normal case in applied machine learning: fraud, churn, intrusion, readmission and defect prediction all have a minority class of interest [⟦c:he2009⟧, ⟦c:krawczyk2016⟧, ⟦c:johnson2019⟧]. Because accuracy is uninformative under imbalance, such systems are evaluated with precision–recall metrics, most often the F-measure, and the score produced by the learner is converted into a decision by a threshold tuned on held-out data [⟦c:lipton2014⟧, ⟦c:zou2016⟧, ⟦c:esposito2021⟧, ⟦c:leevy2023⟧]. A recent 45-task study confirms that threshold tuning is one of the most consistently useful interventions for imbalanced tabular problems [⟦c:musaev2026⟧].

A tuned threshold has a weakness that is easy to overlook. Unlike cost-based decisions, for which a fixed posterior threshold stays optimal, the F-measure is not decomposable over instances, and its optimal threshold depends on the class prevalence itself [⟦c:lipton2014⟧, ⟦c:ye2012⟧]. Prevalence, however, is exactly what changes after deployment: fraud rates move with campaigns, positive rates in marketing follow the economy, and disease rates differ between sites [⟦c:moreno2012⟧, ⟦c:dalpozzolo2018⟧, ⟦c:godau2025⟧]. A threshold that was F-optimal on validation data is therefore systematically wrong on the deployment stream, even if the classifier itself is unchanged.

The literature offers the ingredients of a remedy but not a tested recipe. Prior-shift adaptation re-weights posteriors with an estimated deployment prior [⟦c:saerens2002⟧, ⟦c:lipton2018⟧, ⟦c:alexandari2020⟧, ⟦c:sipka2022⟧], and quantification estimates that prior from unlabelled data [⟦c:forman2008⟧, ⟦c:gonzalez2017⟧, ⟦c:esuli2023⟧]. Yet correcting the posterior is not the same as correcting an F-optimal decision, and the closest deployment-oriented workflow sidesteps the F-measure by recommending expected cost instead [⟦c:godau2025⟧]. Three questions remain open for practitioners who must report F-type metrics: how large the loss of a static threshold is, which label-free rule recovers it, and whether any of this survives real shifts, which are rarely pure label shifts.

This paper answers these questions with a formal characterisation, a method, and a deliberately two-sided evaluation. Our contributions are the following.

1. **Characterisation.** We express any confusion-matrix metric at deployment prevalence π as a closed-form function of π and the validation ROC curve, generalise the F1 optimality condition of Lipton et al. [⟦c:lipton2014⟧] to F~β~ under a shifted prior, and show that the common practice of applying a fixed, validation-tuned threshold to prior-corrected posteriors is biased in a predictable direction (Section 3).
2. **Method.** We propose PAFT (prevalence-adaptive F-measure thresholding), a decision layer that needs only validation scores: a mixture-likelihood posterior over the deployment prevalence, a Bayes threshold that maximises the posterior-expected metric, and a goodness-of-fit guard that falls back to the static threshold when the label-shift model does not fit the batch (Section 4). It is inductive (it outputs a threshold on the score), calibration-free, and applies to F~β~ and the Matthews correlation coefficient alike.
3. **Evidence under controlled shift.** On ⟦n_ds⟧ datasets, four base learners and ⟦n_batches_main⟧ batches, we quantify the loss of static thresholds and compare eight label-free rules with dataset-level paired tests. Adaptive rules recover most of the oracle gap; PAFT matches, but does not exceed, expected-F maximisation on EM-corrected posteriors (Section 6).
4. **Evidence under real shift, including negative results.** On five temporal or cross-site suites, no label-free rule helps reliably and several hurt. We trace the failure to prevalence estimation rather than to the threshold map, show that the guard prevents the harm, and evaluate a label-assisted variant that spends a small audit sample on prevalence only (Sections 6.6–6.7).

We state plainly what the paper does not claim: PAFT is not more accurate than the best existing transductive rule under label shift, and unguarded prevalence adaptation is not safe on real tabular drift. The value of the work lies in the characterisation, the guard, and an evaluation that separates the regime where prevalence adaptation works from the regime where it does not.

# 2 Related work

**Threshold selection for imbalanced data.** Moving the decision threshold is a long-standing alternative to resampling [⟦c:chawla2002⟧, ⟦c:he2009⟧]. Lipton et al. [⟦c:lipton2014⟧] showed that the F1-optimal threshold on calibrated probabilities equals half the optimal F1, so it varies with prevalence and classifier quality; Ye et al. [⟦c:ye2012⟧] contrasted threshold tuning with decision-theoretic maximisation of expected F on a test set, and Jansche [⟦c:jansche2005⟧] optimised an expected F-measure during training. Zou et al. [⟦c:zou2016⟧], GHOST [⟦c:esposito2021⟧] and Leevy et al. [⟦c:leevy2023⟧] tune thresholds on development data, and Dal Pozzolo et al. [⟦c:dalpozzolo2015⟧] correct thresholds for the prior change introduced by undersampling. All of these assume that deployment prevalence equals the development prevalence, or that the change is known.

**Prior-probability (label) shift.** When P(y) changes and P(x | y) does not [⟦c:storkey2008⟧, ⟦c:moreno2012⟧], posteriors can be corrected by re-weighting with the new prior, which Saerens et al. [⟦c:saerens2002⟧] estimate by expectation–maximisation (EM). Black-box shift estimation [⟦c:lipton2018⟧], regularised variants [⟦c:azizzadenesheli2019⟧], the maximum-likelihood view with bias-corrected calibration [⟦c:alexandari2020⟧, ⟦c:garg2020⟧] and practical guides [⟦c:sipka2022⟧] refine the estimate, and Tasche [⟦c:tasche2017⟧] and Vaz et al. [⟦c:vaz2019⟧] study its consistency. These works evaluate prior estimation error, calibration and accuracy; the induced Bayes rule is optimal for decomposable losses, not for F-type metrics. Du Plessis and Sugiyama [⟦c:duplessis2014⟧] estimate the class balance by distribution matching with the same limitation.

**Quantification.** Estimating prevalence in unlabelled samples is a task in its own right [⟦c:forman2008⟧, ⟦c:gonzalez2017⟧, ⟦c:esuli2023⟧]. Adjusted counts [⟦c:forman2008⟧], probabilistic variants [⟦c:bella2010⟧], Hellinger-distance mixture matching [⟦c:castro2013⟧] and the DyS family [⟦c:maletzke2019⟧] are available in QuaPy [⟦c:moreo2021⟧]. Tasche [⟦c:tasche2014⟧] analyses exact fit of two-component mixtures. The quantification literature optimises prevalence error and usually stops before the decision.

**ROC analysis under changing class distributions.** Provost and Fawcett [⟦c:provost2001⟧] used the ROC convex hull to choose operating points when class distributions and costs are imprecise, and Fawcett [⟦c:fawcett2006⟧] and Flach and Kull [⟦c:flach2015⟧] discuss how precision-type metrics depend on prevalence. The identity we use, that a confusion-matrix metric is a function of prevalence and a ROC point, is therefore classical; what has been missing is its use as an operational, label-free deployment rule with uncertainty handling and an empirical test.

**Deployment under prevalence shift.** Godau et al. [⟦c:godau2025⟧] study 30 medical-imaging tasks, show that prevalence shift degrades calibration and tuned decision rules, and propose prevalence-aware re-calibration followed by the Bayes rule for expected cost; prevalences are assumed known or externally estimated, imbalance ratios do not exceed 10, and shifts are simulated. Alsulaimawi [⟦c:alsulaimawi2026⟧] adapts a Bayes threshold online from likelihood ratios for cost-sensitive decisions. Neither gives an F-optimal rule or tests it on real drift. Concept-drift research [⟦c:gama2014⟧, ⟦c:lu2019⟧, ⟦c:gama2004⟧] and shift detection [⟦c:rabanser2019⟧] address changes in P(y | x), usually with labels or by retraining.

**Gap.** Table ⟦T:gap⟧ summarises the position of this work. No prior study, to our knowledge, combines an F-optimal decision rule under estimated prevalence, an explicit validity check of the label-shift assumption, strong label-free baselines including decision-theoretic F maximisation, extreme imbalance on tabular data, and real non-simulated shifts. We make no claim that the ingredients are new.

⟦TABLE:gap⟧

# 3 Problem formulation and analysis

## 3.1 Setting

Let $s:\mathcal{X}\to\mathbb{R}$ be a fixed scoring function produced by any learner, with larger values indicating the positive class $y=1$. Development data follow a distribution $P$ with prevalence $\pi_0=P(y=1)$; a deployment batch $\{x_i\}_{i=1}^{m}$ is drawn from $Q$ with unknown prevalence $\pi=Q(y=1)$ and unobserved labels. A decision rule predicts $\hat y_i=\mathbb{1}[s(x_i)\ge t]$. The label-shift assumption states that the class-conditional distributions of the score are unchanged,

$$Q(s\mid y)=P(s\mid y),\qquad y\in\{0,1\}. \qquad (1)$$

Assumption (1) is weaker than invariance of $P(x\mid y)$, because it concerns only the one-dimensional score. Define the validation ROC functions $G_1(t)=P(s\ge t\mid y=1)$ (true-positive rate) and $G_0(t)=P(s\ge t\mid y=0)$ (false-positive rate).

## 3.2 Metrics as functions of prevalence and a ROC point

Under (1) the expected confusion matrix of threshold $t$ on the deployment distribution has entries $\pi G_1$, $(1-\pi)G_0$, $\pi(1-G_1)$ and $(1-\pi)(1-G_0)$. Any metric $M$ that is a function of the normalised confusion matrix is therefore a function $M(t;\pi)$ of the threshold and the prevalence alone. For the F-measure with weight $\beta$,

$$F_\beta(t;\pi)=\frac{(1+\beta^2)\,\pi\,G_1(t)}{\beta^2\pi+\pi\,G_1(t)+(1-\pi)\,G_0(t)}. \qquad (2)$$

The static, validation-tuned threshold is $t_0=\arg\max_t F_\beta(t;\pi_0)$, and the best threshold for the deployment distribution is $t^\star(\pi)=\arg\max_t F_\beta(t;\pi)$. The *regret* of the static rule is $F_\beta(t^\star(\pi);\pi)-F_\beta(t_0;\pi)\ge 0$. Figure ⟦F:mechanism⟧ illustrates (2) for one of the benchmark datasets.

## 3.3 Optimality condition under a shifted prior

**Proposition 1.** *Assume $G_1$ and $G_0$ are differentiable with densities $g_1,g_0$, and let $t^\star$ be an interior maximiser of (2) with value $F^\star_\beta(\pi)$. Then the deployment posterior at the threshold satisfies*

$$\eta_\pi(t^\star)\;=\;\frac{\pi g_1(t^\star)}{\pi g_1(t^\star)+(1-\pi)g_0(t^\star)}\;=\;\frac{F^\star_\beta(\pi)}{1+\beta^2}, \qquad (3)$$

*equivalently, in terms of the prevalence-free likelihood ratio $\lambda(t)=g_1(t)/g_0(t)$,*

$$\lambda(t^\star)=\frac{1-\pi}{\pi}\cdot\frac{F^\star_\beta(\pi)}{1+\beta^2-F^\star_\beta(\pi)}. \qquad (4)$$

*Proof.* Write $F_\beta=N/D$ with $N=(1+\beta^2)\pi G_1$ and $D=\beta^2\pi+\pi G_1+(1-\pi)G_0$. Since $G_y'=-g_y$, the first-order condition $N'D=ND'$ gives $(1+\beta^2)\pi g_1 D=N\,(\pi g_1+(1-\pi)g_0)$. Dividing by $D\,(\pi g_1+(1-\pi)g_0)$ yields (3); solving (3) for $g_1/g_0$ yields (4). ∎

For $\beta=1$ and $\pi=\pi_0$, (3) is the result of Lipton et al. [⟦c:lipton2014⟧]. The shifted form has two practical consequences.

**Corollary 1 (fixed posterior thresholds are biased).** For any fixed $t$ with $G_1(t)>0$, dividing numerator and denominator of (2) by $\pi$ shows that $F_\beta(t;\pi)$ is non-decreasing in $\pi$; hence so is $F^\star_\beta(\pi)$. By (3), the optimal threshold on the *prior-corrected* posterior rises with prevalence. A rule that corrects the posterior for the new prior but keeps the validation-tuned posterior threshold $F^\star_\beta(\pi_0)/(1+\beta^2)$ is therefore too permissive when prevalence has fallen and too strict when it has risen.

**Corollary 2 (direction on the raw score).** In (4) the factor $(1-\pi)/\pi$ decreases in $\pi$ while the second factor increases. When the first effect dominates, which is the usual case for classifiers far from perfect separation, the optimal threshold on the *uncorrected* score falls as prevalence rises: a larger positive pool raises precision at every threshold, so recall can be bought more cheaply. Section 6.9 reports how often this direction is observed.

**Remark (sensitivity to the prevalence estimate).** If $F_\beta(\cdot;\pi)$ is twice differentiable with a unique interior maximiser, then a plug-in threshold $t^\star(\hat\pi)$ has regret of second order in $\hat\pi-\pi$, by the usual argument for perturbed maximisers. Small quantification errors are thus cheap, but the regret grows quickly when $\hat\pi$ is wrong by a large factor, which is what happens for rare classes in small batches and, as Section 6.6 shows, when (1) is violated. This motivates treating $\pi$ as uncertain rather than plugging in a point estimate.

# 4 Proposed method: PAFT

PAFT has a development stage, executed once on the labelled validation split, and a deployment stage, executed on every unlabelled batch (Figure ⟦F:workflow⟧). It never modifies or retrains the scorer.

## 4.1 Development stage

From the validation scores we compute (i) the empirical ROC functions $\hat G_1,\hat G_0$ on a set of $T$ candidate thresholds, consisting of the validation scores of both classes, each class thinned to at most 1,024 quantiles so that the minority-class tail is resolved; (ii) class-conditional histograms $\hat h_1,\hat h_0$ of the score on $B$ bins whose edges are class-balanced quantiles, with $B=\min(32,\max(8,\lfloor n_1/10\rfloor))$ for $n_1$ validation positives and Laplace smoothing of 0.5 per bin; and (iii) the metric surface $M(t_j;\pi_g)$ on the $T$ thresholds and a grid of $G=161$ prevalences that is uniform on the logit scale and centred on $\pi_0$ (±9 logits). The static threshold $t_0$ is the maximiser of the row $\pi_g=\pi_0$, which coincides with the usual empirical F-maximising threshold on the validation split.

## 4.2 Posterior over the deployment prevalence

Let $n_b$ be the number of batch scores in bin $b$. Under (1) the scores follow a two-component mixture with known components and unknown weight, with log-likelihood

$$\ell(\pi)=\sum_{b=1}^{B} n_b\,\log\!\big(\pi\,\hat h_{1b}+(1-\pi)\,\hat h_{0b}\big). \qquad (5)$$

Maximising (5) is the EM estimator of Saerens et al. [⟦c:saerens2002⟧] applied to histogram-calibrated scores, and is a likelihood counterpart of Hellinger-distance mixture matching [⟦c:castro2013⟧, ⟦c:maletzke2019⟧]; we do not claim it as a new quantifier. Because (5) is concave in the scalar $\pi$, we evaluate it on the grid and combine it with a weakly informative prior on the log-odds, $\operatorname{logit}\pi\sim\mathcal{N}(\operatorname{logit}\pi_0,\sigma^2)$:

$$w_g\;\propto\;\exp\!\Big\{\ell(\pi_g)-\tfrac{1}{2\sigma^2}\big(\operatorname{logit}\pi_g-\operatorname{logit}\pi_0\big)^2\Big\},\qquad \sum_g w_g=1. \qquad (6)$$

We use $\sigma=2$ throughout, which places 95% prior mass on odds changes of up to a factor of about 50 in either direction; this value was fixed before any deployment-batch result was inspected, and Section 6.3 reports its sensitivity.

## 4.3 Bayes threshold

Instead of plugging a point estimate into $t^\star(\cdot)$, PAFT chooses the threshold with the highest posterior-expected metric,

$$\hat t=\arg\max_{j}\;\sum_{g} w_g\,M(t_j;\pi_g). \qquad (7)$$

We call the plug-in rule $t^\star(\hat\pi_{\mathrm{MLE}})$ *PAFT-Plug* and rule (7) *PAFT-Bayes*. Rule (7) hedges against quantification error: when the batch is small or the classes are poorly separated, the posterior is wide and the chosen threshold stays close to thresholds that perform acceptably across plausible prevalences.

## 4.4 Validity guard

Adaptation is justified only if (1) holds. We test it on each batch by comparing the empirical distribution function $\hat F_m$ of the batch scores with the fitted mixture $\hat\pi\hat F_1+(1-\hat\pi)\hat F_0$, where $\hat F_y$ are the validation class-conditional distribution functions and $\hat\pi$ maximises (5):

$$D=\sup_s\big|\hat F_m(s)-\hat\pi\hat F_1(s)-(1-\hat\pi)\hat F_0(s)\big|. \qquad (8)$$

The mixture is itself estimated from $n_1$ and $n_0$ validation scores, so we refer $D$ to the two-sample Kolmogorov distribution with effective size $m\,n_\mathrm{eff}/(m+n_\mathrm{eff})$, $n_\mathrm{eff}=\big(\hat\pi^2/n_1+(1-\hat\pi)^2/n_0\big)^{-1}$, using the finite-sample correction of Stephens [⟦c:stephens1970⟧]. *PAFT-Guard* returns $\hat t$ if the *p*-value is at least $\alpha_g=0.01$ and the static threshold $t_0$ otherwise. The reference distribution is approximate (the weight is fitted and bootstrap resampling of small positive pools is ignored), so $\alpha_g$ is a tuning level rather than an exact size; Section 6.4 reports the observed firing rates.

## 4.5 Label-assisted variant

The real-shift results of Section 6.6 led us to add one variant after the main experiments had been analysed; it is reported as such. If $n_L$ instances of the batch can be labelled, of which $k$ are positive, their count is binomial in $\pi$ regardless of any change in $P(x\mid y)$. *PAFT-L* replaces (5) by $k\log\pi+(n_L-k)\log(1-\pi)$ in (6); the guarded version adds (5) for the unlabelled remainder only when the guard passes. Labels are thus spent on a single scalar, while the threshold map still comes from validation data.

## 4.6 Complexity

The development stage is dominated by sorting the validation scores. A deployment batch costs $O(m\log B)$ for binning, $O(GB)$ for (5)–(6), $O(GT)$ for (7) and $O(m\log m)$ for the guard; the stored state is the $G\times T$ surface. Measured costs are given in Section 6.8.

# 5 Experimental setup

## 5.1 Research questions

RQ1: How much F1 does a static threshold lose under prevalence shift, and how much do label-free rules recover? (Experiments 1–2.) RQ2: Which components of PAFT matter? (Experiment 3.) RQ3: How robust are the rules to small batches, small validation sets and violations of label shift? (Experiment 4.) RQ4: Do the conclusions transfer to other metrics, other base learners and real shifts? (Experiment 5.) RQ5: What does adaptation cost computationally, and where does it fail? (Experiments 6–8.) All experiments, the split, the seeds, the shift grid and the PAFT hyper-parameters were fixed before deployment-batch results were examined; the label-assisted variant (Experiment 9) is the single post-hoc addition.

## 5.2 Controlled-shift benchmark

**Datasets.** We use every dataset we identified on OpenML [⟦c:vanschoren2014⟧] that met criteria fixed in advance: a binary target (or a standard binarisation), at least 10,000 instances, at least 400 minority instances and a minority prevalence of at most 30%. This gives ⟦n_ds⟧ datasets from fraud, churn, marketing, credit, industrial failure, physics and census domains (Table ⟦T:datasets⟧), with prevalences from ⟦prev_min⟧% to ⟦prev_max⟧%; the minority class is always the positive class. Among them are the credit-card fraud data of Dal Pozzolo et al. [⟦c:dalpozzolo2015⟧], bank marketing [⟦c:moro2014⟧] and 30-day readmission [⟦c:strack2014⟧].

⟦TABLE:datasets⟧

**Splits and leakage control.** Each dataset is split into training (50%), validation (20%) and test pool (30%) by stratified group ten-fold assignment, repeated for seeds 0–4. Rows with identical feature vectors form one group, so exact duplicates (up to ⟦dup_max⟧ rows in one dataset, Table ⟦T:datasets⟧) cannot straddle splits; the audit confirms zero cross-split duplicates in all ⟦n_fits⟧ fits. Encoders, imputation and scaling statistics are fitted on the training split only. Early stopping uses a 10% slice of the training split, so the validation split is used only by the decision layer and is the same for all rules.

**Base learners.** LightGBM [⟦c:ke2017⟧] is the main learner; XGBoost [⟦c:chen2016⟧], logistic regression and a two-hidden-layer perceptron (256 and 128 units) test model-agnosticism, in line with evidence that boosted trees remain strong on tabular data [⟦c:grinsztajn2022⟧, ⟦c:shwartz2022⟧]. Hyper-parameters are fixed across datasets (Supplementary Section S1); no rule benefits from tuning that another is denied, since all rules consume the same scores.

**Shift protocol.** For each fit we draw deployment batches from the test pool by the artificial-prevalence protocol used in quantification research [⟦c:forman2008⟧, ⟦c:esuli2023⟧]: the odds of the positive class are multiplied by ρ ∈ {1/8, 1/4, 1/2, 1, 2, 4, 8}, the number of positives is binomial in the batch size *m* ∈ {500, 2,000, 10,000} (conditioned on at least one instance of each class), and instances are drawn with replacement within each class. With 30 repetitions per cell this gives ⟦n_batches_fit⟧ batches per fit and ⟦n_batches_main⟧ batches in the main experiment. Label shift holds by construction.

## 5.3 Real-shift suites

Five suites use naturally ordered or naturally partitioned data; nothing from a deployment batch is used for fitting (Table ⟦T:realinfo⟧). *Bank-time*: the chronologically ordered bank telemarketing data [⟦c:moro2014⟧] without the leaking call-duration feature; the first half is development data, followed by batches of 2,000 calls. *Creditcard-time*: the fraud transactions in time order, first half for development, then batches of 10,000. *Electricity-time*: the Elec2 price-direction stream [⟦c:gama2004⟧], first 30% for development, then three-week batches. *Elliptic-time*: labelled Bitcoin transactions [⟦c:weber2019⟧] with the customary split at time step 34 and one batch per later step. *ACS-income-states*: the ACSIncome task [⟦c:ding2021⟧] on the 2018 one-year survey, developed on California and deployed on each of the other 49 states and Puerto Rico. Development data are split 5:2 into training and validation as above.

⟦TABLE:realinfo⟧

## 5.4 Compared rules

All rules receive the same validation scores and the same unlabelled batch.
*Default 0.5*: threshold 0.5 on the isotonic-calibrated probability [⟦c:zadrozny2002⟧, ⟦c:niculescu2005⟧].
*Static-F*: the validation-tuned threshold $t_0$, the standard practice.
*Static-Bag*: the median of $t_0$ over 50 stratified bootstrap resamples of the validation split, a variance-reduced static rule in the spirit of GHOST [⟦c:esposito2021⟧] (it differs from GHOST, which optimises Cohen's kappa on training subsets).
*EM-0.5*: EM prior correction [⟦c:saerens2002⟧] of the calibrated posterior followed by the 0.5 rule, i.e. the Bayes rule for accuracy and the decision rule of prevalence-aware workflows [⟦c:godau2025⟧].
*EM-Fixed*: the EM-corrected posterior with the static threshold mapped to posterior space.
*PrevMatch*: predict as positive the top $\hat\pi m$ scores, with $\hat\pi$ from adjusted classify-and-count [⟦c:forman2008⟧].
*DTA-EM*: the decision-theoretic approach [⟦c:ye2012⟧] applied to EM-corrected posteriors $q_i$: predict the top $k$ instances, with $k$ maximising the plug-in expected F~β~, $(1+\beta^2)\sum_{i\le k}q_{(i)}/(\beta^2\sum_i q_i+k)$. This is the strongest label-free competitor we are aware of; it is transductive (it ranks the whole batch) and specific to F~β~.
*PAFT-Plug, PAFT-Bayes, PAFT-Guard*: Section 4.
Two oracles bound what is achievable: *Oracle-π* applies $t^\star(\pi)$ with the true batch prevalence, and *Oracle-threshold* applies the best threshold for the labelled batch. Baselines use isotonic calibration; a Platt-type alternative is examined in Section 6.4 and changes no conclusion.

## 5.5 Metrics and statistical analysis

The primary metric is F1 on each batch; F~0.5~, F~2~ and the Matthews correlation coefficient (MCC) [⟦c:chicco2020⟧] test generality, and regret is measured against Oracle-threshold. Batch values are averaged over repetitions, batch sizes, shift levels and seeds to one value per dataset, so the unit of analysis is the dataset (n = ⟦n_ds⟧) in the controlled benchmark and the deployment batch (averaged over seeds) in each real suite. We use two-sided Wilcoxon signed-rank tests [⟦c:wilcoxon1945⟧] with Holm correction within each family of comparisons, as recommended for comparisons over multiple datasets [⟦c:demsar2006⟧], report the mean paired difference with a 10,000-resample percentile bootstrap 95% confidence interval [⟦c:efron1992⟧] and the matched-pairs rank-biserial correlation as effect size [⟦c:kerby2014⟧], and add a Friedman test [⟦c:friedman1937⟧] with average ranks and the Nemenyi critical difference [⟦c:demsar2006⟧]. No seed or configuration was selected after seeing results; the standard deviation over seeds is reported in Supplementary Table S3.

# 6 Results

## 6.1 Static thresholds lose F1 under prevalence shift, and adaptive rules recover most of it (RQ1)

Table ⟦T:main⟧ and Figure ⟦F:main⟧ give the main result for LightGBM. Averaged over the six shifted levels, Static-F reaches a mean F1 of ⟦f_static⟧, against ⟦f_oraclepi⟧ for Oracle-π and ⟦f_oraclethr⟧ for Oracle-threshold. The loss grows with the size of the shift in either direction and is largest when prevalence rises eightfold in odds (⟦f_static_8⟧ against ⟦f_oraclepi_8⟧).

PAFT-Bayes attains ⟦f_paft⟧, closing ⟦gap_closed⟧% of the gap between Static-F and Oracle-π. It improves on Static-F on all ⟦n_ds⟧ datasets, with a mean difference of ⟦gain_static⟧ (95% CI ⟦gain_static_ci⟧; Holm-adjusted *p* ⟦p_static⟧; rank-biserial ⟦rb_static⟧; Table ⟦T:stats⟧). The gain is stable across seeds (⟦seed_gain⟧). It also exceeds EM-Fixed by ⟦gain_emfixed⟧ and PrevMatch by ⟦gain_prevmatch⟧ on every dataset. As Corollary 1 predicts, EM-Fixed is close to PAFT when prevalence rises but falls to or below the level of Static-F when prevalence falls (Table ⟦T:main⟧, ρ < 1), where its fixed posterior threshold is too permissive.

**PAFT does not outperform DTA-EM.** The decision-theoretic rule on EM-corrected posteriors reaches ⟦f_dta⟧. PAFT-Bayes is better on ⟦wins_dta⟧ of ⟦n_ds⟧ datasets (median difference ⟦med_dta⟧) and has the best average rank (⟦rank_paft⟧ against ⟦rank_dta⟧; Figure ⟦F:ranks⟧), but its mean is ⟦gain_dta_abs⟧ lower because of two datasets with very few validation positives (creditcard and APSFailure, Section 6.9); the difference is not significant (*p* = ⟦p_dta⟧), and the difference in average rank is below the Nemenyi critical difference of ⟦cd⟧. The two rules should be regarded as equivalent in F1 under label shift. A Friedman test rejects equality of the ten rules (χ² = ⟦friedman_chi⟧, *p* ⟦friedman_p⟧).

**Cost when nothing has shifted.** At ρ = 1 every adaptive rule pays a small price for estimating a prevalence that has not changed: PAFT-Bayes is ⟦noshift_cost⟧ below Static-F (95% CI ⟦noshift_ci⟧; Holm-adjusted *p* = ⟦noshift_p⟧), and DTA-EM is at the same level (⟦f_dta_1⟧ against ⟦f_paft_1⟧).

⟦TABLE:main⟧

⟦TABLE:stats⟧

## 6.2 Other base learners and metrics (RQ4)

The pattern is the same for all four learners (Supplementary Table S1): the gain of PAFT-Bayes over Static-F is ⟦gain_xgb⟧ for XGBoost, ⟦gain_logreg⟧ for logistic regression and ⟦gain_mlp⟧ for the perceptron (all Holm-adjusted *p* < 0.001), and in each case PAFT-Bayes wins on most datasets against DTA-EM while having a slightly lower mean. Table ⟦T:metrics⟧ reports other metrics at *m* = 2,000. PAFT-Bayes improves on Static-F for F~0.5~ (⟦gain_f05⟧), F~2~ (⟦gain_f2⟧) and MCC (⟦gain_mcc⟧; all Holm-adjusted *p* < 0.01). For MCC the decision-theoretic formula does not apply, so PAFT is the only adaptive rule in the comparison that addresses it directly. For F~0.5~ DTA-EM has a higher mean than PAFT-Bayes (⟦dta_f05⟧ against ⟦paft_f05⟧), not significantly (*p* = ⟦p_f05_dta⟧).

⟦TABLE:metrics⟧

## 6.3 Ablation (RQ2)

Table ⟦T:ablation⟧ removes or replaces each component. (i) *Posterior versus point estimate.* The Bayes threshold improves on the plug-in rule with the same likelihood on all ⟦n_ds⟧ datasets (⟦abl_plug⟧, Holm-adjusted *p* < 0.001); the benefit is concentrated in small batches (⟦abl_plug_100⟧ at *m* = 100 against ⟦abl_plug_10000⟧ at *m* = 10,000), where a point estimate of a rare prevalence is unreliable. With *m* = 100 the plug-in rule is slightly worse than Static-F. (ii) *Prior width.* σ between 0.5 and 2 gives nearly identical results; wider or flat priors lose up to ⟦abl_flat⟧ (Figure ⟦F:ablation⟧). A tighter prior (σ = 1) is marginally better than our pre-specified σ = 2 (⟦abl_sd1⟧, Holm-adjusted *p* = ⟦abl_sd1_p⟧); we keep σ = 2 as specified. (iii) *Quantifier.* With the plug-in rule, the choice among mixture maximum likelihood, adjusted counts, probabilistic adjusted counts and EM matters little, and all are clearly better than Hellinger-distance matching in small batches; handling uncertainty matters more than the particular estimator. (iv) *Decision rule.* Applying the decision-theoretic top-*k* rule to PAFT's own histogram posterior gives ⟦abl_dtahist⟧ relative to (7) (*p* = ⟦abl_dtahist_p⟧), so the equivalence with DTA-EM is not an artefact of the calibrator. (v) *Guard.* Under pure label shift the guard should be inactive; it costs ⟦abl_guard⟧ on average because it fires on some batches (Section 6.4).

⟦TABLE:ablation⟧

## 6.4 Robustness (RQ3)

**Batch size.** Adaptive rules keep their advantage down to *m* = 100 (Table ⟦T:ablation⟧; Supplementary Table S2); the advantage of PAFT-Bayes over Static-F is ⟦gain_m100⟧ at *m* = 100 and ⟦gain_m10000⟧ at *m* = 10,000.

**Small validation sets.** With 25% and 10% of the validation split (Table ⟦T:robust⟧), PAFT-Bayes still gains ⟦val25_gain⟧ and ⟦val10_gain⟧ over Static-F, but at 10% DTA-EM is significantly better than PAFT-Bayes (⟦val10_dta⟧, Holm-adjusted *p* = ⟦val10_dta_p⟧): the empirical ROC surface degrades faster than isotonic posteriors when validation positives are scarce.

**Calibrator of the baselines.** Replacing isotonic by Platt-type calibration leaves DTA-EM at ⟦platt_dta⟧ against ⟦platt_paft⟧ for PAFT-Bayes (*p* = ⟦platt_p⟧).

**Violated label shift.** Adding Gaussian noise of 0.1, 0.25 and 0.5 feature standard deviations to the deployment features changes the class-conditional score distributions (Figure ⟦F:robust⟧). Unguarded adaptive rules then lose their advantage: at the lowest noise level PAFT-Bayes (⟦noise01_paft⟧) and DTA-EM (⟦noise01_dta⟧) are no better than Static-F (⟦noise01_static⟧). PAFT-Guard, whose test fires on ⟦noise01_fire⟧% of these batches, obtains the highest mean F1 of all label-free rules at every noise level (⟦noise01_guard⟧, ⟦noise025_guard⟧, ⟦noise05_guard⟧), but its differences from PAFT-Bayes and Static-F are not significant at the dataset level (Supplementary Table S6), so we report this as a consistent tendency rather than an established effect.

**Guard firing under pure label shift.** With α~g~ = 0.01 the guard fires on ⟦fire_500⟧%, ⟦fire_2000⟧% and ⟦fire_10000⟧% of batches of size 500, 2,000 and 10,000, above the nominal level. The excess is concentrated in two datasets (Supplementary Table S5): creditcard, where it fires on every batch, and APSFailure (⟦fire_aps⟧%). Both have few validation and test positives (about ⟦cc_valpos⟧ and ⟦cc_testpos⟧ for creditcard), so that batches resampled from the small test pool differ detectably from the validation sample; on the other 16 datasets the median firing rate is ⟦fire_median16⟧%. The guard is therefore conservative for very rare classes, where it reduces to the static rule. Table ⟦T:guard⟧ shows that results are insensitive to α~g~ between 0.001 and 0.05.

⟦TABLE:robust⟧

## 6.5 Summary under controlled shift

Under the conditions for which they are designed, prevalence-adaptive rules are clearly better than a static threshold, at a negligible cost when nothing has shifted. PAFT-Bayes and DTA-EM are equivalent in F1; PAFT is preferable when a threshold on the score is needed, when calibrated posteriors are not available, or for metrics other than F~β~, and DTA-EM is preferable when validation positives are very scarce.

## 6.6 Real shifts: label-free adaptation does not help (RQ4)

Table ⟦T:real⟧ and Figure ⟦F:real⟧ report the five real suites. The picture is different from the controlled benchmark.

*ACS-income-states.* State prevalences range from ⟦acs_pimin⟧ to ⟦acs_pimax⟧ against ⟦acs_pi0⟧ in California. Unguarded rules are worse than the static threshold: PAFT-Bayes loses ⟦acs_loss⟧ (95% CI ⟦acs_loss_ci⟧; Holm-adjusted *p* < 0.001) and DTA-EM slightly more. Oracle-π would gain ⟦acs_oraclegain⟧, so the validation threshold map transfers; what fails is quantification, with a mean absolute prevalence error of ⟦acs_mae⟧ (Table ⟦T:realinfo⟧), because the feature distribution differs between states.

*Electricity-time.* PAFT-Bayes loses ⟦elec_loss⟧ (Holm-adjusted *p* < 0.001) and all EM-based rules behave alike. Here even Oracle-π gives no gain over Static-F (⟦elec_oraclepi⟧ against ⟦elec_static⟧), while Oracle-threshold reaches ⟦elec_oraclethr⟧: the relation between score and label has changed, and no prevalence correction can repair it.

*Creditcard-time and Elliptic-time.* Differences between PAFT-Bayes, DTA-EM and Static-F are small and not significant (Supplementary Table S7). In the Elliptic data the classifier collapses after time step 42 (F1 near zero for all rules, Figure ⟦F:real⟧), a known concept change in that dataset.

*Bank-time.* Adaptive rules exceed Static-F by ⟦bank_gain⟧, but this result should not be read as a success: the mean deployment AUC is ⟦bank_auc⟧, i.e. the classifier carries no information after the temporal change, the estimated prevalence is close to one, and the "gain" is that of predicting nearly everything positive as prevalence climbs to ⟦bank_pimax⟧.

**The guard.** On the real suites the goodness-of-fit test rejects the label-shift model on ⟦real_fire_min⟧–100% of batches, so PAFT-Guard coincides with Static-F almost everywhere: it avoids the significant losses on ACS and electricity, and forgoes the spurious gain on bank-time. Across the five suites we found no case in which unguarded label-free adaptation was reliably beneficial for a classifier that still worked.

⟦TABLE:real⟧

## 6.7 Label-assisted adaptation (post-hoc Experiment 9)

Since quantification, not the threshold map, failed on the ACS suite, we tested whether a small labelled audit sample spent on prevalence alone helps (Table ⟦T:labelled⟧). With 50 labels per state, guarded PAFT-L improves on Static-F by ⟦lab_acs50⟧ (95% CI ⟦lab_acs50_ci⟧; Holm-adjusted *p* < 0.001) and on directly re-tuning the threshold on the same 50 labels by ⟦lab_acs50_retune⟧; with 1,000 labels direct re-tuning becomes better (⟦lab_acs1000_retune⟧). Under controlled label shift, 50 labels give ⟦lab_ctl50⟧ over Static-F, as much as the label-free rule, while re-tuning on 50 labels is worse than not adapting. On creditcard-time, PAFT-L is ⟦lab_cc200_retune⟧ better than re-tuning on 200 labels, which contain almost no positives, but not significantly better than Static-F.

The variant has a clear limit. On electricity-time, re-tuning on 50 labelled instances raises F1 from ⟦elec_static⟧ to ⟦lab_elec_retune⟧, whereas PAFT-L stays at the static level: under concept change the threshold map itself is obsolete and labels must be spent on the decision, or on retraining, rather than on prevalence. PAFT-L is therefore useful when few labels are affordable and the change is mainly one of prevalence, and it is not a substitute for drift adaptation.

⟦TABLE:labelled⟧

## 6.8 Computational cost (RQ5)

On one CPU thread (Table ⟦T:eff⟧), PAFT-Bayes takes ⟦eff_paft_2000⟧ ms for a batch of 2,000 scores and ⟦eff_paft_100000⟧ ms for 100,000, the guard adds ⟦eff_guard_2000⟧ ms and ⟦eff_guard_100000⟧ ms, and DTA-EM takes ⟦eff_dta_2000⟧ ms and ⟦eff_dta_100000⟧ ms. For small batches PAFT is slower than DTA-EM because of the fixed cost of the $G\times T$ expectation; for large batches it is faster because it needs no iteration or sort. The stored state is ⟦eff_mem⟧ MB, and the one-off development stage takes ⟦eff_fit⟧ s, most of it for the 50 bootstrap thresholds of the Static-Bag baseline. All of this is negligible next to retraining (⟦eff_retrain⟧ s for LightGBM on the same dataset) and to scoring itself. The improvement, where there is one, is thus essentially free; the costs of adaptation are statistical, not computational.

## 6.9 Error analysis (RQ5)

**Where PAFT loses.** Per batch, PAFT-Bayes is better than Static-F in ⟦err_win8⟧% of batches at ρ = 8 but only ⟦err_win05⟧% at ρ = 1/2, where it is worse in ⟦err_loss05⟧% (Supplementary Table S8): for small shifts the estimation noise is comparable to the benefit. Losses are bounded; the worst decile of the per-batch difference is never below ⟦err_worstdecile⟧.

**Few positives.** Regret against the oracle threshold falls from ⟦err_regret_15⟧ for batches with 1–5 positives to ⟦err_regret_1000⟧ for batches with more than 1,000, in step with the relative prevalence error (median ⟦err_rel_15⟧ and ⟦err_rel_1000⟧; Table ⟦T:error⟧, Figure ⟦F:error⟧). With 1–5 positives PAFT-Bayes is no better than Static-F (⟦err_paft_15⟧ against ⟦err_static_15⟧), while DTA-EM retains an advantage (⟦err_dta_15⟧).

**Datasets.** The gain over Static-F is smallest for creditcard (⟦err_cc_gain⟧), where PAFT-Bayes is ⟦err_cc_dta⟧ below DTA-EM. With about ⟦cc_valpos⟧ validation positives the empirical true-positive-rate function is a coarse step function and the mixture histogram has only eight bins; the transductive rule, which ranks the batch itself, is less affected. The gain does not correlate with classifier AUC (Spearman ρ = ⟦err_sp_auc⟧, *p* = ⟦err_sp_auc_p⟧) and tends to increase with development prevalence (ρ = ⟦err_sp_prev⟧, *p* = ⟦err_sp_prev_p⟧).

**Direction of the threshold.** In ⟦err_dir⟧% of shifted batches the PAFT threshold on the raw score moves opposite to the prevalence, as Corollary 2 anticipates.

**Real shifts.** The dominant failure is not a property of PAFT but of the shared assumption: when (1) fails, the mixture likelihood attributes covariate change to prevalence change (Figure ⟦F:real⟧, top row), and every rule that trusts the estimate inherits the error.

⟦TABLE:error⟧

# 7 Discussion

**What should a practitioner do?** The evidence supports a simple protocol. If deployment prevalence is expected to move and the mechanism generating features from classes is stable (which is plausible when the class causes the features [⟦c:scholkopf2012⟧] and was true by construction in our controlled benchmark), adapt the threshold: the expected gain in F1 is a few points and the cost is negligible. Always run the guard: it is cheap, and on five real suites it was the difference between harming and not harming. If the guard fires persistently, label-free prevalence estimates should not be trusted; a few dozen labels spent on prevalence help if the classifier still ranks well, and otherwise the problem is concept drift and calls for re-tuning on labels or retraining [⟦c:gama2014⟧, ⟦c:lu2019⟧].

**PAFT or DTA-EM?** Our results do not support replacing the decision-theoretic rule by PAFT on grounds of F1. The two differ in kind. DTA-EM is transductive: it needs the whole batch before any instance can be decided and returns a set, not a threshold. PAFT returns a threshold on the score, which can be applied to instances as they arrive after a warm-up batch, can be logged and audited, and can be compared across batches; it requires no calibrated posteriors; and it applies unchanged to MCC or any other confusion-matrix metric. DTA-EM is more robust when validation positives number fewer than about one hundred. An equally important finding is that the decision-theoretic rule, which is rarely used as a baseline in the prior-shift literature, is very strong.

**Why does quantification fail on real data?** Mixture-based quantifiers interpret any movement of the score distribution as a change of mixing weight. On the ACS suite the score distribution moves because occupations and education differ between states, not only because incomes do. The oracle analysis separates the two links of the chain: the map from prevalence to threshold transferred to other states, while the map from scores to prevalence did not. This suggests that work on shift-robust quantification, rather than on thresholding, is where further progress would come from.

**Relation to the known theory.** Equation (2) and the dependence of F-optimal thresholds on prevalence are classical [⟦c:provost2001⟧, ⟦c:lipton2014⟧, ⟦c:flach2015⟧]. Our analytical additions are modest: the shifted-prior form of the optimality condition and the two corollaries, both of which are borne out by the experiments (the asymmetric behaviour of EM-Fixed, and the direction of threshold movement).

# 8 Limitations

(1) PAFT is not superior to DTA-EM in F1, and is inferior with very few validation positives. (2) The positive results rely on simulated label shift; on the five real suites label-free adaptation did not help, and we cannot point to a real dataset where it did for a working classifier. Suites in which label shift is the dominant real mechanism (for example case mix across clinical sites) were not available to us in tabular form and remain to be tested. (3) The guard's reference distribution is approximate: it fires above its nominal level, always on the rarest-class dataset, and with large batches it will reject for deviations too small to matter. (4) The controlled protocol resamples positives with replacement from pools that are small for the rarest classes, which understates batch-to-batch variability there. (5) All problems are binary and tabular; multi-class F-measures require a search over a simplex and were not studied. (6) Base learners use fixed hyper-parameters; better-tuned models would change absolute F1 but, since all rules share the scores, are unlikely to change their ordering. (7) The label-assisted variant was added after seeing the real-shift results and needs confirmation on independent data. (8) We did not study sequential dependence between batches beyond an exploratory variant (Supplementary Table S4), nor adversarial manipulation of the batch.

# 9 Conclusion

A threshold tuned for the F-measure is tuned for one prevalence. Under label shift, re-deriving it from validation ROC information and an estimated prevalence recovers most of the loss on 18 tabular datasets at negligible computational cost, and the proposed Bayes rule does so as well as, but not better than, expected-F maximisation on EM-corrected posteriors, while extending to other metrics and yielding an explicit threshold. On real temporal and cross-site shifts the label-shift assumption failed, label-free adaptation gave no reliable benefit, and a goodness-of-fit guard was needed to avoid harm. We therefore recommend guarded adaptation as a default, a small labelled audit sample for prevalence when the guard fires and the classifier still ranks well, and re-tuning or retraining otherwise.

# Declarations

**Funding** [To be completed by the authors.]

**Competing interests** [To be completed by the authors.]

**Ethics approval** Not applicable. The study uses only public, de-identified datasets and involves no human participants or animals.

**Data availability** All datasets are public: OpenML (data identifiers in the code repository), the UCI Machine Learning Repository (Bank Marketing), the Elliptic data set, and the US Census Bureau 2018 American Community Survey one-year public-use microdata. The download script retrieves all of them.

**Code availability** The complete source code, configuration, raw results, and the scripts that regenerate every table and figure are provided with the submission [anonymised link to be inserted].

**Author contributions** [To be completed by the authors.]

**Use of AI tools** [To be completed by the authors in accordance with the publisher's policy; see the note in the cover letter.]

# References

⟦REFERENCES⟧
