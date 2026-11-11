import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.methods.score_model import ScoreModel, fbeta_from_rates
from src.methods import quantify as Q
from src.methods.rules import DecisionLayer, dta_topk, mixture_ks_pvalue
from src.evaluation.batch_eval import Batch, evaluate_rules
from sklearn.metrics import f1_score, fbeta_score, matthews_corrcoef


def _data(n, pi, rng, shift=0.0):
    y = (rng.random(n) < pi).astype(int)
    z = rng.normal(2.0 * y + shift, 1.0)
    return 1 / (1 + np.exp(-(z - 1.0))), y


def test_surface_matches_empirical_f1():
    rng = np.random.default_rng(0)
    s, y = _data(5000, 0.1, rng)
    sm = ScoreModel(s, y)
    for i in [10, len(sm.thr) // 2, len(sm.thr) - 10]:
        t = sm.thr[i]
        assert abs(sm.surface(sm.pi0)[i] - f1_score(y, s >= t)) < 1e-9
        assert abs(sm.surface(sm.pi0, "f2")[i] - fbeta_score(y, s >= t, beta=2)) < 1e-9
        assert abs(sm.surface(sm.pi0, "mcc")[i] - matthews_corrcoef(y, s >= t)) < 1e-9


def test_batch_scoring_matches_sklearn():
    rng = np.random.default_rng(1)
    s, y = _data(2000, 0.2, rng)
    b = Batch(s, y)
    for t in [0.2, 0.5, 0.8]:
        k = b.k_of_threshold(t)
        assert k == int((s >= t).sum())
        assert abs(b.score(k)[0] - f1_score(y, s >= t)) < 1e-12
        assert abs(b.score(k, "mcc")[0] - matthews_corrcoef(y, s >= t)) < 1e-9
    assert b.oracle() >= max(b.score(b.k_of_threshold(t))[0] for t in np.linspace(0, 1, 50)) - 1e-12


def test_quantifiers_recover_prevalence_under_label_shift():
    rng = np.random.default_rng(2)
    sv, yv = _data(20000, 0.1, rng)
    sm = ScoreModel(sv, yv)
    grid = Q.pi_grid(sm.pi0)
    for pi in [0.02, 0.1, 0.4]:
        s, y = _data(20000, pi, rng)
        assert abs(Q.q_mle(sm, s, grid) - y.mean()) < 0.03
        assert abs(Q.q_hdy(sm, s, grid) - y.mean()) < 0.03


def test_lipton_condition_generalised():
    # at the optimum the posterior at the threshold equals F*/(1+beta^2) (Proposition 1)
    from scipy.stats import norm
    pi = 0.1
    for beta in [0.5, 1.0, 2.0]:
        t = np.linspace(-4, 6, 200001)
        tpr, fpr = norm.sf(t - 2.0), norm.sf(t)
        f = fbeta_from_rates(pi, tpr, fpr, beta)
        i = int(np.argmax(f))
        post = pi * norm.pdf(t[i] - 2) / (pi * norm.pdf(t[i] - 2) + (1 - pi) * norm.pdf(t[i]))
        assert abs(post - f[i] / (1 + beta ** 2)) < 1e-3


def test_paft_beats_static_under_shift_and_guard_detects_violation():
    rng = np.random.default_rng(3)
    sv, yv = _data(20000, 0.05, rng)
    dl = DecisionLayer(sv, yv)
    gains = []
    for _ in range(20):
        s, y = _data(4000, 0.4, rng)
        r = evaluate_rules(dl.decide(s)[0], Batch(s, y))
        gains.append(r["paft_bayes"][0] - r["static"][0])
    assert np.mean(gains) > 0.01
    s, y = _data(4000, 0.05, rng)
    assert mixture_ks_pvalue(dl.sm, np.sort(s), 0.05)[0] > 0.01
    s, y = _data(4000, 0.05, rng, shift=1.5)        # covariate/concept shift in score space
    out, info = dl.decide(s)
    assert info["guard_p"] < 0.01 and out["paft_guard"][1] == dl.t_static


def test_dta_topk():
    q = np.array([0.9, 0.8, 0.1, 0.05])
    assert dta_topk(q) == 2
