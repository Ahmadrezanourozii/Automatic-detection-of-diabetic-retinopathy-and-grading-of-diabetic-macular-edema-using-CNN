"""
model.py — the multi-output network and its losses.

The thesis contribution is fixed: one shared feature extractor, two heads, trained with
L = alpha*L_DR + beta*L_DME. Everything else (backbone, head design, loss form) is open,
and this file changes those.

THE ONE IDEA WORTH READING
--------------------------
Both targets are ordinal, so each head predicts K-1 *threshold* probabilities rather than
K class probabilities:

    head_k(x) = P(y > k)          k = 0 .. K-2

For DR that is 4 logits, for DME 2. Three things follow, and the third is the reason the
whole project is built this way:

1. It matches the metric. QWK charges (i-j)^2 for confusing grade i with j; a softmax over
   unordered classes has no idea that grade 1 is between 0 and 2. Threshold decomposition
   puts that ordering into the loss.

2. Decoding is rank-consistent by construction: count the leading thresholds that fire and
   stop at the first that does not. The model cannot output "not grade >0, but yes grade >1".

3. **It makes Messidor-2's coarse DME label usable with no approximation at all.**
   Messidor-2 labels "referable DME" = hard exudates within 1 disc diameter, which is
   exactly IDRiD's grade 2. In threshold form:

       referable = 1  ->  P(y>1) target 1     (P(y>0) also 1)
       referable = 0  ->  P(y>1) target 0     (P(y>0) UNKNOWN -> masked)

   So a corpus that cannot distinguish grade 0 from grade 1 simply supervises one threshold
   and leaves the other unsupervised, per sample. No marginal loss, no label guessing, no
   flattening IDRiD to binary. DME supervision goes from 348 gated images to 2 260.

See data/LABEL_MAPPING.md for the label justification and ISSUES.md §2 for why the class
names in the old code were clinically wrong.
"""
from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

N_DR, N_DME = 5, 3


# ── backbone ──────────────────────────────────────────────────────────────────
RETFOUND_ENCODER_SHA256 = "847f9dd0e33bf8d450cc6121295d2919fc4bba3c185757a17ce6427bfa14ed37"


def _build_retfound(weights_root: str):
    """RETFound CFP ViT-L/16, loaded from the stripped encoder and verified by hash.

    The path is not trusted — PROTOCOL.md §9 — and a partial state-dict load raises rather
    than silently yielding a randomly-initialised transformer, which would be indistinguishable
    from a null result. Kaggle sometimes mounts every dataset under one directory
    (ISSUES.md §15), so the search is recursive.
    """
    import glob, hashlib
    import timm
    if os.path.isfile(weights_root):
        hits = [weights_root]
    else:
        hits = sorted(glob.glob(os.path.join(weights_root, "**",
                                             "retfound_cfp_encoder.pth"), recursive=True))
    if not hits:
        raise SystemExit(f"retfound_cfp_encoder.pth not found under {weights_root}")
    path = hits[0]
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if h != RETFOUND_ENCODER_SHA256:
        raise SystemExit(f"RETFound encoder hash mismatch — refusing to run.\n"
                         f"  expected {RETFOUND_ENCODER_SHA256}\n  got      {h}")
    print(f"[model] RETFound encoder {path} sha256 verified", flush=True)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    m = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if not k.startswith("head")]
    if real_missing:
        raise SystemExit(f"RETFound encoder did not fully load; missing {real_missing[:8]}")
    print(f"[model] loaded {len(sd)} tensors, {len(unexpected)} unexpected", flush=True)
    return m, m.num_features


def build_backbone(name: str = "densenet121", pretrained: bool = True):
    """Returns (module, feature_dim).

    Prefers timm (present on Kaggle); torchvision serves densenet121 so the same code runs
    anywhere. Falling back to densenet121 when a *different* backbone was requested is
    forbidden: it would turn a backbone experiment into a silent duplicate of the baseline
    and then report "the backbone makes no difference" (the failure mode of ISSUES.md §10).
    """
    if name.startswith("retfound:"):
        return _build_retfound(name.split(":", 1)[1])

    try:
        import timm
        m = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
        return m, m.num_features
    except ImportError:
        if name != "densenet121":
            raise SystemExit(
                f"backbone '{name}' needs timm, which is not installed here. Refusing to "
                f"substitute densenet121: that would silently turn this run into the "
                f"baseline and the comparison would be meaningless.")
        print("[model] timm not installed; using torchvision densenet121")
    except Exception as e:
        raise SystemExit(f"timm could not create backbone '{name}': {e}")

    import torchvision.models as tvm
    w = tvm.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    m = tvm.densenet121(weights=w)
    dim = m.classifier.in_features
    m.classifier = nn.Identity()
    return m, dim


# ── heads ─────────────────────────────────────────────────────────────────────
class OrdinalHead(nn.Module):
    """K-1 threshold logits from a shared trunk feature."""

    def __init__(self, in_dim: int, n_classes: int, hidden: int = 256, p_drop: float = 0.4):
        super().__init__()
        self.n_classes = n_classes
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
        )
        self.thresholds = nn.Linear(hidden, n_classes - 1)

    def forward(self, x):
        return self.thresholds(self.trunk(x))          # (B, K-1) logits for P(y>k)


class CoralHead(nn.Module):
    """CORAL: one shared weight vector, K-1 learned biases (Cao, Mirjalili & Raschka 2020).

    The existing OrdinalHead learns an independent weight vector per threshold, so nothing in
    *training* prevents P(y > 2) from exceeding P(y > 1); `model.decode` repairs that at
    inference with `cummin`. CORAL makes the ordering structural instead: a single projection
    w·h with per-threshold biases b_k, so the logits are w·h + b_k and monotonicity holds by
    construction whenever the biases are ordered.

    THE HYPOTHESIS THIS TESTS (IDEAS.md). Because cut-points are already optimised by
    cross-fitted tuning, a better ordinal loss cannot help by moving the cuts — it must improve
    the underlying RANKING of images. CORAL constrains the model to a single ranking direction,
    which is either a useful inductive bias or an unnecessary restriction of capacity.
    """

    def __init__(self, in_dim: int, n_classes: int, hidden: int = 256, p_drop: float = 0.4):
        super().__init__()
        self.n_classes = n_classes
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
        )
        self.proj = nn.Linear(hidden, 1, bias=False)          # the shared direction
        self.bias = nn.Parameter(torch.zeros(n_classes - 1))  # per-threshold offsets

    def forward(self, x):
        return self.proj(self.trunk(x)) + self.bias           # (B, K-1)


class SoftmaxHead(nn.Module):
    """The old K-way head, kept so the ordinal change can be ablated against it."""

    def __init__(self, in_dim: int, n_classes: int, hidden: int = 256, p_drop: float = 0.4):
        super().__init__()
        self.n_classes = n_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class MultiOutputNet(nn.Module):
    """Shared extractor, DR head and DME head. The thesis architecture, modernised."""

    def __init__(self, backbone="densenet121", pretrained=True, head="ordinal",
                 hidden=256, p_drop=0.4):
        super().__init__()
        self.backbone, dim = build_backbone(backbone, pretrained)
        self.feature_dim = dim
        self.head_type = head
        H = {"ordinal": OrdinalHead, "coral": CoralHead}.get(head, SoftmaxHead)
        self.dr_head = H(dim, N_DR, hidden, p_drop)
        self.dme_head = H(dim, N_DME, hidden, p_drop)

    def _feat(self, x):
        f = self.backbone(x)
        if f.ndim > 2:                       # some timm models return a feature map
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return f

    def forward(self, x, xm=None):
        """xm is an optional macula-centred crop of the same eye (I07).

        When it is supplied the DME head reads features from the macula crop while the DR
        head keeps the whole fundus. The backbone is SHARED -- one set of weights, two
        forward passes -- so any DME change is attributable to what the head is looking at
        rather than to extra capacity. The DME grade is defined by exudate distance to the
        macula centre, and global average pooling over the whole fundus dilutes that region
        by roughly 16x at 448 px.
        """
        f = self._feat(x)
        fm = f if xm is None else self._feat(xm)
        return self.dr_head(f), self.dme_head(fm)


# ── ordinal target construction ───────────────────────────────────────────────
# CORAL emits the same (B, K-1) threshold logits with the same P(y > k) semantics as the
# plain ordinal head, so every ordinal code path -- targets, masked BCE, decode, expected
# grade -- applies unchanged. Only the head's parameterisation differs.
ORDINAL_HEADS = ("ordinal", "coral")


def is_ordinal(head: str) -> bool:
    return head in ORDINAL_HEADS


def ordinal_targets(y: torch.Tensor, n_classes: int):
    """y (B,) -> targets (B, K-1) where t[:,k] = 1 if y > k."""
    ks = torch.arange(n_classes - 1, device=y.device).unsqueeze(0)
    return (y.unsqueeze(1) > ks).float()


def dme_targets_and_mask(dme_lo: torch.Tensor, dme_hi: torch.Tensor):
    """
    Build per-threshold DME targets and a per-threshold supervision mask from the
    candidate interval [lo, hi] each row is known to lie in.

        IDRiD exact grade g   -> lo = hi = g            both thresholds supervised
        Messidor-2 referable  -> lo = hi = 2            both supervised (P(y>0)=P(y>1)=1)
        Messidor-2 not-ref.   -> lo = 0, hi = 1         only P(y>1)=0 supervised
        no DME label at all   -> lo = -1                nothing supervised

    A threshold k is supervised exactly when the answer to "is y > k?" is the same for
    every grade in the interval. That is the whole trick, and it is exact -- no
    approximation is being made anywhere.
    """
    ks = torch.arange(N_DME - 1, device=dme_lo.device).unsqueeze(0)   # (1, K-1)
    lo, hi = dme_lo.unsqueeze(1), dme_hi.unsqueeze(1)
    always = lo > ks                       # every candidate grade exceeds k
    never = hi <= ks                       # no candidate grade exceeds k
    mask = (always | never) & (lo >= 0)
    return always.float(), mask.float()


# ── losses ────────────────────────────────────────────────────────────────────
def masked_bce(logits, targets, mask, pos_weight=None, smoothing=0.0):
    """BCE over supervised (sample, threshold) entries only.

    Returns 0 when nothing in the batch is supervised, so a batch of DR-only images does
    not poison the DME head with a spurious gradient.
    """
    if smoothing > 0:
        targets = targets * (1 - smoothing) + 0.5 * smoothing
    loss = F.binary_cross_entropy_with_logits(
        logits, targets, weight=None, pos_weight=pos_weight, reduction="none")
    denom = mask.sum()
    if denom < 1:
        return logits.sum() * 0.0
    return (loss * mask).sum() / denom


def multitask_loss(dr_logits, dme_logits, batch, alpha=0.6, beta=0.4,
                   dr_pos_weight=None, dme_pos_weight=None, smoothing=0.0,
                   head="ordinal"):
    """L = alpha * L_DR + beta * L_DME, with per-sample masking on both heads."""
    if is_ordinal(head):
        dr_t = ordinal_targets(batch["dr"], N_DR)
        dr_m = batch["dr_mask"].unsqueeze(1).expand_as(dr_t)
        l_dr = masked_bce(dr_logits, dr_t, dr_m, dr_pos_weight, smoothing)

        dme_t, dme_m = dme_targets_and_mask(batch["dme_lo"], batch["dme_hi"])
        l_dme = masked_bce(dme_logits, dme_t, dme_m, dme_pos_weight, smoothing)
    else:
        dr_m = batch["dr_mask"]
        l_dr = (F.cross_entropy(dr_logits, batch["dr"].clamp(min=0), reduction="none",
                                label_smoothing=smoothing) * dr_m).sum() / dr_m.sum().clamp(min=1)
        # a softmax head cannot use a partial label -- only exactly-known rows contribute,
        # which is precisely the 2 260 -> 618 shrinkage the ordinal head avoids
        exact = (batch["dme_lo"] == batch["dme_hi"]) & (batch["dme_lo"] >= 0)
        em = exact.float()
        l_dme = (F.cross_entropy(dme_logits, batch["dme_lo"].clamp(min=0), reduction="none",
                                 label_smoothing=smoothing) * em).sum() / em.sum().clamp(min=1)
    return alpha * l_dr + beta * l_dme, l_dr.detach(), l_dme.detach()


# ── decoding ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def decode(logits, n_classes, head="ordinal", threshold=0.5):
    """Logits -> integer grade.

    Ordinal: count leading thresholds that fire, stopping at the first that does not.
    `cummin` makes the decode rank-consistent even if the raw probabilities are not
    monotone, so the model can never claim "not > 0 but yes > 1".
    """
    if not is_ordinal(head):
        return logits.argmax(dim=1)
    fired = (torch.sigmoid(logits) > threshold).float()
    return torch.cummin(fired, dim=1).values.sum(dim=1).long().clamp(0, n_classes - 1)


@torch.no_grad()
def expected_grade(logits, head="ordinal"):
    """Continuous score, for threshold tuning and for QWK-optimal rounding.

    Ordinal: E[y] = sum_k P(y > k). Softmax: sum_c c * p_c.
    """
    if not is_ordinal(head):
        return (torch.softmax(logits, 1) *
                torch.arange(logits.shape[1], device=logits.device)).sum(1)
    return torch.sigmoid(logits).sum(dim=1)
