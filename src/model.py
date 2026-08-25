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
import torch
import torch.nn as nn
import torch.nn.functional as F

N_DR, N_DME = 5, 3


# ── backbone ──────────────────────────────────────────────────────────────────
def build_backbone(name: str = "densenet121", pretrained: bool = True):
    """Returns (module, feature_dim). Prefers timm (present on Kaggle), falls back to
    torchvision so the same code runs locally."""
    try:
        import timm
        m = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
        return m, m.num_features
    except Exception as e:
        print(f"[model] timm unavailable or model unknown ({e}); using torchvision")
        import torchvision.models as tvm
        if not name.startswith("densenet121"):
            print(f"[model] torchvision fallback cannot serve '{name}', using densenet121")
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
        H = OrdinalHead if head == "ordinal" else SoftmaxHead
        self.dr_head = H(dim, N_DR, hidden, p_drop)
        self.dme_head = H(dim, N_DME, hidden, p_drop)

    def forward(self, x):
        f = self.backbone(x)
        if f.ndim > 2:                       # some timm models return a feature map
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return self.dr_head(f), self.dme_head(f)


# ── ordinal target construction ───────────────────────────────────────────────
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
    if head == "ordinal":
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
    if head != "ordinal":
        return logits.argmax(dim=1)
    fired = (torch.sigmoid(logits) > threshold).float()
    return torch.cummin(fired, dim=1).values.sum(dim=1).long().clamp(0, n_classes - 1)


@torch.no_grad()
def expected_grade(logits, head="ordinal"):
    """Continuous score, for threshold tuning and for QWK-optimal rounding.

    Ordinal: E[y] = sum_k P(y > k). Softmax: sum_c c * p_c.
    """
    if head != "ordinal":
        return (torch.softmax(logits, 1) *
                torch.arange(logits.shape[1], device=logits.device)).sum(1)
    return torch.sigmoid(logits).sum(dim=1)
