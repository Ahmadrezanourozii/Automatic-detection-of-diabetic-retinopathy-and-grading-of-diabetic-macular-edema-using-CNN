# Part A — IDRiD derivation gate: acceptance criterion, registered BEFORE any number

Committed before the derivation was computed. `FINAL-PROMPT.md` A.3 requires this, and the
project's own rule (`PROTOCOL.md`, pre-registration) forbids looking at agreement and then
deciding what counts as a pass.

## The question

The DME grade is defined as a deterministic geometric function:

* **grade 0** — no hard exudates;
* **grade 2** — minimum distance from any hard-exudate pixel to the fovea centre ≤ one optic
  disc diameter;
* **grade 1** — otherwise.

Does that derivation reproduce IDRiD's expert 3-class grade, under ideal conditions —
pixel-level hard-exudate masks, expert landmarks, hard and soft exudates separated?

## What will be reported, whatever it says

1. **Overall exact-match rate.**
2. **The full 3×3 confusion matrix**, derived versus expert.
3. **Per-class recall, with grade 1 reported separately.** Grade 1 is the entire reason this
   exercise exists; an overall match rate that hides a grade-1 failure is **not a pass**.
4. **The number of images the test actually covers**, and how the correspondence between the
   segmentation subset's numbering and the grading set's numbering was established.
5. **The verdict repeated under three definitions of "one optic disc diameter"**, because the
   threshold is *defined* in disc diameters and the disc is an ellipse:
   * major axis of the disc mask;
   * equivalent-area diameter, `2·sqrt(area/π)`;
   * minor axis.
   If the verdict flips between definitions, that is the finding and the derivation is not
   robust regardless of which number is highest.

## Verdict categories, fixed in advance

* **Reproduces** — high exact match **and** grade-1 recall that is not degenerate. The method
  is valid under ideal conditions; SUSTech-SYSU becomes worth the engineering, with its two
  caveats stacked on top. **Stop and report before starting it.**
* **Does not reproduce** — the derivation route dies here at zero GPU cost, and that is a
  reportable negative: the clinical grading definition is not recoverable from the annotations
  the field publishes, which would explain why no second 3-class DME corpus exists.
* **Partly reproduces** — e.g. grades 0 and 2 recovered, grade 1 not. The most interesting
  outcome, and it gets its own analysis: it would locate the divergence at the boundary case.

**No threshold on "high" is set here deliberately.** Fixing an arbitrary cut (say 80 %) would
invite arguing about the cut rather than the mechanism. The commitment is instead: report all
five items above, and let the grade-1 row carry the verdict, since a derivation that cannot
place the middle grade is useless for the purpose that motivated it — generating more middle-
grade labels.

## What this test cannot establish

It tests the derivation under IDRiD's *own* annotations. A pass would not license applying it
to a corpus whose exudate labels are bounding boxes, or which merges hard and soft exudates —
both true of SUSTech-SYSU. Those caveats stack on top of this verdict rather than being
answered by it.
