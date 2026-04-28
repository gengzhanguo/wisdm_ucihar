# Speaking Script — 7 Minutes
**Cross-Dataset HAR Transfer**
Gengzhan · Forest · Gengzhan | University of Pennsylvania

---

## Slide 1 — Title (0:00–0:15)

Hi everyone. We're going to talk about a practical problem in human activity recognition —
what happens when you train a model on one dataset and try to deploy it on another.
Spoiler: it usually fails badly, and we wanted to understand why.

---

## Slide 2 — Motivation (0:15–1:00)

So here's the concrete problem. We took UCI HAR as the source — that's the standard
waist-mounted sensor dataset — and tried to run it on WISDM, which is collected from
phones in people's front pockets.

The result: macro F1 of 0.085. That's basically random chance for a 5-class problem.

And this isn't a model quality issue. The same CNN that achieves 95% on UCI HAR
completely collapses on WISDM. So the question we asked was: is this domain shift,
and if so, what kind?

---

## Slide 3 — Two Datasets (1:00–1:30)

These two datasets look similar on the surface — same activities, same accelerometer,
roughly similar size. But when you look closely, they're actually quite different.

UCI HAR is waist-mounted, 50 Hz, and Butterworth low-pass filtered before release.
WISDM is front-pocket, 20 Hz, completely raw.

After physical alignment — resampling to 20 Hz, converting units, matching window size —
both datasets give us 51×3 windows. But the distributions are still very different.

---

## Slide 4 — Seven Sources of Domain Shift (1:30–2:00)

We identified seven sources of domain shift. I'll go through the top ones.

The dominant source — and it's not even close — is gravity axis projection.
Wasserstein distance of 8.7 meters per second squared on the X-axis alone.
That number is basically the entire gravity vector.

The other sources — amplitude differences, noise structure, class prior shift — they all
matter, but they're secondary.

---

## Slide 5 — Dominant Shift: Gravity Axis (2:00–3:00)

Here's why gravity dominates. In UCI HAR, the sensor is strapped to the waist pointing
forward. Gravity falls almost entirely along the X-axis — mean around +8.58 m/s².

In WISDM, the phone is in your front pocket, free to rotate. Gravity ends up on the Y-axis,
with a lower mean of about +6.51 m/s².

So when a model trains on UCI HAR, it learns: "X has a big DC component — that's the
gravity signal, that's how I tell walking from standing." Then on WISDM, X is near
zero-mean. The gravity cue the model learned is just gone.

This isn't a subtle distribution shift. The sensor is physically rotated roughly 90 degrees.

---

## Slide 6 — Feature Space Gap (3:00–3:30)

This shows up dramatically in feature space. Before any adaptation, the multi-kernel MMD
between UCI HAR and WISDM is 0.830.

Looking at the PCA projection — the two datasets are almost completely disjoint.
A model trained on the red cluster genuinely cannot generalize to the green cluster.

---

## Slide 7 — PhysicalDistortion Pipeline (3:30–4:30)

So our approach was: instead of adapting the model, correct the data.

We built a five-operator pipeline called PhysicalDistortion that transforms UCI HAR
training windows to look like they came from a front-pocket sensor.

First, we apply a 90-degree rotation around the Z-axis, plus some random wobble to
simulate loose pocket placement.

Second, we scale down the gravity component — 7.29 over 9.69, which is the empirically
measured ratio.

Third — and this is important — we apply per-activity, per-axis amplitude scaling.
Not a global scale factor. The ratio for Sitting Y-axis is 4.82×, for Walking Y-axis it's 2.06×.
A global scale factor would get this very wrong.

Fourth, we boost the 0.8 to 2 Hz gait frequency band for locomotion activities.
WISDM has about 40% more energy there because it's unfiltered.

And fifth, we inject temporally correlated noise — AR(1) with coefficient 0.9865 —
because UCI HAR noise is white after filtering, but WISDM noise has strong temporal
correlation.

Each operator targets a specific, measured physical difference. No target labels needed.

---

## Slide 8 — After Alignment (4:30–5:00)

After PhysicalDistortion, the MMD drops from 0.830 to 0.453 — a 45% reduction.

You can see in the PCA plot that the augmented UCI data (in orange) now partially overlaps
with WISDM. The gap is substantially smaller.

Not gone — that remaining gap is probably why some per-class results are still imperfect.

---

## Slide 9 — Ablation Results (5:00–5:30)

Here's the main ablation. Four conditions across five architectures.

C0 Raw is basically non-functional — average F1 of 0.085. CNN1D with PhysicalDistortion
alone reaches 0.723. That's a 6× improvement for CNN1D, and 6.8× on average across
all five architectures.

The interesting finding: adding TTBN or DANN on top of PhysicalDistortion actually hurts.
Every single model gets worse when you add model-level adaptation on top of good
data augmentation.

---

## Slide 10 — Per-class Analysis (5:30–6:00)

Looking at per-class performance, the picture is uneven. Sitting at 0.966, Standing at 0.820,
Walking at 0.872 — these are all solid.

But Upstairs is 0.445. That's genuinely bad. Stair climbing produces a complicated signal
with both periodic and aperiodic components, and the inter-subject variability in how people
climb stairs is large. Window-level augmentation can't model step-to-step variability.
We don't have a good solution for this.

---

## Slide 11 — Why TTBN and DANN Don't Help (6:00–6:20)

Very briefly — after PhysicalDistortion, TTBN re-estimates batch norm statistics from the
test batch, but WISDM is 56% Walking versus UCI's 20%. The re-estimated statistics
get skewed by the class imbalance.

DANN tries to force domain-invariant features, but the shift is tied to the activity itself —
gravity projection changes per activity. Forcing invariance destroys the axis-specific
information the classifier actually needs.

---

## Slide 12 — Lessons Learned (6:20–6:45)

Three takeaways. One: diagnose before adapting. We spent time measuring the shift
before building anything, and that directly told us what to fix.

Two: physical corrections beat model corrections here. The root cause was in the data,
so fixing the data worked better than fixing the model.

Three: model adaptation made things worse when added on top of good augmentation.
The combination is not always additive.

---

## Slide 13 — Conclusion (6:45–7:00)

To wrap up: the dominant source of cross-dataset failure between UCI HAR and WISDM
is gravity axis projection — a purely physical difference in sensor placement.

PhysicalDistortion corrects for this without any target labels, taking CNN1D from F1 0.120
to 0.723. The lesson is: fix the physics first. Then think about the model.

Thanks — happy to take questions.

---

*Total estimated time: ~7:00*
*Pace: ~140 words/minute average*
