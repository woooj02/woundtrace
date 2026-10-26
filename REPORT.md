# WoundTrace: technical report

OpenCV AI Competition 2026 · solo entry by aryan-neopane · live: https://asme326si3qdbv4raxgm4fbbge0fohve.lambda-url.us-east-1.on.aws/

## 1. Problem and users

Chronic wounds are common and growing: Medicare beneficiaries with a wound rose from 8.2 million in 2014 to
10.5 million in 2019 (Carter et al. 2023; the two years used different Medicare data sets) [3]. Between
visits, home-health and wound-clinic nurses track wounds by area. Two numbers matter: how big the wound is,
and how much it has shrunk. For diabetic foot ulcers, the Society for Vascular Surgery recommends
considering adjunctive therapy when the area has not fallen by more than 50% after at least 4 weeks of
standard care [1]; the 4-week change predicts 12-week healing (Sheehan et al. 2003, n = 203) [2].

Measuring at home fails in three ways: a ruler's length × width overestimates irregular wounds; phone photos
are tilted, blurred or lit by a flash; and a card printed at home is often printed at the wrong size.
**User:** a nurse following patients who send photos from home. **Promise:** one photo next to a printed card
becomes an area in cm² with its uncertainty, or a specific instruction to take a better photo; stalled
healing becomes a drafted note that the nurse approves.

## 2. What it does

One visit (diagram: [docs/architecture.svg](docs/architecture.svg)):

1. `check_photo`: find the card, estimate tilt, blur, resolution, flatness, brightness. Fail → `request_retake`
   with the specific fix ("hold the phone flat", "tap to focus", "turn off the flash").
2. `calibrate_card` (once per printed card): find a plastic bank/library card in the photo and measure the
   printed card's true size. None found → ask once for it, or for a ruler reading of one black square.
3. `measure_wound`: rectify to true millimetres, outline the wound, measure area ± uncertainty, length,
   width, perimeter; the clinic's rules decide record, retake (glare), measure again with a second method
   (weak edge), or ask the nurse to confirm the outline.
4. `get_history`, `assess_healing`: compare with baseline and last visit. Growth counts as worsening only if
   it exceeds 10% **and** the measurement noise of both photos.
5. `record_measurement`; `request_nurse_review` drafts a note (numbers, uncertainty, "photo measurement, not
   a diagnosis") that waits in the nurse queue until a human approves it.

## 3. OpenCV 5 pipeline (`app/vision.py`)

- **Card:** two 30 mm ArUco markers (`DICT_4X4_50`), `ArucoDetector` with sub-pixel corner refinement.
  Eight corners → `findHomography` to card millimetres; its RMS residual (mm) checks the card is flat.
- **Quality:** tilt from `solvePnP` (IPPE, assumed phone focal length); blur as the 10-90% rise width of the
  markers' outer edges, sampled along normals with `remap`, in mm; resolution in px/mm; brightness of the
  card's white strip. Gates: ≥ 2.5 px/mm, tilt ≤ 45°, edge ≤ 1.3 mm, residual ≤ 0.6 mm, brightness ≥ 70.
- **Rectification:** `warpPerspective` onto the skin plane at 6 px/mm.
- **Segmentation:** distance from the skin colour in Lab. Otsu split; blobs touching the frame are treated
  as background and the split is redone without them; the skin colour is then re-estimated from a ring 6-15
  mm around the first wound candidate. An **edge-seeking threshold** sweeps 16 thresholds and keeps the
  outline whose boundary has the strongest Sobel gradient (wound edges are sharp, periwound redness fades);
  yellow slough always counts as wound; outlines that leave holes deep inside the candidate (slough flecks)
  are rejected. Line-like or thin shapes (aspect > 8 or no point ≥ 3 mm across, from
  `minAreaRect` and `distanceTransform`) are not wounds. `grabCut` refines the outline.
- **Measurement:** area, maximum Feret length and perpendicular width (convex hull), the ruler's L × W,
  perimeter, solidity, glare fraction, and edge contrast across the boundary.
- **Card calibration:** rectify a wide window, `Canny` + `findContours`, keep rounded rectangles with the ID-1
  aspect ratio (85.60 × 53.98 mm); average the inner and outer contour of the edge line; print scale = true
  size / measured size, both sides must agree within 2.5%. The plastic card is masked out of segmentation.
- **Uncertainty:** relative sd² = k0² + (k1 · perimeter · (edge blur + pixel size) / area)² + (k2 · card residual)²,
  fitted by maximum likelihood on the tuning seed (k0 = 0.0039, k1 = 0.205, k2 = 0).

## 4. Agent (`app/agent.py`, `app/tools.py`)

Amazon Bedrock Converse with tool use (Nova 2 Lite), up to 10 steps, 8 tools. The model chooses tools and
words messages; plain code enforces the rules: no measuring a failed photo, no recording without a "record"
decision or an unchecked card, no retake for a measurable photo, ids pinned to the visit, nurse notes queued
for approval. An **outgoing-message guard** checks each patient reply for leaked internals, clinical opinions
("looks good") and length, and substitutes the template message if it fails. A deterministic rules engine
runs the same tools when Bedrock is unavailable or the daily cap (150 visits) is reached; the UI always says
which engine ran. Each visit is stored as a trace in S3 and logged as one JSON line to CloudWatch.

## 5. Evaluation

**Protocol.** Synthetic photos come from `app/synth.py`: an irregular wound (granulation texture, slough,
periwound redness) on one of six skin tones, photographed by a simulated camera (perspective, distance,
defocus, flash glare, noise, JPEG) with the exact true area known. Seed 2026 tuned the gates and fitted the
uncertainty; seed 7 was used to diagnose failures (below); **seed 11 is the held-out set** reported here.

| Synthetic, held-out (600 hard + 150 clean) | Result |
|---|---|
| Recorded / retake / nurse outline (hard set) | 245 / 355 / 0 |
| Mean area error when recorded · 95th percentile | 1.4% · 4.1% |
| Recorded with error > 10% | **0** |
| Same pipeline with gates off | 14.5% mean error |
| Clean set mean error | 0.9% |
| Ruler L × W vs true area | +44.0% |
| Error by skin tone 1 → 6 (light → dark) | 1.9 · 1.6 · 1.6 · 0.8 · 1.0 · 1.4% |
| Uncertainty: true area inside ± 1 sd / ± 2 sd | 66.3% / 96.6% (targets 68% / 95%) |

**Real phone photos with known area** (iPhone, room light; `real/summary.md`). Not wounds.

| Set | Photos | Outcome | Recorded errors |
|---|---|---|---|
| Colour sheet A, faulty printer (colour banding) | 11 | 9 nurse outline, 1 retake, 1 recorded | −20.7% |
| Colour sheet B, same printer | 12 | 6 nurse outline, 3 retakes, 3 recorded | −1.9, −5.3, −7.5% |
| Black-ink geometry sheet | 1 | recorded | +2.2% |
| $1 coin (25.00 mm), card calibrated by ruler | 2 | recorded | +6.1, +6.9% (≈ 4-5% is the coin's 3 mm thickness) |

The card was found in 26/26 photos. Only 2 of 7 recorded measurements fell inside their ± 2 sd: the
uncertainty model, calibrated on synthetic photos, is too confident on real ones.

**Card calibration** (`eval/calibration_summary.md`). In 12 synthetic photos with cards printed at 92.5-104%
and tilts of 5-35°, the plastic card was found in 11 (missed at 35° tilt, which leads to a retake); the print
scale was within 1.4% and corrected areas within 1.4% of the truth (uncorrected: up to +19%). In the
agent evaluation (card printed at 92.5%) the measured scale was 0.921 and recorded areas were 1.0% off.

**Agent on Bedrock** (`eval/agent_summary.md`): 60 visits, 10 scenarios × 6 skin tones, each also run through
the rules engine from the same state.

| Result | Value |
|---|---|
| Same outcome as the rules (record / retake / which nurse request) | 60/60 |
| Tool calls refused by the guards | 0 |
| Numbers in patient messages that came from tool outputs | 100% |
| Replies replaced by the message guard | 2 (clinical opinions) |
| Latency per visit, median / p95 | 4.9 / 7.1 s |
| Tokens per visit, input / output | 13,074 / 235 (≈ $0.005 at third-party listed Nova 2 Lite prices [5]) |

## 6. Failure cases found, and what changed

1. **Dark skin, slough only.** On held-out seed 7, 7 photos were recorded 64-98% too small, all on the
   darker skin tones (4-6): the outline traced yellow slough and missed the red wound bed, whose colour is close
   to dark skin. Causes: background polluting the Otsu split, a contaminated skin sample, and an outline
   chosen inside the wound. Fixed by the background-free re-split, the ring skin sample and the deep-core
   rule; on fresh seed 11, 0 bad accepts and no skin tone above 1.9%.
2. **Printed text measured as a wound.** On real sheet A, 5 photos were recorded wrongly, 4 of them near
   −60%: the outline was the printed caption or a thin paper edge. Added the line and thickness filters plus
   a policy backstop (implausible shape → nurse). Remaining: a-06, a blurry hand-held photo recorded −20.7%
   with edge contrast just above the gate.
3. **Faulty printer.** Banded colour made the printed skin striped; 15 of 23 colour-sheet photos went to the
   nurse instead of being recorded: correct caution, but no measurement.
4. **Misprinted card.** The coin read +24.5% because the printer shrank the card to 92.3% (confirmed with a
   ruler). Every home measurement would be biased the same way, and the photo looks perfect. Added card
   calibration; the agent will not record until the card is checked.
5. **Agent reasoning sent to patients.** In the first agent run, 16/60 replies on retake paths were the
   model's working notes ("Okay, the user sent a photo from patient e1…") and 2 said "your wound looks good".
   The message guard caught all 18; retake visits now use the patient message the model wrote for
   `request_retake`. Re-run: 2/60 replaced, both opinions.
6. **Conservative gates.** 220 of 322 measurable retakes would have measured within 5%: the gates trade
   patient effort for zero bad accepts. Loosening them is the main next step once real-wound data exists.

## 7. Limitations

- **Not validated on real wounds or patients.** Accuracy on real tissue (moisture, depth, hair, curved
  limbs, undermining) is unknown; all ground truth here is synthetic, printed or a coin.
- Synthetic skin is flat and evenly lit; the card-flatness term of the uncertainty could not be learned.
- The uncertainty is too narrow on real photos (2 of 7 inside ± 2 sd).
- Area only: no depth or volume, no tissue classification.
- Tilt uses an assumed focal length; the card must lie in the wound's plane.
- The healing threshold comes from diabetic foot ulcer evidence; for other wounds it is a configurable prompt.

## 8. Responsible use

A measurement aid, never a diagnosis, outcome prediction or treatment recommendation. Every nurse note is
a draft until a human approves it; the patient-facing text is checked by code. No real patient images were
used; the author's own test photos are kept out of the repository. The S3 bucket is private, the Lambda
role has least privilege, logs are kept 14 days, and Bedrock use is capped per day.

## 9. References

1. Hingorani A, et al. The management of diabetic foot: a clinical practice guideline by the Society for
   Vascular Surgery, APMA and SVM. J Vasc Surg 2016;63(2 Suppl):3S-21S.
2. Sheehan P, et al. Percent change in wound area of diabetic foot ulcers over a 4-week period is a robust
   predictor of complete healing in a 12-week prospective trial. Diabetes Care 2003;26(6):1879-82.
3. Carter MJ, et al. Chronic wound prevalence and the associated cost of treatment in Medicare
   beneficiaries: changes between 2014 and 2019. J Med Econ 2023;26(1):894-901.
4. ISO/IEC 7810: ID-1 card format, 85.60 × 53.98 mm.
5. Amazon Nova 2 Lite on Bedrock, about $0.33 / $2.75 per million input / output tokens (us. inference
   profile), from third-party price trackers; check AWS pricing.
