# WoundTrace

![WoundTrace: a wound measurement you can trust, or a clear reason to retake](docs/hero.png)

**Measure a wound from a phone photo. An agent checks the photo, measures the area in cm² with its
uncertainty, and asks a nurse to review when healing stalls. A human approves every alert.**

Built for the OpenCV AI Competition 2026 (powered by AWS). OpenCV 5 does the measuring; an agent on
Amazon Bedrock decides the next step; plain code gates every action.

- **Live demo:** https://asme326si3qdbv4raxgm4fbbge0fohve.lambda-url.us-east-1.on.aws/
- **Code:** https://github.com/woooj02/woundtrace
- **Technical report:** [REPORT.md](REPORT.md) (method, evaluation, failure cases, limitations)
- **Video:** _link added on submission_

> Measurement aid, not a diagnosis. Demo patients are printed or synthetic test targets. Real wounds enter
> only through three public, de-identified research datasets (cited below), used to train and test the outline.

## Why

Home-care nurses track chronic wounds by area over time, but a paper ruler's length × width overestimates
irregular wounds (by +43% on our synthetic test set), photos sent from home are often tilted, blurred or lit by a
flash, and a card printed at home is often printed at the wrong size. WoundTrace turns one phone photo next
to a printed card into a trustworthy number, or a specific instruction to take a better one.

## Try it (2 minutes, no printer needed)

1. Open the live demo. On **Patient**, click a sample photo and watch the agent's trace on the right.
   - *Patient A · tilted* or *flash glare*: the agent asks for a retake and says exactly what to change.
   - *Patient A · good photo*: measured, compared with 4 weeks of history, **stalled** → a nurse note is
     drafted and waits for approval.
   - *Patient C · card not checked* → the agent asks for a one-time card check; *Patient C · with a bank
     card* → it finds the plastic card, sees the printer shrank the card to 92%, and corrects the area.
   - *Real wound · no card* (a CC BY dataset photo): OpenCV's network still outlines the wound and the
     vision model checks it, but with no card there is no scale, so it gives no size and asks for the card.
2. **Nurse**: approve or dismiss the drafted note; see each patient's trend with ± error bars.
3. **How it works · evidence**: the architecture and every evaluation number, generated from the eval files.

With a printer: print `print/card-only.png` at 100% (black ink is enough), lay it flat next to the
target, and photograph from straight above.

## Results (details and failure cases in [REPORT.md](REPORT.md))

| Test | Result |
|---|---|
| **Real wounds, US clinic**: 200 expert-annotated foot-ulcer photos (FUSeg 2021 validation, never used in training) | outline Dice **0.82** vs the experts (85% of photos ≥ 0.7, 5 missed). My first, colour-based method: Dice **0.03**, 97 missed |
| **Real wounds, Colombia**: 122 smartphone photos of leprosy wounds (CO2Wounds-V2 validation, held out) | Dice **0.77** (79% ≥ 0.7, 2 missed). Before I added its training split: 0.48 |
| **Real wounds, a clinic never trained on**: 2,686 lower-limb photos | Dice **0.39**: outlines sit on the wound (precision 0.83) but cover less than this dataset marks, which includes surrounding skin (recall 0.37); 16% missed |
| **Healthy feet** (1,378 held out, no wound) | **0** given an outline (previous model: 23%) |
| **Vision model's outline check**: 261 real-wound outlines, held out from its design | sends **79%** of bad outlines to the nurse; also sends 28% of good ones. First design: caught 0% |
| Synthetic photos with known area, held-out set (750; tilt 0-60°, blur, glare, 6 skin tones) | 1.5% mean area error when recorded; 3 of 245 recorded with error > 10% |
| Same photos with the quality gates switched off | 5.8% mean error |
| Stated uncertainty (± 2 sd), held-out | contains the true area 95.6% of the time (target 95%) |
| Card printed at 92.5%, corrected with a plastic card (agent visits) | measured scale 0.921; recorded area error 1.3% (uncorrected: about +17%) |
| Real phone photos with known area (26: printed targets + a coin) | card found 26/26; 2 recorded, both within 8%; 17 sent to the nurse to confirm (11 of them within 10%); ± range too narrow on real photos |
| Agent on Bedrock vs the clinic's rules (60 visits, 10 scenarios) | same outcome **60/60**; outline check agreed on all 29 measured wounds; 0 guard refusals; 2 replies replaced |

![Three lower-limb wound photos (dataset mask vs ours) and a healthy foot with no outline](docs/real_examples.png)

*Photos: Lower Limb and Feet Wound Image Dataset (Islam et al., Mendeley Data, CC BY 4.0), never used to train
the model; picked at the 90th, 50th and 10th percentile of our Dice so the figure shows a typical and a poor
result, not only the best. FUSeg and CO2Wounds appear as numbers only: their licences don't cover republishing.*

## How it works

![Architecture](docs/architecture.svg)

![The OpenCV pipeline on one sample photo: markers and pose, flattened plane, model probabilities, GrabCut snap, outline on the photo, plastic-card check](docs/pipeline.png)

*One synthetic sample photo through the pipeline (`venv/bin/python -m scripts.pipeline_frames PHOTO`).*

- **OpenCV 5** (`app/vision.py`): ArUco detection with sub-pixel corners, IPPE pose (tilt), homography
  residual (card flat), marker edge width (blur in mm), perspective rectification to true millimetres,
  a learned wound segmenter run by the **DNN module** (LR-ASPP MobileNetV3, ONNX, [model card](app/models/README.md))
  whose probabilities seed **GrabCut** to snap the outline to the colour edge at full resolution, contour
  metrics (area, Feret length, width, perimeter), glare, and a plastic-card (ID-1) detector (Canny +
  contours) for the print scale. The classical colour pipeline remains as the cross-check.
- **Agent** (`app/agent.py`, `app/tools.py`): Bedrock Converse with 9 tools. The model picks the next tool
  and writes the messages. It also checks OpenCV's outline with its own eyes (`app/review.py`): shown the
  plain photo, it marks the wound, and code compares its box with the outline; if they disagree, the
  outline goes to the nurse instead of the record. Plain code refuses anything the clinic's rules don't allow: measuring a failed
  photo, recording without a "record" decision or an unchecked card, retaking a measurable photo, and
  sending any nurse note without human approval. A message guard checks every reply before the patient
  sees it. If Bedrock fails or the daily cap is reached, a rules engine runs the same tools.
  *Why an agent, not a script:* Across the 60 evaluation visits the model took 9 different tool paths, each chosen by what OpenCV found: 18 stopped at the photo check, 7 at the card check, 6 after measuring found glare, and 2 measured a second time. It adds what plain code can't: eyes (it finds the wound itself to check OpenCV's outline) and words (the patient's one specific fix, the nurse's note). It does not make clinical calls: plain code gates every record and alert, and matching the clinic's rules in 60 of 60 visits shows that handing it the wheel never broke one.
- **AWS**: Lambda (arm64 Graviton, Python 3.12, 2 GB) behind a Function URL, a private S3 bucket (photos
  expire after 30 days), Bedrock (Nova 2 Lite), CloudWatch Logs with one JSON line per visit and a
  dashboard, a least-privilege IAM role, and a per-visitor rate limit (30 visits an hour, hashed IP).

## Run it locally

```bash
./run.sh setup        # venv + pinned dependencies (requirements-dev.txt)
./run.sh              # http://127.0.0.1:8100 · rules engine, local storage, no AWS needed
./run.sh test         # 19 offline tests
venv/bin/python -m scripts.demo_data seed   # demo patients A, B, C in data/store
```

Use the Bedrock agent locally (needs AWS credentials with access to Nova 2 Lite in us-east-1):

```bash
USE_BEDROCK=1 AWS_DEFAULT_REGION=us-east-1 ./run.sh
```

## Reproduce the evaluation

Real-wound tests need the datasets in `data/` (not in this repo): FUSeg (`git clone
https://github.com/uwm-bigdata/wound-segmentation data/fuseg/repo`), CO2Wounds-V2 (Mendeley Data
10.17632/s2w7rjwz49.2, unzipped to `data/co2wounds/x/`) and the Lower Limb and Feet Wound Image Dataset
(Mendeley Data 10.17632/hsj38fwnvr.3, unzipped to `data/lowerlimb/x/`).

```bash
venv/bin/python -m eval.fuseg_eval                    # 200 real wound photos: eval/fuseg_summary.md
venv/bin/python -m eval.external_eval                 # two more clinics + healthy feet: eval/external_summary.md
AWS_DEFAULT_REGION=us-east-1 venv/bin/python -m eval.review_eval  # outline check, ~300 Bedrock calls, ~$0.15
data/train-venv/bin/python -m eval.train_segmenter --more-data --epochs 30   # retrain (PyTorch, GPU, ~45 min)
venv/bin/python -m eval.run_eval --seed 2026          # tuning seed (gates, uncertainty fit)
cp eval/results.json /tmp/fit.json
venv/bin/python -m eval.run_eval                      # held-out seed 11: eval/summary.md (~4 min, 12 cores)
venv/bin/python -m eval.fit_uncertainty /tmp/fit.json eval/results.json eval/uncertainty.json
AWS_DEFAULT_REGION=us-east-1 venv/bin/python -m eval.agent_eval   # 60 Bedrock visits, ~$0.30
venv/bin/python -m eval.calibration_eval > eval/calibration_summary.md   # misprinted card
venv/bin/python -m scripts.real_eval                  # needs real/ photos (see below)
venv/bin/python -m scripts.evidence                   # web/evidence.json for the app
```

The real phone photos were taken in a private home and are not published; their per-photo results are in
`real/summary.md` and `real/results.json`.

## Deploy to AWS

```bash
aws login                     # AWS CLI v2
./deploy/aws.sh seed          # bucket, IAM role, Lambda + Function URL, log group, demo patients
```

Idempotent; prints the live URL. It creates one private S3 bucket, one IAM role (its own log group, its
bucket's data prefixes, Nova 2 Lite only), one Lambda function with a public Function URL, and a log group
with 14-day retention. Cost: about $0.006 of Bedrock per agent visit (≈16k input and 240 output tokens, plus ~400 for the outline check),
capped at 150 visits a day; Lambda and S3 stay within the free tier at demo volume.

Settings (environment): `BEDROCK_MODEL`, `AGENT_ENGINE` (`bedrock`/`rules`), `DAILY_LLM_RUNS`,
`STALL_DAYS` and `STALL_REDUCTION` (healing rule, default 28 days / 50%), `BUCKET`, `DATA_DIR`.

## Repository

| Path | What |
|---|---|
| `app/vision.py` | OpenCV 5: card, quality checks, rectification, segmentation (DNN + GrabCut, classical), metrics, card calibration, uncertainty |
| `app/review.py` · `app/models/` | The vision model's outline check; the ONNX segmenter and its model card |
| `app/policy.py` | The clinic's rules: quality and outline decisions, healing rule |
| `app/tools.py` · `app/agent.py` | The 9 agent tools with their guards; the Bedrock loop, rules engine and message guard |
| `app/synth.py` · `app/card.py` | Synthetic targets and simulated phone photos with known area; the printed card |
| `app/main.py` · `web/index.html` | FastAPI routes; the single-page app (patient, nurse, evidence) |
| `eval/` | Real-wound (FUSeg, CO2Wounds, lower-limb, healthy feet), outline-check, synthetic, uncertainty and agent evaluations; model training |
| `real/summary.md` | Real phone photos of printed targets and a coin |
| `scripts/` | Demo data and printable sheets, real-photo evaluation, evidence file, figures (`pipeline_frames`, `examples_figure`) |
| `deploy/aws.sh` | One-command AWS deployment |
| `tests/` | Offline tests (rules engine, temporary storage) |

Dependencies are pinned in `requirements.txt` (runtime, also bundled into Lambda) and
`requirements-dev.txt` (server, tests).

## Responsible use

WoundTrace measures; it does not diagnose, predict an individual outcome or recommend treatment. Its
healing flag follows a guideline threshold and only prompts a clinician review, and every nurse note is a
draft until a human approves it. On real wounds only the outline has been tested (three clinics' photos, no
reference card); area accuracy on patients' own photos is untested. See the limitations in the
report and on the app's evidence page.

## Credits

Built by aryan-neopane (solo), with help from Claude Code (an AI coding assistant) during development. The
agent that counts for this entry is the runtime agent in the app (Amazon Bedrock), not the coding assistant.
The FastAPI app skeleton comes from our own starter kit, written before this event. OpenCV 5, FastAPI,
Mangum, NumPy, boto3; PyTorch and torchvision for training only. Real-wound data: C. Wang et al., "Fully
Automatic Wound Segmentation with Deep Convolutional Neural Networks", Scientific Reports 10:21897 (2020), and
the FUSeg 2021 challenge (photos from the AZH Wound and Vascular Center, shared by the authors with the
clinic's permission); D. Sánchez-Jiménez et al., CO2Wounds-V2 (Mendeley Data, CC BY-NC 3.0); M. M. Islam et
al., Lower Limb and Feet Wound Image Dataset (Mendeley Data, CC BY 4.0). MIT licence for this code (`LICENSE`);
the model weights are for non-commercial use, because part of their training data is CC BY-NC.
