# Devpost submission draft: WoundTrace

Draft for the user to review and paste. Numbers come from eval/ and real/ (see REPORT.md). Do not submit
without the user's go-ahead.

**Project name:** WoundTrace

**Elevator pitch (≤ 200 chars):**
An agent that measures a wound from a phone photo with OpenCV 5, checks OpenCV's outline with its own eyes,
coaches a better photo, catches a misprinted card, and asks a nurse when healing stalls.

**Links:** live demo https://asme326si3qdbv4raxgm4fbbge0fohve.lambda-url.us-east-1.on.aws/ ·
repo _[GitHub URL after push]_ · video _[YouTube URL]_ · report: REPORT.md in the repo

**Prizes to select:** overall · Agentic Vision. (Not Best Use of COOL.)

---

## Inspiration

Nurses who follow chronic wounds between home visits need one number they can trust: the area, and how
fast it is shrinking. For diabetic foot ulcers, the Society for Vascular Surgery recommends considering
other therapy when the area hasn't fallen by more than half after 4 weeks. Yet a paper ruler's length ×
width overestimates irregular wounds, and photos sent from home are tilted, blurry, lit by a flash, or taken
next to a card the home printer quietly shrank.

## What it does

The patient photographs the wound next to a printed card. An agent on Amazon Bedrock runs OpenCV 5 tools and
decides what happens next:

- **Bad photo?** It names the fix: hold the phone flat, tap to focus, turn off the flash, move closer.
- **Card never checked?** Once, it finds a plastic bank or library card in the photo and measures how big
  the printed card really came out. My printer had shrunk it to 92%, which made every area 17% too large
  while the photo looked perfect.
- **Good photo?** It measures the area in cm² with its own uncertainty (for example 7.09 ± 0.20), then checks
  OpenCV's outline: the vision model marks the wound on the plain photo, and code compares the two. If they
  disagree, a nurse confirms the outline before anything is saved. It compares the area with earlier visits
  and only calls a rise "worsening" if it is bigger than the measurement noise.
- **Healing stalled?** It drafts a note for the nurse with the numbers and their limits. Nothing is sent
  until a human approves it.

## How I built it

- **OpenCV 5:** ArUco markers with sub-pixel corners; IPPE pose for tilt; marker edge width for blur in mm;
  homography residual for card flatness; `warpPerspective` to true millimetres; a segmentation network I
  trained (LR-ASPP MobileNetV3, exported to ONNX) run by the **DNN module**, whose probabilities seed
  **GrabCut** to snap the outline to the edge; contour metrics; Canny + contours to find the plastic card; a
  fitted uncertainty model. The classical colour pipeline remains as the cross-check.
- **Agent:** Bedrock Converse tool use (Nova 2 Lite) with 9 tools, including `review_outline`. The model picks the tools and writes the
  messages; plain code enforces the clinic's rules, and a message guard checks every reply before a patient
  sees it. A rules engine takes over if Bedrock is down.
- **AWS:** Lambda on arm64 Graviton behind a Function URL (OpenCV and the ONNX model, no PyTorch), a private
  S3 bucket with 30-day expiry, CloudWatch logs (one JSON line per visit) and a dashboard, a least-privilege
  IAM role, a per-visitor rate limit, one deploy script.

## Evaluation

- **Real wounds (outline):** 200 expert-annotated foot-ulcer photos (FUSeg 2021 validation set, never used in
  training). Dice 0.84 against the experts' outlines, 88% of photos at 0.7 or better, 2 missed. My first,
  colour-based method scored 0.03 and missed 97.
- **The outline check:** on 261 real-wound outlines held out from its design, it sends 79% of bad outlines to
  the nurse, and also 28% of good ones.
- **Synthetic held-out set:** 750 photos with known area, six skin tones. 1.2% mean error when recorded; the
  stated ± 2 sd contained the true area 98.2% of the time.
- **Real phone photos (26):** printed targets and a coin. The card was found in all 26; 6 were recorded, all
  within 10%.
- **The agent:** 60 Bedrock visits matched the clinic's rules 60/60.

## Challenges I ran into

- **My first segmenter failed on real wounds.** Tuned on synthetic photos, it measured within 1.4% there, then
  scored Dice 0.03 on real foot ulcers. I trained a network on the dataset's training split instead.
- **My first outline check approved everything.** Shown OpenCV's outline and asked "does it follow the
  wound?", the vision model said yes to 100% of the bad outlines. Now it never sees the outline: it marks the
  wound itself, and code compares.
- **My printer shrank the card** to 92%, so every area read 17% too large; the agent now checks the card once.
- The first agent run leaked the model's reasoning into 16 patient replies; a message guard now checks
  every reply.

## Accomplishments I'm proud of

- Testing the work against real wounds, and replacing what failed instead of hiding it.
- An agent that checks the vision result with its own eyes, and plain code that decides what that means.
- An uncertainty that holds up on held-out data, and decisions that match the clinic's rules on every visit.

## What I learned

The hard part isn't the measuring; it's knowing when *not* to give a number. Most of the value is in the
retake and the calibration.

## What's next

- A clinical study: area accuracy on patients' own photos, across skin tones, and a real-world uncertainty fit.
- Fewer false alarms from the outline check, and looser quality gates (216 of 322 retakes would have
  measured within 5%).
- Curved limbs and depth.

## Limitations (say these plainly)

- On real wounds only the outline is tested (clinic photos, no card): area accuracy on patients' own photos
  is unproven.
- The uncertainty is too narrow on real photos (1 of 6 inside ± 2 sd); the outline check has 28% false alarms.
- A measurement aid, not a diagnosis.

**Built with:** opencv, python, fastapi, amazon-web-services, aws-lambda, amazon-bedrock, amazon-s3,
amazon-cloudwatch, graviton, numpy, onnx, pytorch (training only), claude-code

**AI use (say it plainly):** I used Claude Code, an AI coding assistant, to help write and test the code and
docs. The rules note that an AI coding assistant is not an agentic workflow; WoundTrace's agentic workflow is
the runtime agent on Amazon Bedrock, whose decisions are driven by the OpenCV results.
