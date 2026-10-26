# Devpost submission draft: WoundTrace

Draft for the user to review and paste. Numbers come from eval/ and real/ (see REPORT.md). Do not submit
without the user's go-ahead.

**Project name:** WoundTrace

**Elevator pitch (≤ 200 chars):**
An agent that measures a wound from a phone photo with OpenCV 5, coaches a better photo when it can't,
catches a misprinted card, and asks a nurse to review stalled healing.

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
- **Good photo?** It measures the area in cm² with its own uncertainty (for example 5.19 ± 0.18), compares it
  with earlier visits, and only calls a rise "worsening" if it is bigger than the measurement noise.
- **Healing stalled?** It drafts a note for the nurse with the numbers and their limits. Nothing is sent
  until a human approves it.

## How I built it

- **OpenCV 5:** ArUco markers with sub-pixel corners; IPPE pose for tilt; marker edge width for blur in mm;
  homography residual for card flatness; `warpPerspective` to true millimetres; Lab colour distance with an
  edge-seeking threshold, slough handling and GrabCut; contour metrics; Canny + contours to find the plastic
  card; a fitted uncertainty model.
- **Agent:** Bedrock Converse tool use (Nova 2 Lite) with 8 tools. The model picks the tools and writes the
  messages; plain code enforces the clinic's rules, and a message guard checks every reply before a patient
  sees it. A rules engine takes over if Bedrock is down.
- **AWS:** Lambda on arm64 Graviton behind a Function URL, a private S3 bucket, CloudWatch logs with one JSON
  line per visit, a least-privilege IAM role, one deploy script.

## Evaluation

- **Synthetic held-out set:** 750 photos with known area, six skin tones. 1.4% mean error when recorded;
  zero measurements recorded more than 10% off (14.5% mean error with the quality gates off); the stated
  ± 2 sd contained the true area 96.6% of the time.
- **Real phone photos (26):** printed targets and a coin. The card was found in all 26; 7 were recorded, and
  1 of those was more than 10% off.
- **The agent:** 60 Bedrock visits matched the clinic's rules 60/60.

## Challenges I ran into

Testing on real photos found three bugs the synthetic set hadn't:
- It measured printed text as a wound.
- On darker skin it outlined only the yellow slough.
- My printer shrank the card.

The first agent run also leaked the model's reasoning into 16 patient replies; the message guard caught
every one, and a fix brought it down to zero.

## Accomplishments I'm proud of

- Zero bad measurements recorded on the held-out set.
- An uncertainty that holds up on held-out data.
- An agent whose decisions match the clinic's rules on every evaluation visit.
- I built honest evidence, including the failures.

## What I learned

The hard part isn't the measuring; it's knowing when *not* to give a number. Most of the value is in the
retake and the calibration.

## What's next

- Validation on real wounds with a clinical partner, and a real-world uncertainty fit.
- Loosening the quality gates: 220 of 322 retakes would have measured within 5%.
- Curved limbs and depth.

## Limitations (say these plainly)

- Not validated on real wounds or patients.
- The uncertainty is too narrow on real photos (2 of 7 inside ± 2 sd).
- A measurement aid, not a diagnosis.

**Built with:** opencv, python, fastapi, amazon-web-services, aws-lambda, amazon-bedrock, amazon-s3,
amazon-cloudwatch, graviton, numpy, claude-code

**AI use (say it plainly):** I used Claude Code, an AI coding assistant, to help write and test the code and
docs. The rules note that an AI coding assistant is not an agentic workflow; WoundTrace's agentic workflow is
the runtime agent on Amazon Bedrock, whose decisions are driven by the OpenCV results.
