# Scope: WoundTrace (working name)

## Candidates (1–5)

| Idea | Fit | Value | Diff | Tech | Demo | Feas | Evid | Strat | Total |
|---|---|---|---|---|---|---|---|---|---|
| **WoundTrace**: an agent that measures a wound in cm² from a phone photo with an ArUco card, coaches retakes, tracks healing and escalates to a nurse | 5 | 5 | 4 | 5 | 5 | 4 | 5 | 4 | **37** |
| Generic agentic visual inspection (defects, gauges) | 4 | 3 | 1 | 4 | 4 | 4 | 4 | 4 | 28 |
| Rover active perception with ArUco (user's hardware background) | 4 | 3 | 4 | 4 | 4 | 2 | 3 | 4 | 28 |
| Home physio range-of-motion from video (pose) | 4 | 4 | 3 | 3 | 4 | 3 | 3 | 3 | 27 |
| Lab seed-germination counter | 3 | 2 | 3 | 4 | 4 | 4 | 4 | 3 | 27 |

Pick: WoundTrace. Generic inspection is cloned a dozen times in the field. The rover needs hardware on video
within one day. WoundTrace is classical OpenCV that a founder-judge can check (marker pose, homography,
segmentation, measurement). It is also naturally agentic: the image decides whether to retake, measure or
escalate. Trade-off: it touches health, so we stay a measurement aid, not a diagnosis, and use synthetic targets.

## Promise

A home-care patient snaps their wound next to a printed card; the agent checks the photo, asks for a better one
if it can't measure, measures the area in cm², and asks a nurse to review when healing stalls.

## User

A home-health or wound-clinic nurse following patients with chronic wounds (diabetic foot ulcers, pressure
injuries) between visits. [verify sources before using any number] Chronic wounds affect about 8 million
Medicare beneficiaries (Nussbaum et al., 2018). Ruler length × width overestimates area. A diabetic foot ulcer
that shrinks less than ~50% in 4 weeks is unlikely to heal by 12 weeks (Sheehan et al., 2003). That rule drives
the escalation.

## Aha moment

A blurry, tilted photo goes in. The trace shows: marker found, 38° tilt, blur score low → the agent asks
"hold the phone flat above the wound." The retake gets measured at 6.1 cm². Against last week's 6.4 cm² that is
only a 5% drop over 4 weeks, so the agent drafts a nurse alert and **waits for approval** before sending it.

## How it works (detail in spec.md)

- OpenCV 5 tools: ArUco detection and pose, blur (Laplacian variance), glare (saturated pixels), tilt from the
  homography, perspective rectification to true scale, wound segmentation (Lab colour + GrabCut), area,
  perimeter, longest axis.
- Agent: an LLM on Amazon Bedrock picks the tools and writes the instructions. Plain code holds the thresholds,
  the healing rule and the approval gate.
- AWS: the backend in a Lambda container (arm64) behind a Function URL, images in S3, Bedrock, CloudWatch logs.
- Evaluation: synthetic test set with known ground-truth area (rendered shapes plus random tilt, blur, glare and
  distance), giving area error and retake-decision accuracy, plus failure cases. Then a few real phone photos of
  printed targets.

## Non-goals

- No diagnosis, no infection or tissue classification, no real patient photos.
- No accounts, EHR integration, or native mobile app (mobile web only).
- No COOL award: it needs COOL from AWS Marketplace on Graviton plus benchmarks, too much for one day.
