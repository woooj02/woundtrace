DEVPOST: paste each block into its field, top to bottom. Every number comes from eval/ and real/ (see REPORT.md).
Everything is filled in. Paste top to bottom, then submit.

==================== PROJECT NAME ====================
WoundTrace

==================== ELEVATOR PITCH (max 200 characters) ====================
An agent that measures a wound from a phone photo with OpenCV 5 on AWS, checks its own outline, coaches a better photo, catches a misprinted card and asks a nurse when healing stalls.

==================== TRY IT OUT LINKS ====================
https://asme326si3qdbv4raxgm4fbbge0fohve.lambda-url.us-east-1.on.aws/
https://github.com/woooj02/woundtrace

==================== VIDEO DEMO LINK ====================
https://youtu.be/AAn2KQZTjbw

==================== BUILT WITH (tags, one at a time) ====================
opencv, python, fastapi, amazon-web-services, aws-lambda, amazon-bedrock, amazon-nova, amazon-s3, amazon-cloudwatch, graviton, onnx, numpy, pytorch, claude-code

==================== THUMBNAIL (3:2) ====================
submission/gallery/0-thumbnail.png

==================== IMAGE GALLERY (upload, in this order, with these captions) ====================
submission/gallery/1-app.png       The live app: the agent's trace, the measurement with its uncertainty, the outline check.
submission/gallery/2-pipeline.png  Inside OpenCV 5: ArUco + solvePnP, homography to true millimetres, DNN segmenter, GrabCut snap, plastic-card check.
submission/gallery/3-architecture.png  Lambda on Graviton runs OpenCV 5 and the ONNX model; Nova 2 Lite on Bedrock picks the tools; plain code gates every action.
submission/gallery/4-real-wounds.png  Real wound photos from a clinic the model never trained on (CC BY 4.0 dataset), picked at the 90th, 50th and 10th percentile, plus a healthy foot.
submission/gallery/5-results.png   Outline accuracy on three clinics' held-out photos, before and after retraining.
submission/gallery/6-real-wound-no-card.png  A real wound photo with no card (CC BY 4.0 dataset) in the live app: OpenCV's network outlines it, the vision model agrees, and with no scale it gives no size.

==================== ABOUT THE PROJECT (Markdown) ====================
## Inspiration

In 2019, 10.5 million Medicare patients had a wound, and nurses track healing by its area. For diabetic foot ulcers, the Society for Vascular Surgery recommends reconsidering treatment when the area hasn't fallen by more than half after 4 weeks. But at home the area is guessed with a paper ruler (length × width overestimates irregular wounds by about 43%), photos arrive tilted, blurry or lit by a flash, and the reference card is printed at home. My own printer quietly shrank mine to 92%, which made every area 17% too large while the photos looked perfect.

## What it does

The patient takes one photo of the wound next to a printed card. An agent on Amazon Bedrock runs OpenCV 5 tools and decides what happens next:

- **Bad photo?** It names the one fix: hold the phone flat, tap to focus, turn off the flash, move closer. It never guesses a number from a photo it can't trust. No card at all? It still shows the patient where it sees the wound (OpenCV's network, checked by the vision model) so they know the right spot is in view, but gives no size without a scale.
- **Card never checked?** Once per patient, OpenCV finds a plastic bank card in the photo and measures how big the printed card really came out, so a misprinted card can't bias every later visit.
- **Good photo?** It measures the area in cm² with its own uncertainty (for example 7.12 ± 0.20), then checks OpenCV's work with its own eyes: the vision model marks the wound on the plain photo and plain code compares the two. If they disagree, a nurse confirms the outline before anything is saved.
- **Healing stalled?** It compares with earlier visits, calls a rise "worsening" only when it beats the measurement noise, and drafts a note for the nurse with the numbers and their limits. Nothing is sent until a human approves it.

## How I built it

- **OpenCV 5** does the measuring: ArUco markers with sub-pixel corners; `solvePnP` (IPPE) for tilt; marker edge width for blur in mm; the homography residual to catch a bent card; `warpPerspective` to true millimetres; a wound segmenter I trained (LR-ASPP MobileNetV3, exported to ONNX) run by the **DNN module**, whose probabilities seed **GrabCut** to snap the outline to the real edge; contour metrics; Canny + contours to find the plastic card; and a fitted uncertainty model.
- **The agent** is Bedrock Converse tool use with Nova 2 Lite and 9 tools. The model picks the tools and writes the messages; plain code enforces the clinic's rules (no measuring a failed photo, no recording before the card and outline checks, no nurse note without approval). A message guard checks every reply before a patient sees it, and a rules engine takes over if Bedrock is down.
- **AWS:** Lambda on arm64 Graviton behind a Function URL (OpenCV and the ONNX model, no PyTorch at runtime), a private S3 bucket with expiry, CloudWatch logs (one JSON line per visit) and a dashboard, a least-privilege IAM role, a per-visitor rate limit, and one deploy script.

**Why an agent, not a script?** Across the 60 evaluation visits the model took 9 different tool paths, each chosen by what OpenCV found: 18 stopped at the photo check, 7 at the card check, 6 after measuring found glare, and 2 measured a second time. It adds what plain code can't: eyes (it finds the wound itself to check OpenCV's outline) and words (the patient's one specific fix, the nurse's note). It does not make clinical calls: plain code gates every record and alert, and matching the clinic's rules in 60 of 60 visits shows that handing it the wheel never broke one.

## How I know it works

- **Real wounds from three clinics** (outline vs the annotators' masks, photos never used in training): Dice **0.82** on 200 US foot ulcers (FUSeg 2021), **0.77** on 122 smartphone photos from Colombia (CO2Wounds-V2), **0.39** on 2,686 photos from a clinic it never trained on, and **no** false outline on 1,378 healthy feet.
- **The outline check:** on 261 real-wound outlines held out from its design, it sends 79% of bad outlines to the nurse, and also 28% of good ones.
- **Area with known truth:** 750 held-out synthetic photos (six skin tones, tilt, blur, glare): 1.5% mean error when recorded; the stated ± 2 sd contained the true area 95.6% of the time.
- **The agent:** 60 Bedrock visits matched the clinic's rules in 60/60.
- **Real phone photos** of printed targets: the card was found in 26 of 26.

## Challenges I ran into

- **My first segmenter failed on real wounds.** Colour rules tuned on synthetic photos measured within 1.4% there, then scored Dice 0.03 on real foot ulcers. I trained a network instead: 0.84.
- **Then that network failed on other clinics:** 0.48 on Colombia's photos, and it drew wounds on 23% of healthy feet. I retrained with their training split and healthy feet as examples of no wound: 0.77 and 0%, for a cost of 0.02 on the first dataset.
- **My first outline check approved everything.** Shown OpenCV's outline and asked "does it follow the wound?", the vision model said yes to 100% of the bad ones. Now it never sees the outline: it finds the wound itself, and code compares.
- **My printer shrank the card** to 92%. The agent now checks the card once per patient.
- The first agent run leaked the model's reasoning into 16 patient replies; a message guard now checks every reply.

## Accomplishments that I'm proud of

- Testing on real wounds from three clinics, publishing the weak number (0.39) next to the strong ones, and fixing what failed instead of hiding it.
- An agent whose decisions are driven by what OpenCV measures, that checks the vision result with its own eyes, and that hands every judgement call to a human.
- An uncertainty that holds up on held-out data: the app knows when *not* to give a number.

## What I learned

The hard part isn't measuring; it's knowing when not to trust the measurement. Every test I added found a failure the previous one couldn't see, so the evaluation became the product.

## What's next

- A clinical study: area accuracy on patients' own photos across skin tones, and an uncertainty fitted on real photos.
- Better recall on new clinics, and fewer false alarms from the outline check.
- Curved limbs and depth.

## Limitations (said plainly)

- On real wounds only the outline is tested (clinic photos, no card): area accuracy on patients' own photos is unproven.
- On a clinic the model never trained on it misses 16% of wounds and outlines less than the annotators mark.
- The outline check has 28% false alarms; the ± range is too narrow on real photos. A measurement aid, not a diagnosis.

## Data and AI use

Real wound photos come only from three public, de-identified research datasets, cited in the repo; none is redistributed, and only the CC BY 4.0 set appears in images. The model weights are for non-commercial use because part of the training data is CC BY-NC. I used Claude Code, an AI coding assistant, to help write and test the code and docs; WoundTrace's agentic workflow is the runtime agent on Amazon Bedrock.

==================== IF THE FORM ASKS ====================
Prizes / special awards: Overall, and the Agentic Vision Award. (Not Best Use of COOL.)
Team: solo, Aryan Neopane (Devpost: aryan-neopane).
Technical report: REPORT.md in the repo (https://github.com/woooj02/woundtrace/blob/main/REPORT.md).
AWS services: Lambda (arm64 Graviton, Function URL), Amazon Bedrock (Nova 2 Lite, Converse tool use), S3, CloudWatch Logs and dashboard, IAM.
How OpenCV 5 is used: every measurement is OpenCV 5 (opencv-contrib-python-headless 5.0.0.93): aruco, solvePnP, findHomography, warpPerspective, dnn (ONNX segmenter), grabCut, contours, Canny. See README "How it works".
Where the agent acts on vision results: the photo checks decide retake vs measure; the measurement and the outline check decide record vs nurse; the history decides the nurse note. Trace and evaluation: the app's trace panel and eval/agent_summary.md.

==================== SPECIAL AWARD CONSIDERATION ====================
Agentic Vision Award (only; not Best Use of COOL).

==================== REPOSITORY URL ====================
https://github.com/woooj02/woundtrace

==================== UPLOAD A FILE ====================
submission/WoundTrace-report.pdf

==================== TESTING INSTRUCTIONS ====================
No login, no install, no printer needed.

1. Open the live app (AWS Lambda + Amazon Bedrock): https://asme326si3qdbv4raxgm4fbbge0fohve.lambda-url.us-east-1.on.aws/
2. Patient tab: click a sample photo and watch "What the agent did" (each OpenCV tool call, the vision model's outline check, the decision):
   - "Patient A · tilted" or "flash glare": the agent refuses to measure and asks for one specific fix.
   - "Patient A · good photo": measured with its uncertainty (about 7.1 ± 0.2 cm²), outline checked by the vision model, healing stalled after 4 weeks, so a nurse note is drafted and waits for approval.
   - "Patient C · card not checked", then "Patient C · with a bank card": the agent asks for a one-time card check, then finds the plastic card and corrects for the misprinted card (printed at about 92%).
   - "Real wound · no card": a real photo from a public CC BY dataset; OpenCV still outlines the wound and the vision model checks it, but with no card there is no scale, so it gives no size.
3. Nurse tab: approve or dismiss the drafted note; see each patient's trend with ± error bars.
4. How it works · evidence tab: the architecture and every evaluation number, generated from the repo's eval files.

Notes: demo patients and samples are synthetic or printed test targets (labelled in the app). A visit takes about 5-10 s on Bedrock. If the daily Bedrock cap is reached, a rules engine runs the same tools and the app says so.

Run locally (Python 3.12, Linux/macOS): ./run.sh setup && ./run.sh test (19 offline tests) && ./run.sh  -> http://127.0.0.1:8100 (rules engine, no AWS needed). Deploy: ./deploy/aws.sh seed. Reproducing the evaluations: see README "Reproduce the evaluation".
