# Voice-over for the demo (4:42, recorded without sound)

Start each line at its time; the teleprompter shows it in yellow at that moment. Calm pace: each line has a
short breath after it. 587 words in all. The closing card holds about 3 seconds after your last line.
Captions of these lines are burned into the final video, so it also works muted.

| Start | On screen | Say |
|---|---|---|
| 0:00 | CAM · Title: WoundTrace | Hi, I'm Aryan. I built WoundTrace on my own for the OpenCV AI Competition. |
| 0:06.5 | CAM · (same) | In 2019, 10.5 million Medicare patients had a wound, and nurses track healing by its area. |
| 0:14.5 | CAM · (same) | But photos sent from home are tilted and blurry, and my own printer shrank the reference card. |
| 0:22.5 | Live app on AWS, patient view | This is the live app on AWS. The patient takes one photo next to a printed card. |
| 0:30.5 | Pipeline 1: my real phone photo of a printed test target | Here's what OpenCV 5 does with a real photo from my phone, of a printed test target. |
| 0:38.5 | Pipeline 2: ArUco markers, pose axes, quality readings | It finds the card's ArUco markers and refines their corners to sub-pixel accuracy. solvePnP gives the tilt. |
| 0:46 | (same) | The sharpness of the marker edges gives the blur, in millimetres. |
| 0:51.5 | Pipeline 3: flattened plane, 10 mm grid | A homography from the eight corners flattens the photo into true millimetres. |
| 0:57.5 | Pipeline 4: model probabilities (heat map) | Then a segmentation network I trained, running in OpenCV's DNN module, scores every pixel. |
| 1:04.5 | Pipeline 4b: model outline (orange) vs GrabCut snap (green) | GrabCut, seeded by those scores, snaps the outline to the real edge at full resolution. |
| 1:11.5 | Pipeline 6: outline back on the photo | Area, length, width and perimeter, projected back onto the photo. |
| 1:16.5 | Sample: tilted photo; the agent works | Now the agent on Amazon Bedrock. First, a tilted photo. |
| 1:21.5 | Result: retake, tilt 56° | Fifty-six degrees is too steep to measure, so it asks for one specific fix instead of guessing. |
| 1:29.5 | Sample: good photo; the agent works | Now a good photo. The agent picks each OpenCV tool in turn. |
| 1:35.5 | Trace: review_outline, IoU 0.93, agree | Then it checks OpenCV's work with its own eyes. Nova marks where it sees the wound on the plain photo. |
| 1:45 | (same) | Code compares that with OpenCV's outline. If they disagree, a nurse checks it before anything is saved. |
| 1:52.5 | Result: 7.09 ± 0.20 cm², photo with outline and the model's box | Here they agree: 7.09 square centimetres, plus or minus 0.2, from this photo's sharpness and resolution. |
| 2:00.5 | Stats: ruler L×W | A ruler's length times width would have said 9.5. |
| 2:05 | Trace: stalled → nurse note drafted | After four weeks it's only 13 percent smaller, so the agent drafts a care-plan note for the nurse. |
| 2:13.5 | My ruler on my printed card: 27.7 mm, not 30 | Testing at home taught me something: my printer had quietly shrunk the card to 92 percent. |
| 2:21 | (same) | That makes every area 17 percent too large, and the photo still looks perfect. |
| 2:27.5 | Patient C: card not checked → asks for a one-time card check | So for a new patient, the agent records nothing until the printed card is checked. |
| 2:35 | Sample with a bank card; the agent works | The patient adds any bank card. OpenCV finds its outline and compares it with the standard size. |
| 2:43 | Result: card printed at 92.0%, corrected; card outlined | This test card printed at 92 percent, so this measurement and every later one is corrected. |
| 2:50.5 | Nurse view: the drafted note waits | In the nurse view, drafted notes wait with their evidence. Nothing is sent on its own. |
| 2:58 | Click Approve | The nurse approves with one click. |
| 3:01.5 | Patient trend: error bars, 4-week line | Each chart shows the measurements with error bars against the four-week benchmark. |
| 3:07 | Architecture on AWS (docs/architecture.svg) | Under the hood, Lambda on Graviton runs OpenCV 5 and the ONNX model. The agent is Nova 2 Lite on Bedrock. |
| 3:17 | (same) | The model picks the tools, checks the outline and writes the messages. Plain code gates every action. |
| 3:25 | Card: how do I know it works? | So how do I know it works? My first version used colour rules, tuned on synthetic photos. |
| 3:33 | (same) | On those, its mean error was 1.4 percent. Then I tested it on real wounds. |
| 3:40 | FUSeg: 3 real foot ulcers, experts vs ours vs classical (cited) | On 200 foot-ulcer photos outlined by wound experts, it failed: a Dice overlap of 0.03 out of 1. |
| 3:48.5 | (same) | So I trained a network on the dataset's separate training photos, plus synthetic ones. On the 200 unseen photos, Dice is 0.84. |
| 3:58.5 | Evidence: real wounds + outline check (live app) | I tested the outline check the same way. My first version approved every bad outline. |
| 4:05.5 | (same) | The redesign sends 79 percent of bad outlines to the nurse, and also 28 percent of good ones. |
| 4:14 | Evidence: synthetic + agent (live app) | On 750 held-out synthetic photos, the mean area error is 1.2 percent, and the agent matched the clinic's rules in 60 of 60 visits. |
| 4:25 | Evidence: not validated | Not proven yet: area accuracy on patients' own photos. That needs a clinical study. |
| 4:31.5 | CAM · End card | WoundTrace: a wound measurement you can trust, or a clear reason to take another photo. |

## Honest notes
- App shots are the live app on AWS (Lambda + Bedrock Nova 2 Lite), recorded headless in a browser on my laptop
  right after resetting the demo data. Demo patients and sample photos are synthetic test targets, not people
  or wounds; the app says so on screen. The agent's wording varies per run; the narration never quotes it.
- The "inside OpenCV" frames come from scripts/pipeline_frames.py run on one real iPhone photo of my printed
  black-ink test target (real/bw-medium/01.jpg), not a wound. They show what the code computes; the captions
  were added by the script.
- The three wound photos are from FUSeg 2021 (Wang et al., Sci Rep 2020; AZH Wound and Vascular Center),
  shared publicly by the authors with the clinic's permission; cited on screen. Picked at the 90th, 50th and
  10th percentile of our Dice, not the best ones.
- "10.5 million Medicare patients had a wound" (2019): Carter MJ et al., J Med Econ 2023 (beneficiaries with
  a wound; 2014 and 2019 used different data sets).
- "Four-week benchmark" / "care-plan note": SVS 2016 recommends considering adjunctive therapy for diabetic
  foot ulcers that have not improved by more than 50% in area after at least 4 weeks.
- "Shrunk the card to 92 percent ... 17 percent too large": a ruler read one 30 mm square as 27.7 mm (92.3%);
  1 / 0.923² = 1.17. The on-screen sample simulates a 92.5% print (measured 92.0%).
- Dice 0.03 and 0.84: eval/fuseg_summary.md (200 FUSeg validation photos, never used in training).
  79% / 28%: eval/review_summary.md (261 outlines, excluding the 40 used to design the check); the first
  design passed 100% of bad outlines (eval/review_overlay_summary.md). 1.4% (first version) and 1.2%:
  eval/summary.md history and current. 60 of 60: eval/agent_summary.md.
- The ruler photo is my own printed card; the red edge lines and the reading were added on the computer, and I
  re-measured it by hand (27.7 mm).
- Built with help from Claude Code (AI coding assistant), disclosed in the README and on Devpost.
