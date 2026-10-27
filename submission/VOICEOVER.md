# Voice-over for the demo (4:51, recorded without sound)

Start each line at its time; the teleprompter shows it in yellow at that moment. Calm pace: each line has a
short breath after it. 609 words in all. The closing card holds about 3 seconds after your last line.
Captions of these lines are burned into the final video, so it also works muted.

| Start | On screen | Say |
|---|---|---|
| 0:00 | CAM · Title: WoundTrace | Hi, I'm Aryan. I built WoundTrace on my own for the OpenCV AI Competition. |
| 0:06.5 | CAM · (same) | In 2019, 10.5 million Medicare patients had a wound, and nurses track healing by its area. |
| 0:14.5 | CAM · (same) | But photos sent from home are tilted and blurry, and my own printer shrank the reference card. |
| 0:22.5 | Live app on AWS, patient view | This is the live app on AWS. The patient takes one photo next to a printed card. |
| 0:30.5 | Pipeline 1: my real phone photo of a printed test target | Here's what OpenCV 5 does with a real photo from my phone, of a printed test target. |
| 0:38.5 | Pipeline 2: ArUco markers, pose axes, quality readings | It finds the card's ArUco markers to sub-pixel accuracy. solvePnP gives the tilt, the marker edges give the blur. |
| 0:47 | Pipeline 3: flattened plane, 10 mm grid | A homography from the eight corners flattens the photo into true millimetres. |
| 0:53 | Pipeline 4: model probabilities (heat map) | Then a segmentation network I trained, running in OpenCV's DNN module, scores every pixel. |
| 0:59.5 | Pipeline 4b: model outline (orange) vs GrabCut snap (green) | GrabCut, seeded by those scores, snaps the outline to the real edge. |
| 1:05.5 | Sample: tilted photo; the agent works | Now the agent on Amazon Bedrock. First, a tilted photo. |
| 1:10.5 | Result: retake, tilt 56° | Fifty-six degrees is too steep to measure, so it asks for one specific fix instead of guessing. |
| 1:18.5 | Real wound photo, no card (CC BY dataset): wound outlined, vision model's box, no size | A real wound photo without the card: the network still finds it and Nova agrees, but with no scale, no size. |
| 1:28.5 | Sample: good photo; the agent works | Now a good photo. The agent picks each OpenCV tool in turn. |
| 1:34 | Trace: review_outline, IoU 0.93, agree | Then it checks OpenCV's work with its own eyes. Nova marks where it sees the wound on the plain photo. |
| 1:43.5 | (same) | Code compares that with OpenCV's outline. If they disagree, a nurse checks it before anything is saved. |
| 1:51.5 | Result: 7.12 ± 0.20 cm², photo with outline and the model's box | Here they agree: 7.12 square centimetres, plus or minus 0.2. |
| 1:56.5 | Trace: stalled → nurse note drafted | After four weeks it's only 13 percent smaller, so the agent drafts a care-plan note for the nurse. |
| 2:05 | My ruler on my printed card: 27.7 mm, not 30 | At home, my printer had quietly shrunk the card to 92 percent. Every area came out 17 percent too large. |
| 2:14 | Patient C: card not checked → asks for a one-time card check | So for a new patient, the agent records nothing until the printed card is checked. |
| 2:21.5 | Sample with a bank card; the agent works | The patient adds any bank card, and OpenCV compares its outline with the standard size. |
| 2:28.5 | Result: card printed at 92.0%, corrected; card outlined | This test card printed at 92 percent, so every measurement is corrected. |
| 2:34.5 | Nurse view: the drafted note waits | In the nurse view, drafted notes wait with their evidence. Nothing is sent on its own. |
| 2:42 | Click Approve | The nurse approves with one click. |
| 2:45.5 | Patient trend: error bars, 4-week line | Each chart shows the measurements, with error bars, against the four-week benchmark. |
| 2:51 | Architecture on AWS (docs/architecture.svg) | Under the hood, Lambda on Graviton runs OpenCV 5 and the ONNX model. The agent is Nova 2 Lite on Bedrock. |
| 3:01 | (same) | The model picks the tools and writes the messages. Plain code gates every action. |
| 3:07.5 | Card: how do I know it works? | So how do I know it works? My first version used colour rules, tuned on synthetic photos, within 1.4 percent. |
| 3:17 | Scoreboard 1: FUSeg, colour rules 0.03 vs network 0.84 | On 200 real foot ulcers outlined by experts, it failed: a Dice overlap of 0.03 out of 1. |
| 3:25 | (same) | So I trained a network on that dataset's separate training photos. Dice rose to 0.84. |
| 3:32.5 | Scoreboard 2: two more clinics, before retraining | Two more datasets: it fell to 0.48 on wounds from Colombia, and drew wounds on healthy feet. |
| 3:40.5 | Scoreboard 3: after retraining (held-out photos) | So I retrained with their training photos, and healthy feet as examples of no wound. |
| 3:47.5 | (same) | On photos it never saw: 0.77 in Colombia, zero false alarms on 1,378 healthy feet, and 0.82 on the first set. |
| 3:57 | Lower-limb examples, never trained on (CC BY 4.0, cited) | On a clinic it never trained on, it's still weak: 0.39. Its outlines sit on the wound but cover less than marked. |
| 4:07 | Evidence: outline check (live app) | I tested the agent's outline check the same way. My first version approved every bad outline. |
| 4:15 | (same) | The redesign sends 79 percent of bad outlines to the nurse, and also 28 percent of good ones. |
| 4:23 | Evidence: synthetic + agent (live app) | On 750 held-out synthetic photos, the mean area error is 1.5 percent, and the agent matched the clinic's rules in 60 of 60 visits. |
| 4:34 | Evidence: not validated | Not proven yet: area accuracy on patients' own photos. That needs a clinical study. |
| 4:41 | CAM · End card | WoundTrace: a wound measurement you can trust, or a clear reason to take another photo. |

## Honest notes
- App shots are the live app on AWS (Lambda + Bedrock Nova 2 Lite), recorded headless in a browser on my laptop
  right after resetting the demo data. Demo patients and sample photos are synthetic test targets, not people
  or wounds; the app says so on screen. The agent's wording varies per run; the narration never quotes it.
- The "inside OpenCV" frames come from scripts/pipeline_frames.py run on one real iPhone photo of my printed
  colour test sheet A (real/a/04.jpg; true area 7.07 cm², measured 6.81), not a wound. The stripes are a
  printer fault. They show what the code computes; the captions were added by the script.
- The real wound photo (no card) is from the Lower Limb and Feet Wound Image Dataset (Islam et al., CC BY
  4.0), never used in training; it is also a sample in the live app. I picked it because the wound is clearly
  visible (its Dice is at the 90th percentile of that set). Without the card the app gives no size.
- The scoreboard numbers are from eval/fuseg_summary.md, eval/external_summary.md and the previous model's
  runs of the same scripts on the same held-out photos (kept in data/v1/). "Retrained" = the deployed model.
- No FUSeg or CO2Wounds photo is shown (their licences don't allow the commercial licence the rules ask of
  submitted media); only numbers. The four example photos are from the Lower Limb and Feet Wound Image
  Dataset (Islam et al., Mendeley Data, CC BY 4.0), cited on screen, picked at the 90th, 50th and 10th
  percentile of our Dice plus one healthy foot, not the best ones.
- "10.5 million Medicare patients had a wound" (2019): Carter MJ et al., J Med Econ 2023 (beneficiaries with
  a wound; 2014 and 2019 used different data sets).
- "Four-week benchmark" / "care-plan note": SVS 2016 recommends considering adjunctive therapy for diabetic
  foot ulcers that have not improved by more than 50% in area after at least 4 weeks.
- "Shrunk the card to 92 percent ... 17 percent too large": a ruler read one 30 mm square as 27.7 mm (92.3%);
  1 / 0.923² = 1.17. The on-screen sample simulates a 92.5% print (measured 92.0%).
- 79% / 28%: eval/review_summary.md (261 outlines, excluding the 40 used to design the check); the first
  design passed 100% of bad outlines (eval/review_overlay_summary.md). 1.4% (first version) and 1.5%:
  eval/summary.md history and current. 60 of 60: eval/agent_summary.md.
- The ruler photo is my own printed card; the red edge lines and the reading were added on the computer, and I
  re-measured it by hand (27.7 mm).
- Built with help from Claude Code (AI coding assistant), disclosed in the README and on Devpost.
