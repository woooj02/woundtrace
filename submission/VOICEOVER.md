# Voice-over for the demo (3:30, recorded without sound)

Start each line at its time; the teleprompter shows it in yellow at that moment. Calm pace: each line has a
short breath after it. 438 words in all. The closing card holds about 3 seconds after your last line.
Captions of these lines are burned into the final video, so it also works muted.

| Start | On screen | Say |
|---|---|---|
| 0:00 | CAM · Title: WoundTrace | Hi, I'm Aryan. I built WoundTrace on my own for the OpenCV AI Competition. |
| 0:06.5 | CAM · (same) | In 2019, 10.5 million Medicare patients had a wound, and their nurses track it by area. |
| 0:14.5 | CAM · (same) | But photos sent from home are tilted, blurry, or next to a card printed at the wrong size. |
| 0:22.5 | Live app on AWS, patient view | This is the live app on AWS. The patient photographs the wound beside a printed card. |
| 0:30.5 | Sample: tilted photo; the agent works | First, a tilted photo. OpenCV finds the card's markers and works out the camera's tilt. |
| 0:37.5 | Result: retake, tilt 56° | Fifty-six degrees is too steep to measure, so the agent asks for one specific fix. |
| 0:44.5 | Sample: good photo; the agent works | Now a good photo. The agent on Amazon Bedrock picks which OpenCV tool to run next, step by step. |
| 0:53.5 | Result: 7.21 ± 0.20 cm², photo with outline | The card's corners flatten the photo to true millimetres, and the wound measures 7.21 square centimetres. |
| 1:01 | Stats: ± range, ruler, % smaller | Plus or minus 0.2, from this photo's sharpness and resolution. A ruler's length times width would say 9.67. |
| 1:09.5 | Trace: stalled → nurse note drafted | After four weeks it's only 11 percent smaller. Guidelines use half the area by four weeks to review care. |
| 1:18 | (same) | So the agent drafts a note for the nurse. Every step and every decision is in this trace. |
| 1:26.5 | My ruler on my printed card: 27.7 mm, not 30 | Testing with my own printer taught me something: it had quietly shrunk the card to 92 percent. |
| 1:34.5 | (same) | That makes every area about 17 percent too large, and the photo still looks perfect. |
| 1:41.5 | Patient C: card not checked → asks for a one-time card check | So for a new patient, the agent won't record anything until the printed card is checked. |
| 1:49.5 | Sample with a bank card; the agent works | The patient adds any bank or library card. OpenCV finds its outline and compares it with the standard size. |
| 1:58 | Result: card printed at 92.0%, corrected; card outlined | This test card printed at 92 percent, so this measurement and every later one is corrected. |
| 2:05.5 | Nurse view: the drafted note waits | In the nurse view, the drafted note waits with its evidence. Nothing is sent on its own. |
| 2:13.5 | Click Approve | The nurse approves it with one click. |
| 2:17.5 | Patient trend: error bars, 4-week line | Each patient's chart shows the measurements with error bars against the four-week benchmark. |
| 2:24 | Architecture on AWS (docs/architecture.svg) | Under the hood, Lambda on Graviton runs OpenCV 5, and the agent is Nova 2 Lite on Amazon Bedrock. |
| 2:32.5 | (same) | The model picks the tools and writes the messages. Plain code gates every action. |
| 2:39.5 | Evidence: synthetic held-out results | Now the evidence. On 750 held-out test photos across six skin tones, the mean error is 1.4 percent. |
| 2:47.5 | (same) | None was recorded more than 10 percent off, and the stated range held 96.6 percent of the time. |
| 2:56 | Evidence: real photos and the agent | On 26 real phone photos of printed targets and a coin, the card was found every time. |
| 3:04 | (same) | And the agent reached the same decision as the clinic's rules in all 60 test visits. |
| 3:11.5 | Evidence: not validated | Those real photos also found three bugs, which I fixed. It's not yet tested on real wounds. |
| 3:19.5 | CAM · End card | WoundTrace: a wound measurement you can trust, or a clear reason to take another photo. |

## Honest notes
- Everything on screen is the live app on AWS (Lambda + Bedrock Nova 2 Lite), recorded headless right after
  resetting the demo data. Demo patients and sample photos are synthetic test targets, not people or wounds;
  the app says so on screen. The agent's wording varies per run; the narration never quotes it.
- "10.5 million Medicare patients had a wound" (2019): Carter MJ et al., J Med Econ 2023 (beneficiaries with
  a wound; 2014 and 2019 used different data sets).
- "Guidelines use half the area by four weeks to review care": SVS 2016 recommends considering adjunctive
  therapy for diabetic foot ulcers that have not improved by more than 50% in area after at least 4 weeks.
- "My printer shrank the card to 92 percent ... about 17 percent too large": a ruler read one 30 mm square as
  27.7 mm (92.3%); 1 / 0.923² = 1.17. The on-screen sample simulates a 92.5% print (measured 92.0%).
- Evidence numbers are read from the app's Evidence section (eval/ and real/ results in the repo).
- The ruler photo is my own printed card; the red edge lines and the reading were added on the computer, and I
  re-measured it by hand (27.7 mm).
- "Hi, I'm Aryan": change to your own name if different.
