# AGENTS.md

Project operating notes for contributors and coding agents. Read this before
changing desktop capture, recognition, packaging, or project documentation.

## Working agreement

1. Inspect the working tree and this file before changing code. Preserve any
   unrelated local changes.
2. Verify every code change in proportion to its risk. A user-facing desktop
   change requires focused tests; a release also requires a local package
   check before GitHub packaging.
3. At the end of every material task, update this file's **Current state** or
   **Progress log**. Record the outcome, affected version/commit when relevant,
   verification performed, and any remaining limitation. Do not add routine
   narration or duplicate git history.
4. Keep documentation aligned with the product:
   - Update `README.md` and `README.zh-CN.md` when installation, supported
     behavior, privacy, or user-visible limitations change.
   - Update the relevant file in `docs/` when a contract, architecture, or
     subsystem behavior changes.
   - Update release/version files together: `pyproject.toml`,
     `src/poker_engine/__init__.py`, `packaging/pokersense.spec`, and
     `packaging/pokersense.iss`.
5. Remove a Markdown document only after confirming it is obsolete or fully
   duplicated, checking references with `rg`, and moving any still-useful
   information into its replacement. An unlinked design document is not, by
   itself, evidence that it is disposable.

## Current state

- Default branch: `main`.
- Current release: `v0.1.11` — [GitHub Release](https://github.com/windgeek/PokerSense/releases/tag/v0.1.11), source commit `f963181`.
- The `main` desktop path reads WePoker Android from a portrait LDPlayer
  instance over ADB, recognizes **hero and board cards**, derives the street,
  reads the total pot, eight-slot occupancy/stacks, the visual-slot Dealer
  marker, Hero's current decision turn, and completed visual-slot action
  labels, and displays visible-card equity against a random range. Published v0.1.11
  installers still contain the legacy H5 path; do not describe Android capture
  as released yet.
- A different hero-card pair must be visible for two consecutive frames before
  it starts a new hand. This prevents deal-animation reads from replacing the
  prior hand while allowing the companion window to refresh on every deal.
- All calibrated Android fields, including per-slot stack/action observations,
  require two consecutive matching production frames before state processing.
- Android occupancy, stack, Dealer, action, and Hero actor each have independent
  measured geometry and confidence evidence. A versioned eight-slot mapping
  promotes them to canonical seats/positions. Completed action glyphs are
  deduplicated and become canonical events only when actor stack delta and pot
  evidence are coherent. Opponent current-turn timers and complex missed-action/
  side-pot sequences remain unavailable without additional measured Replay.
- Capture frames are memory-only. The only persistent user setting is UI
  language (`auto`, `en`, or `zh`).

## Live-capture constraints

- Production input is `adb -s <serial> exec-out screencap -p` from LDPlayer,
  currently calibrated at 1440x2560 portrait. Host window coordinates, DPI,
  visibility, and occlusion are not part of the Android TableMap.
- Never choose among multiple ADB devices implicitly. `auto` is allowed only
  when exactly one authorized device exists; otherwise require
  `--device-serial` or `POKERSENSE_ADB_SERIAL`.
- Resolve ADB from `POKERSENSE_ADB_PATH` or PATH. Treat offline, unauthorized,
  timeout, corrupt PNG, and device disappearance as recoverable capture errors.
- Android and H5 never share ROIs or calibration evidence. They may share
  platform-neutral recognition algorithms and identical card-art templates.
- Real calibration screenshots and ZIPs are private inputs, ignored by Git,
  and never packaged. Retain only a small redacted labeled regression set.

## Local development and packaging

Python 3.11–3.13 is supported. On this machine Python 3.13 is used:

```bash
python3 -m venv .venv
./.venv/bin/pip install -e ".[dev,perceptual,desktop,packaging]"

./.venv/bin/python -m pytest -q
./.venv/bin/python -m flake8 src tests
./.venv/bin/pyinstaller packaging/pokersense.spec \
  --distpath dist --workpath build --noconfirm
```

For a macOS build, verify the bundle version and signature structure:

```bash
/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' \
  dist/PokerSense.app/Contents/Info.plist
codesign --verify --deep --strict --verbose=2 dist/PokerSense.app
```

The user validates GitHub release installers. Local verification should cover
tests, lint, package construction, bundle version, and code-signature
structure; do not represent a local build as a clean-user installation test.

## Documentation map

- `README.md` and `README.zh-CN.md`: product scope, installation, privacy,
  usage, and release-facing limitations.
- `docs/product-requirements.md`: product requirements, multi-scenario and
  multi-player capability matrix, input/output contracts, and staged
  acceptance criteria.
- `docs/strategy-requirements-matrix.md`: post-recognition strategy functions,
  required inputs, processing rules, output contracts, and router decisions.
- `docs/strategy-regression-test-matrix.md`: executable test suites, fixtures,
  coverage rules, performance budgets, CI tiers, and release gates.
- `docs/capture-replay.md`: hash-pinned raw-frame Replay registration,
  recognizer evidence binding, quality-report contract, and R6 eligibility.
- `docs/capture-card-calibration-guide.zh-CN.md`: AI-executable private-data
  calibration and acceptance specification for the candidate phone-to-PC USB
  capture-card path; its evidence remains separate from LDPlayer and H5. The
  hardware-independent tooling for it lives in
  `tools/capture_card_calibration/`
  (`python -m tools.capture_card_calibration.cli --help`); stages A and B
  still need real hardware and a human operator.
- `PLAN-capture-card-calibration.zh-CN.md`: the step-by-step development and
  work plan that turns the calibration guide's stages A-L into ordered,
  one-action-at-a-time tasks, splitting each into what the agent can build
  (code/config/tooling) and what needs real hardware or a human operator.
- `docs/recognition-ui-handoff.md`: practical handoff for the parallel WPK
  recognition and Live Coach UI work, including existing modules, owned gaps,
  frozen contracts, sequencing, and acceptance checklists.
- `docs/test-report-2026-08-23.md`: commit-bound development regression and
  target-hardware performance results, environmental blockers, requirement
  verdicts, and the remaining release evidence.
- `architecture.md`: overall design and data flow.
- `docs/`: subsystem contracts and architecture decisions. Keep these files
  concise and update the owning document rather than creating duplicate notes.
- `configs/vision/wepoker_android/` and `configs/platform/`: measured recognition
  calibration, not marketing claims. Update calibration evidence together with
  recognizer changes.
- `configs/strategy/`: versioned strategy/equity performance calibration. A
  policy-default change requires a fresh measured artifact and tool hash.

## Progress log

- **2026-09-03 — owner focus: 6-8 handed table is the calibration primary
  direction.** The owner stated "90% 的牌局都是 6-8 人" and asked to focus there.
  An audit of the drop-clean dataset confirms it: head-count buckets are
  {1人: 6, 2人: 31, 6-8人: 64} with **zero 3-5 handed frames** — the owner plays
  6-8 handed and never plays a 3-5 handed table. Section 10's generic head-count
  requirement {2, 3-5, 6-8} was therefore **owner-authorized** to a focused
  {2, 6-8} (new `REQUIRED_HEADCOUNT_BUCKETS` in `coverage.py`, documented inline
  and recorded here — the same pattern as the `MIN_SESSIONS=2` waiver). This is
  a recorded owner decision, not a silent relaxation: a table type the owner
  never plays must not be collected purely to satisfy a generic table, since it
  would be out-of-distribution noise rather than evidence. The real remaining
  gaps (now that occupancy is `ok`) are within the 6-8 handed bucket: the
  `completed_action` / `current_actor` fields are entirely empty (0 samples),
  the TURN street has only 3 frames, and the negative samples (card-back,
  non-pot, occlusion, menu) are near-zero. The updated top-up checklist
  (`reports/label-topup-checklist.zh-CN.md` v3) prioritises these.

- **2026-09-03 — viewpoint evidence tool (`cli viewpoint`):** the owner anchored
  the LIVE discriminator on **the three action buttons** ("以三按钮为准"), and
  this tool surfaces that evidence for a human to confirm by eye. `viewpoint.py`
  is deliberately *not* an auto-classifier (guide rules 1-2 forbid reusing
  another platform's geometry, and the failure-closed philosophy forbids guessing
  which table a frame belongs to): it extracts the explainable signals — the
  revealed-hero-cards white fraction (`hero_signal`), the coloured three-button
  action-band fraction (`_action_signal`), and a caller-supplied `hero_occupied`
  — then returns a conservative `LIVE`/`SPECTATE`/`UNKNOWN` verdict with a
  confidence and a `ViewpointEvidence` breakdown. `hero_occupied` is *never*
  derived here (a second heuristic would break the fail-closed contract); it is
  supplied by the seat-reader pipeline / the labeller and only corroborates.
  `cli viewpoint` renders a self-contained HTML contact sheet in the private
  dataset's `reports/` (images inlined; never in Git) plus an optional `--json-out`
  machine-readable verdicts file. On the 101-frame drop-clean dataset it reports
  session_001 all-LIVE (the owner's real heads-up game) and session_002
  47 LIVE / 17 UNKNOWN (the UNKNOWN frames are mid-street frames where the
  buttons were not lit — exactly the frames the owner should eyeball).   18 new
  unit tests; flake8-clean project-wide.

- **2026-09-03 — stack-value transcription tool (`cli stack-worksheet` /
  `cli stack-apply`):** with occupancy now `ok`, the remaining blocker for a
  "stable positive" frame is an `OCCUPIED` seat whose `stack` is still
  `UNKNOWN` — which de-qualifies the whole frame and leaves calibration/
  validation splits empty. `stack_transcribe.py` turns every such target into a
  form the labeller fills by eye (a zoomed crop of that seat's stack pill plus
  the frame context), and applies only the values the labeller actually returns.
  It **renders, never writes**: `stack-worksheet` emits a self-contained HTML
  (images inlined, never in Git) + a `frame,slot_id,value` CSV template;
  `stack-apply` is the sole writer and keeps a timestamped backup of
  `frames.jsonl` first. Philosophy (guide rule): a blank/unknown/already-set
  cell is never auto-filled, `CONFLICT` is never transcribed (it needs a
  re-read, not a guess), and a non-empty value must validate as a non-negative
  int before promotion to `VALID`. Geometry comes from `seat_reader`'s
  `SLOT_LAYOUT_MULTI` / `SLOT_LAYOUT_S002` per session — it stays in normalized
  canvas space and is not reused from any other platform. `cli review-frames`
  gained a `--session` filter so the labeller can work the primary 6-8 handed
  bucket without a 100+-frame dump. 14 new unit tests; flake8-clean.

- **2026-09-03 — data-trust audit: dropped spectate-segment frames from
  session_001:** the owner flagged that the capture may have mixed in frames of
  a table he was *watching*, not playing, which would corrupt every hero-relative
  field (hero_cards, current_actor, completed_action). A frame-by-frame review of
  the private dataset confirmed it: session_001's first segment (t < 62300 ms,
  `session_001_hand_0000/0001`, 5 frames) was the 8-handed table of an opponent
  ("泰迪小白" et al.) that the owner was spectating, before he entered his own
  heads-up table. Note the room-label line `<不要不要不要01的牌局>` is a *room
  name*, NOT a spectate badge — the reliable LIVE indicator is the owner's own
  nickname + revealed hole cards + action buttons at the bottom. session_001's
  later segment (heads-up, owner = "鱼而已不要") and all of session_002 (8-handed,
  owner playing) are genuine live play. Per owner decision the 5 spectate frames
  were dropped from `labels/frames.jsonl` (106 -> 101; backup kept as
  `labels/frames.jsonl.pre_spectate_drop.bak`), and the coverage / top-up
  checklist was regenerated. `hero_cards` was already UNKNOWN on those frames
  (correct fail-closed), but their board_cards / street / pot / occupancy came
  from a table the owner was *not* playing, so dropping them matters.

- **2026-09-03 — label top-up review page (`cli review-frames`):** stage F
  label coverage is currently the bottleneck (105 of 106 frames carry at least
  one UNKNOWN field), and a terminal `coverage`/`splits` run reports *how many*
  samples are missing but never *which pixel to look at*. Added
  `tools/capture_card_calibration/review_frames.py`, which turns an audit
  report plus the label set into a per-frame labeller-facing HTML page: each
  card embeds the normalized PNG, the audit findings that point at that frame
  (if any), and a slot-by-slot read-out marking which fields are still
  UNKNOWN/CONFLICT. The output is self-contained (images inlined as base64) so
  it opens without the private data dir, and it lives under the private
  dataset's `reports/` — never Git. It never rewrites a label or invents a
  value, honouring the failure-closed philosophy. Wired it into `cli.py`
  (`review-frames --root/--out/--json-out/--limit/--rules/--include-images`)
  and unit-tested it with synthetic-only
  `tests/tools/test_capture_card_review_frames.py` (25 tests: field rendering,
  gap detection, slot views, per-frame issue indexing, HTML/JSON emission,
  escaping, self-containedness). Module stays import-safe without OpenCV. Ran
  it over the real 106-frame set (123 MB with embedded images, 380 KB in
  `--include-images` off mode); the per-frame gap distribution matches the
  top-up checklist exactly (board_cards 27 / hero_cards 6 / pot 1 frame-level;
  stack 461 / dealer 232 / completed_action+current_actor 848 slot-level).

- **2026-09-03 — stage-H splits run against the real label set is BLOCKED
  by stage-F coverage, not a tool defect:** ran `cli splits` (and `cli
  coverage`) over the private 106-frame label set. `splits` assigns frames by
  hand group and reports `train/validation have no stable positive frames`
  because 105 of 106 frames carry at least one known-but-UNKNOWN field: stack
  is UNKNOWN on 461 slot observations, dealer on 232, board_cards 27,
  hero_cards 6. Diagnosed against `_has_unknown_field`: even under a generous
  rule that lets an EMPTY slot keep an UNKNOWN stack, only 14 of 106 frames
  qualify as fully-valid, so the read is legitimately fail-closed — the 2-5
  and 6-8 seat frames and completed-action/temporal/anomaly samples still must
  be transcribed before a usable train/validation split exists. Coverage shows
  the exact gaps (completed_action 0, hero_actor 0, anomaly 0, board/pot/dealer
  negative samples short, head-count 3-5 never observed). A per-field top-up
  checklist was written to the private dataset
  (`reports/label-topup-checklist.zh-CN.md`), but no label or split value was
  invented and no coverage requirement was relaxed. This is a PARTIAL/BLOCKED
  honest state, not a claim that calibration is complete.

- **2026-09-04 — video-mined label top-up: stage-G gaps closed 8 -> 2:**
  mined both raw session videos (~16 min total) at 10 fps with a private
  scene miner, then promoted only visually-verified evidence into the private
  dataset (216 -> 355 labelled frames; every single frame eyeballed on a
  contact sheet, no auto-proposal landed unreviewed). Landed: 61 verified
  completed-action labels (CALL 18 / RAISE 21 / BET 4 / FOLD 18 across
  slots 0-7), 55 `current_actor=HERO` frames (blue "N 跟注" decision circle
  is the reliable hero-turn cue; the grey 让或弃/自动让牌 toggles are always
  on and are NOT a turn signal), 50 MENU + 2 SIGNAL_LOSS + 1 OVERLAY anomaly
  scenes, 30 deal-transition and 70 action-transition temporal groups.
  Coverage: hero_actor, anomaly_scenes, all field negatives, and temporal
  deal/action/street_change are now `ok`. Honest remaining gaps (nothing in
  the current footage can close them): CHECK 0/10 and ALL_IN 0/6 badges never
  rendered (raise-heavy low-stakes play), BET 4/10, hand_end RESULT screens
  not identified (golden burst is a win effect, semantics unconfirmed),
  reconnect groups 2/5 (signal was stable throughout). Action-badge vocabulary
  measured on this platform: CALL=blue, RAISE/BET=orange (text tells them
  apart), FOLD=dimmed avatar + white text, countdown=white "Ns"; 等待审核/
  等待中 is NOT an action. Private checklist regenerated as
  `reports/label-topup-checklist.zh-CN.md` v4.

- **2026-09-03 — capture-card 5-slot board geometry landed; card-template
  pilot (honest negative):** mined the two raw session videos (~16 min) with a
  private scene miner (board-strip blob counting, seat avatar dark/white text
  events, table-vs-anomaly gating), yielding 83 RIVER candidates whose 5 card
  boxes are stable to ±1 px across both sessions. Widened the platform
  `board_cards` ROI to the measured full 5-card strip (110,478)-(388,556) and
  landed `configs/vision/wepoker_android_capture_card/board_slot_layout.json`
  (5 measured slots, relative to the strip); evidence rows appended to the
  private `labels/roi_measurements.csv`. Card-recognition pilot against the
  owner's labelled frames: H5 (`wepoker`) corner-glyph templates FAIL
  verification at capture-card scale (70.5% hero / 90.5% board, black-suit
  S/C collapse with high raw scores — no abstain threshold can gate them), so
  `template_source: wepoker` must not be relied on for suits. Platform-derived
  medoid templates from labelled crops reach 89.6% hold-out with residual
  6/8 and S/C confusion; 10-frame temporal fusion within one street fixes
  rank 6/5 reads (measured) but needs street-boundary gating. Card
  calibration therefore stays UNCALIBRATED (fail-closed); next step is formal
  stage-I template/threshold work on street-gated fused crops. Mining also
  produced 621 action-event / 569 button / 52 anomaly / 24 transition
  candidates ready for owner-reviewed label top-up.

- **2026-09-03 — capture-card boundary measurement, seat reader, and
  two-session floor:** added `tools/capture_card_calibration/boundary.py`
  (stage C section-6 content-boundary drift measurement) and its
  `cli boundary` subcommand, measuring canvas geometry separately from content
  luminance so a leftover UVC letterbox border is caught even while the game
  draws a dark menu band; only stable table frames decide the verdict against
  the guide's 2-pixel tolerance. Added `tools/capture_card_calibration/seat_reader.py`
  (stage F seat pixel reader) reading per-visual-slot occupancy / stack /
  dealer from normalized frames via luminance/chroma thresholds, giving VALID or
  UNKNOWN per read and reusing no LDPlayer or H5 ROI. Recorded the
  owner-authorized waiver of the third capture session as an explicit
  annotation (`MIN_SESSIONS = 2` in `schema.py`, surfaced in `report.py`), with
  the deliberate, documented rationale. Resolved the capture-card identity
  placeholder by replacing the `card_replace_me` platform config / layout_id
  with the real ugreen UVC card and updating `hero_slot_layout.json` and the
  `land_capture_card_configs.py` default. Unit-tested via synthetic-only
  `tests/tools/test_capture_card_boundary.py`; capture-card tool modules and
  tests are flake8-clean and pass with `PYTHONPATH=src`. This still calibrates
  nothing on its own: validation of the seat reader's measured geometry and the
  remaining negative-sample / temporal / action / anomaly gaps below require
  real capture-card evidence.

- **2026-09-03 — capture-card calibration toolchain:** added
  `tools/capture_card_calibration/`, the hardware-independent half of the
  calibration guide. It provides: SHA-256/SHA256SUMS hashing (guide rule 8),
  a strict label/field schema that forces UNKNOWN (a field never carries a
  guessed value) and intercepts REPLACE_ME placeholders, deterministic
  `layout_id` construction (§6), the dataset delivery skeleton (§3) with
  `frames.jsonl` / `roi_measurements.csv` I/O, pixel-ROI -> normalized
  coordinate geometry that emits draft `table_map` + slot layouts, minimum
  coverage checks with an exact top-up list (§10), session+hand-isolated
  splits with leakage detection (§11), an acceptance report that defaults to
  PARTIAL/BLOCKED (§17), and a `cli.py` entrypoint. It also ships the
  stage A/B live-recording helpers (`probe` to read a device's negotiated
  UVC parameters, `record` to capture a session to `source/raw/*.mkv` while
  automatically logging disconnect / black-frame / reconnect signal events).
  Unit-tested (see `tests/tools/test_capture_card_*.py`). This does NOT
  calibrate anything — stages A (freeze hardware) and B (record 45–90 min of
  real hands) still require real hardware and a human operator.

- **2026-09-03 — capture-card realtime backend:** added the UVC capture
  backend `CaptureCardBackend` (`perceptual/capture/capture_card_backend.py`)
  implementing the `CaptureService` contract via OpenCV `VideoCapture`
  (MSMF/DirectShow, YUY2 fourcc, resolution/fps, disconnect + all-black
  signal-loss detection), plus stage-C frame normalization
  (`perceptual/capture/normalization.py`: fixed rotate -> mirror -> crop ->
  content-size validation, versioned). Both are unit-tested with a mocked
  VideoCapture. Added `configs/vision/wepoker_android_capture_card/` as an
  explicit `uncalibrated` platform scaffold. This is the realtime backend only:
  no capture-card recognition calibration exists, so the platform's fields must
  read UNKNOWN until `docs/capture-card-calibration-guide.zh-CN.md` stages A-K
  produce real capture-card evidence and stage L lands it. No end-to-end
  capture-card capability is claimed.

- **2026-09-02 — capture-card calibration handoff:** added a Chinese,
  AI-executable end-to-end specification for independently calibrating a real
  Android phone connected through a USB capture card. It freezes source and
  normalization metadata, visual-slot numbering, ROI/label contracts,
  field-by-field minimum coverage, hand/session-isolated splits, threshold and
  locked-validation rules, Replay/hash evidence, privacy constraints, failure
  conditions, and final deliverables. Linked it from the capture/mapping
  document. This is a calibration handoff only: no UVC capture backend or
  released capture-card capability is claimed.

- **2026-08-24 — Android calibration desktop packages:** committed the
  completed LDPlayer Android table calibration as `c3e9d59`, temporarily used
  public standard GitHub-hosted runners for Build Desktop App run `#47`, then
  restored the repository to private. The native Windows installer and macOS
  DMG both built successfully; the downloaded DMG checksum verified, and its
  bundled app passed strict code-signature, Gatekeeper Notarized Developer ID,
  and stapler validation. The Windows installer is structurally a valid PE32
  executable; live Windows LDPlayer installation and recognition remain
  tester-owned release evidence.

- **2026-08-24 — Android seat/actor/canonical-action closure:** added an
  independent eight-slot occupancy observation contract, serializer,
  confidence gate, temporal confirmation, and empty-plus recognizer. Thirty-
  four manually reviewed stable states produced 272/272 correct occupied or
  empty labels with no conflict. Added a versioned eight-slot Android mapping
  that derives canonical 2–8-player positions from Dealer and occupancy,
  promotes mapped stacks, and keeps unknown slots fail closed. A separately
  measured Hero-turn recognizer accepted all 33 reviewed blue-control frames
  and abstained on the other 201 private frames; it never guesses an opponent
  timer. Completed action glyphs now choose their own actor slot, persistently
  rendered glyphs are deduplicated, and an event is recorded only when stack
  delta and pot evidence reconstruct a legal chip amount. Canonical action
  history now reaches `LiveStrategySession`, whose dealt-player count follows
  active occupancy; without a qualified multiplayer Provider, Advice remains
  ABSTAIN. Updated the English/Chinese product, vision, architecture, and
  handoff documents. The full suite passed 1,806 tests with 3 platform skips
  (1,809 collected); repository-wide Flake8 and diff checks passed. The local
  v0.1.11 macOS bundle built successfully, passed strict code-signature
  verification, and contains one copy of each new mapping/template resource.
  Authorized privacy-reviewed raw Replay, live Windows LDPlayer/reconnect and
  Windows packaging remain external release evidence; opponent current actor
  and complex missed-action/side-pot sequences remain UNKNOWN until measured.

- **2026-08-24 — Android completed-action label calibration:** added eight
  Android-only action ROIs and privacy-safe binary glyph masks for Fold,
  Check, Call, Bet, Raise, and All-in. Forty-five timeline- and slot-spread
  labels were manually reviewed and all 45 matched the correct action/slot;
  600 labels cleared the measured 0.83 floor across all 234 frames with no
  ambiguous runner-up. The 48 strongest rejected candidates were Hero action
  controls, nicknames, avatars, cards, or overlays; their maximum score was
  0.762. Production now emits completed `slot_actions[]` through independent
  calibration, confidence gating, and two-frame temporal confirmation.
  Persistent Fold labels remain observations rather than duplicate events.
  The full suite passed 1,796 tests with 3 platform skips (1,799 collected);
  repository-wide Flake8 and diff checks passed. Seat occupancy, canonical
  mapping, actor, action amounts, raw Replay, live Windows LDPlayer, and
  packaging remain; strategy therefore still fails closed to ABSTAIN.

- **2026-08-24 — Android visual-slot stacks and Dealer calibration:** added
  eight Android-only stack ROIs with independent glyph templates/calibration,
  plus eight Dealer search windows whose output is explicitly a visual slot.
  Eighty timeline-spread stack crops (ten per slot) were manually reviewed and
  all decoded correctly; 68 empty/overlay/transition negatives abstained.
  Forty Dealer detections (five per slot) were reviewed and all mapped to the
  correct slot; across all 234 raw ADB frames, 216 produced exactly one valid
  Dealer slot and 18 hidden/transition states abstained. Added glyph-topology
  filtering after real stack crops exposed an 8-versus-3 correlation error.
  Focused production-profile tests passed; the full suite passed 1,795 tests
  with 3 platform skips, and repository-wide Flake8 and diff checks passed.
  Seat occupancy, canonical mapping, actor/action amounts, raw Replay, live Windows
  LDPlayer, and packaging remain.

- **2026-08-24 — Android board/street production calibration:** added an
  Android-only five-slot board ROI/layout for the 1440x2560 LDPlayer canvas
  and independently calibrated occupancy/street evidence from 27 manually
  reviewed stable raw-ADB states: 17 distinct postflop boards containing 64
  visible card identities plus 10 preflop empty-board states. All stable
  states read correctly; measured deal/flip transitions top out below the
  accepted occupancy floor and remain UNKNOWN/CONFLICT. The separate
  88-minute H.264 recording validated the same geometry after its 48-pixel
  toolbar crop but did not lower raw-ADB thresholds. Added a private video
  normalization/dedup manifest tool and synthetic production-profile tests.
  Also calibrated the fixed total-pot ROI and white-on-dark amount OCR: 22/22
  distinct manually transcribed values and all 53 labeled stable frames were
  correct, while 25 label/menu/overlay/transition negatives abstained. The
  full suite passed 1,793 tests with 3 platform skips; repository-wide Flake8
  and diff checks passed. Seat occupancy/canonical mapping, actor/actions,
  authorized raw-frame Replay, live Windows LDPlayer, and packaging remain.

- **2026-08-24 — 88-minute LDPlayer video intake:** audited a private
  1.9GB H.264 recording (88m35s, 30fps, 1440x2608) without adding it or
  extracted frames to Git. The extra 48 vertical pixels are a stable LDPlayer
  host toolbar; cropping `(0, 48, 1440, 2560)` restores the calibrated Android
  game canvas, so existing Android ROIs remain geometrically applicable.
  A full-decode one-minute sample produced 89 frames spanning changing seat
  occupancy, all streets, all-ins, showdowns/results, and ranking/profile
  overlays. The production hero recognizer returned 62 VALID, 27 UNKNOWN,
  and zero CONFLICT across 60 distinct accepted hands; visual review found
  the UNKNOWN cases concentrated in card backs, folded/dimmed cards, and
  overlays, with no confirmed false VALID in the sample. H.264 compression
  makes this strong temporal/negative/calibration evidence but not a basis
  for lowering thresholds without raw ADB PNG ground truth. No product code
  or calibration changed.

- **2026-08-23 — Android-first README:** clarified the English and Simplified
  Chinese user guides so LDPlayer Android/ADB is the only default live input:
  no H5 or Chrome window is required, the 1440×2560 portrait prerequisite is
  explicit, and single-device `auto` versus multi-device serial selection is
  documented. This remains a `main` capability until the next Windows
  installer is packaged and released.

- **2026-08-23 — application icon:** added a project-owned PokerSense icon
  (PNG source plus macOS `.icns` and Windows `.ico`) and made the PyInstaller
  spec select the native format for each platform. The local macOS bundle
  contains `PokerSense.icns` and passed `codesign --verify --deep --strict`.
  Windows package validation remains CI-owned.

- **2026-08-23 — strategy UI and cross-platform CI repair:** redesigned the
  desktop Strategy Advice panel around primary action, action frequencies,
  sizes, EV, source, safety reasons, and expandable evidence; non-READY
  output remains explicitly withheld. Added repository-wide LF checkout and
  UTF-8 CI defaults so strategy fixtures, documentation, UI contract tests,
  and benchmark hashes do not vary on Windows; fixed the merged verifier's
  lint issue. Full pytest, flake8 (including `tools`), and local browser
  rendering against a READY strategy fixture passed. The Windows GitHub run
  must still confirm the platform-specific repair.

- **2026-08-23 — multiplayer strategy merge:** merged
  `codex/multiplayer-strategy-system` into `main` after retaining the Android
  ADB capture path. The live stream now emits atomic `DesktopFrame` values and
  routes incomplete Android observations through the strategy safety gates,
  which fail closed to `ABSTAIN` until actor, stack, action, and a qualified
  provider are calibrated. Resolved capture/UI integration and documentation
  conflicts; full test suite, flake8, and staged diff checks passed. Live
  LDPlayer, real capture Replay, interactive UI, package, and clean-install
  evidence remain outstanding.

- **2026-08-23 — Android/LDPlayer production pivot:** replaced the default H5
  window path with explicit ADB device capture for portrait frames. Added
  fail-closed device selection, capture error/PNG validation, Android-specific
  hero calibration, and retained shared card-art templates without sharing ROI
  evidence. On 66 private deduplicated frames, all 58 visible hands read
  correctly and all eight negative scenes abstained. Full tests and flake8
  passed; live Windows LDPlayer and package verification remain pending.

- **2026-08-23 — Android dataset audit:** inspected 234 private LDPlayer
  captures without retaining them. The added temporal triplets cover deal
  transitions, folds, results, menus, multiplayer/showdown geometry, and
  negative scenes; missing ground truth deliberately leaves thresholds and
  calibration counts unchanged.

- **2026-08-23 — development regression and performance report:** recorded a
  commit-bound Python 3.13/M1 Pro report for the multiplayer strategy branch.
  The strategy suite passed 1,080 tests in 4.10s; the available full suite
  passed 1,766 with 3 Windows-only skips in 23.55s. The isolated Quartz file
  had 6 passes, 7 TCC-precondition failures, and one real-capture skip, which
  remains an environment blocker rather than being relabeled as passing.
  Five-repeat Adaptive Equity measurements reported p95 1,065.064ms/7.436
  outcomes-per-ms for exact HU, 1,373.229ms/7.282 trials-per-ms for HU MC,
  and 2,006.896ms/4.983 trials-per-ms for 3-way MC; the conservative 3/2
  operations-per-ms policy remains unchanged. Flake8, 298-fixture, JavaScript,
  and diff checks passed. Real WPK Replay, stable-state-to-render p95, UI
  interaction, soak, package, and clean-install evidence remain missing.

- **2026-08-23 — partner-facing recognition/UI handoff:** added a standalone
  implementation handoff for the parallel WPK recognition and Live Coach UI
  work. It inventories the strategy/state capabilities already present, gives
  prioritized recognition and UI work tables, defines the `RawObservation`,
  `PlatformSeatMapping`, and `DesktopFrame` boundaries, identifies strategy-
  owned gaps that the partner must not guess around, and supplies development,
  Replay, UI-sequence, and shared R6 acceptance checklists. Documentation link
  and diff checks passed; this changes no released behavior. Real WPK Replay
  and interactive UI acceptance remain outstanding.

- **2026-08-23 — recognition/UI integration handoff and branch packaging:**
  documented the production boundary between the multiplayer strategy/state
  work and the parallel WPK recognition/Live Coach UI work. The PRD now lists
  required per-field `RawObservation` inputs, ownership of the live 2-max and
  per-size-frequency gaps, Advice display rules, Replay evidence, and a shared
  R6 merge gate; architecture and regression documents link the same frozen
  boundaries. On Python 3.13, 1,766 tests passed and 3 skipped with the Quartz
  permission-dependent file excluded; all 7 tests in that file were collected
  but blocked by the current terminal's Screen Recording TCC gate before their
  mocked window cases. Full Flake8, 298-fixture regeneration check, JavaScript
  syntax, and diff checks passed. Real WPK raw-frame Replay and interactive UI
  acceptance remain required.

- **2026-08-23 — bounded local GTOpen Slow Provider:** added an optional
  loopback-only `GTOpenPreflopProvider` against the separately checked-out
  upstream service. It serializes GTOpen's single mutable preflop session,
  maps exact 2–9-player position/blind/ante/rake/equal-stack contexts and
  authoritative actor/kind/raise-to histories, enforces action-line parity,
  waits for a bounded model gap, then reads the Hero's exact 169-class slice
  while preserving every raise/all-in size. Unequal stacks, imprecise paths,
  illegal actions, transport errors, expiry, timeout, and poor convergence
  fail closed; timeout attempts to stop the upstream solve. Multiway output is
  always HEURISTIC and discloses product-equity/realization, non-unique
  equilibrium, unverified remote revision, and license limits. A real M1 Pro
  CPU-only E2E at upstream `4aee435` built 4,270 nodes/1,710 action nodes/
  5.771688 MB and returned AKo Fold/Call/2-2.5-3BB Raise/All-in frequencies at
  100 iterations with model gap 0.008207490846030292 BB. Twenty-one focused
  Adapter tests, all 1,080 strategy tests, and all 1,522 available non-OpenCV
  tests passed; the generated corpus is current at 298 fixtures and focused
  Ruff, byte compilation, JavaScript syntax, and diff checks passed. Missing
  upstream license, independent Golden parity, real WPK input mapping, and
  multiway postflop solving still block registration as a released Provider.

- **2026-08-23 — local GTOpen execution probe:** cloned upstream commit
  `4aee435bdeb155b25f0c8140e707a8342ce4356f` into the Git-ignored
  `.upstream/GTOpen/` research checkout, without copying source or generated
  strategy assets into PokerSense. The Apple M1 Pro CPU-only Release build
  succeeded and all 104 executed upstream Solver tests passed (one benchmark
  ignored). A real local API probe built a 3-player 20BB BTN/SB/BB tree with
  13 nodes/6 action nodes/0.016224 MB, converged by its first 25-iteration
  check to a reported total model gap of 0.0004571471 BB, and returned a root
  2x169 strategy array. This proves local executability and API shape only;
  missing upstream licensing, independent Golden parity, real WPK context
  mapping, deeper performance, and multiway-postflop limitations still block
  registration or release as a PokerSense Provider.

- **2026-08-23 — multiplayer Provider source re-audit:** rechecked current
  public solver candidates after the generic PRV-003 intake path was ready.
  GTOpen now exposes a promising 2–9-player Preflop Lab API with per-node
  169-class strategy arrays and per-player best-response gaps, but its root
  repository has no LICENSE file and the raw LICENSE URL returns 404. Its own
  documentation also limits multiway terminal values to an equity/product
  approximation and postflop solving to Heads-Up. It was therefore documented
  as a priority candidate only if a compatible license and independent Golden
  validation appear; no code or asset was copied and no capability was
  claimed. MIT DCFR-SOLVER remains 6-max-only and cannot satisfy the 3–9-player
  or multiway-postflop requirements. PRV-003~005 still require an externally
  licensed source or user-supplied licensed export.

- **2026-08-23 — integrity-checked raw-frame Capture Replay contract:**
  advanced the real-evidence side of `ST-002` with strict Replay v1 loading,
  execution, and a deterministic JSON-safe quality report. Artifact, platform
  config, per-field calibration, and every raw frame are SHA-256 pinned;
  references cannot escape an explicit asset root, and their JSON must restate
  matching platform/layout/field/sample metadata. Release eligibility requires
  real-capture stage, authorization, privacy review, production recognizer
  execution, frame/revision-bound field evidence, non-empty calibration for
  every used field, and exact per-frame status/version/event/reason parity.
  Stable-observation or Synthetic Replay can never satisfy R6. Thirty-two
  focused tests cover tampering, drift, missing evidence, strict schema,
  recognizer identity, and eligibility. All 1,059 strategy tests and all 1,498
  available non-OpenCV tests passed; fixture, JavaScript syntax, diff, and
  focused Ruff checks passed. No real WePoker stack/action/actor/dealer raw
  frames were added, so ST-002 and R6 remain partial rather than overstated.

- **2026-08-23 — explicit platform slot-to-seat candidate mapping:** advanced
  `ST-002` with an immutable, versioned `PlatformSeatMapping` and a
  fail-closed `PlatformMappedStateEngine`. Stable actor/action/stack/pot/dealer
  evidence now maps visual geometry to canonical seats, builds one-player
  candidate deltas, reuses the production action reconciler, and atomically
  persists only exact transitions. Missing/unmapped/conflicting slots,
  multi-player stack changes, mixed action/street/card frames, chip mismatch,
  low-confidence values, and forced postings expose no candidate or event.
  Thirty-eight focused tests include parameterized 2–9-player coverage, and
  23 executable `MOCK-PLATFORM-MAPPING-*` Synthetic Replay cases grew the
  corpus to 297. All 1,027 strategy tests and all 1,466 available non-OpenCV
  tests passed; fixture, JavaScript syntax, diff, and focused Ruff checks
  passed. `ST-002` remains partial because WePoker stack/action/actor/dealer
  ROI/slot calibration and authorized real capture Replay do not yet exist;
  the production live profile therefore remains fail-closed.

- **2026-08-22 — licensed strategy-asset intake contract:** partially advanced
  `PRV-003~005` with a read-only `JsonStrategyAssetProvider` for 3–9-player
  preflop and multiway/presolved postflop nodes. Registration verifies file
  SHA-256, schema, provider/source/license metadata, capability ID, and the
  full capability digest; lookup uses the canonical context digest and fails
  closed for missing or malformed nodes. Synthetic tests cover 3-player
  preflop, 3-way flop, 4-way turn, sizing/EV, Router→Advice, bad hash,
  capability mismatch, and damaged nodes. Six executable
  `MOCK-STRATEGY-ASSET-*` fixtures grew the corpus to 274. All 988 strategy
  tests and all 1,427 available non-OpenCV tests passed; fixture, JavaScript
  syntax, diff, and focused Ruff checks passed. Real licensed multiplayer and
  presolved assets plus Golden parity remain required before these Provider
  requirements can be completed or released.

- **2026-08-22 — ordered Fast-source fallback:** completed `RTR-005` with a
  capability-safe `TieredStrategyRouter` that queries Cache, Preflop DB,
  Presolved, then Model and stops after the first usable layer. Added a
  lookup-only cache Provider so a write-through DB/asset wrapper can populate
  the same canonical entry without hiding fallback behavior. Miss,
  not-applicable, and rejected results fall through; all-source failure keeps
  the lookup trail and never invents a candidate. Five executable
  `MOCK-FAST-FALLBACK-*` fixtures grew the corpus to 268. All 967 strategy
  tests and all 1,406 available non-OpenCV tests passed; fixture, JavaScript
  syntax, diff, and focused Ruff checks passed.

- **2026-08-22 — target-hardware adaptive Equity calibration:** completed
  `EQ-004` on the declared MacBookPro18,3 target (Apple M1 Pro 10-core, 32GB,
  Python 3.12.2). Five measured runs found exact p95 throughput of 7.186
  outcomes/ms and worst-case measured MC p95 throughput of 4.757 trials/ms;
  the versioned `adaptive-equity-v2-m1-pro` defaults now use conservative 3/2
  rates. The calibration JSON records environment, command, source revision,
  latencies, and benchmark-tool SHA-256; tests lock the hash, rates, 50% safety
  margin, and 300ms PARTIAL behavior. All 953 strategy tests and all 1,392
  available non-OpenCV tests passed; fixture, JavaScript syntax, diff, and
  focused Ruff checks passed.

- **2026-08-22 — auditable hard refusal gates:** completed `FUS-004` with
  structured PASS/FAIL/SKIPPED results for request freshness, confidence,
  decision context, strategy availability, and legal actions, plus uniquely
  named external gates for range, numerical, and future modules. A failed gate
  cannot coexist with READY Advice; failure reasons flow through Fusion,
  Orchestrator, serialization, desktop view, and UI, while stale conversion
  preserves the audit. Four executable `MOCK-HARD-GATE-*` fixtures grew the
  corpus to 263. All 950 strategy tests and all 1,389 available non-OpenCV
  tests passed; fixture, JavaScript syntax, diff, and focused Ruff checks
  passed.

- **2026-08-22 — transparent bounded strategy matching:** completed `RTR-004`
  with exact hero-position matching and explicit, bounded stack, pot, and
  last-aggressive-size interpolation. Every interpolated candidate now carries
  structured requested/matched/distance/maximum dimensions; candidates that
  omit dimensions or overstate a dimension/capability score fail closed. The
  local resolver protocol, strategy cache, exploit adjustment, Advice wire,
  and UI preserve those dimensions. Four executable `MOCK-ABSTRACTION-*`
  fixtures grew the corpus to 259. All 932 strategy tests and all 1,371
  available non-OpenCV tests passed; fixture, JavaScript syntax, diff, and
  focused Ruff checks passed.

- **2026-08-22 — atomic HandMemory transitions:** completed `MEM-001` by
  replacing Orchestrator's state-then-events writes with prevalidated atomic
  `record_transition`, and hand-boundary's event/complete/start sequence with
  atomic `replace_active_hand`. Invalid event identity/version, non-HAND_END
  boundaries, time errors, and existing successors leave states, events,
  histories, and active-hand identity unchanged. Added 8 direct rollback and
  commit tests plus 4 executable `MOCK-MEMORY-*` fixtures, growing the corpus
  to 255. All 916 strategy tests, all 1,355 available non-OpenCV tests, and
  all 75 memory/orchestrator/integration tests passed; fixture check and
  focused Ruff passed.

- **2026-08-22 — production live Advice binding:** connected the real desktop
  stream through `LiveStrategySession`, `StrategyOrchestrator`, and atomic
  `DesktopFrame` output. Advice is bound to hand/state/request plus the current
  perception-quality fingerprint; state changes, expiry, history changes, or
  same-state confidence degradation force a new request and prevent prior
  READY actions from leaking forward. The current WePoker capture remains
  fail-closed ABSTAIN because actor/stacks/actions are uncalibrated and no
  bundled HU strategy asset is claimed. Added 6 focused production-binding
  tests. All 912 strategy tests and all 1,344 available non-OpenCV tests
  passed; fixture check and focused Ruff passed.

- **2026-08-22 — deterministic hand-boundary orchestration:** completed
  `ST-004` with a fail-closed SAME_HAND/CONFIRMED/AMBIGUOUS detector. Hero-card
  changes and corroborated street/board/pot resets can start a new hand;
  dealer and stack resets count only through explicit slot-to-seat mappings.
  The realtime pipeline now closes the previous hand with a timestamped
  `HAND_END` before creating the successor, while weak/conflicting evidence
  cannot switch hands. Added 12 focused tests, including a no-OpenCV pipeline
  integration, and 6 executable `MOCK-HAND-BOUNDARY-*` cases, growing the
  corpus to 251. All 906 strategy tests and all 1,338 available non-OpenCV
  tests passed; fixture regeneration/check and focused Ruff passed. Real
  platform mapping and capture Replay remain under `ST-002`.

- **2026-08-22 — general temporal consensus:** completed ST-001's stable-
  observation orchestration with configurable consecutive-frame confirmation
  for all base recognition fields and visual slot stacks/actions. Changed
  candidates, UNKNOWN, missing slots, and frame-sequence gaps restart the run;
  CONFLICT remains explicit and pending values become UNKNOWN before the
  StateEngine. The realtime pipeline now uses this general gate, while the
  live profile requires two frames for every future calibrated field. Added
  14 focused tests including a no-OpenCV pipeline integration and 8 executable
  `MOCK-TEMPORAL-*` sequences, growing the corpus to 245. All 894 strategy
  tests and all 1,326 available non-OpenCV tests passed. ST-002 remains partial
  until explicit platform slot-to-seat/candidate-state mapping and real replay.

- **2026-08-22 — fail-closed Advice UI view contract:** added a JSON-safe
  desktop view model and Live Coach rendering for action frequencies, sizes,
  EV, source/version, match quality, confidence, assumptions, and evidence.
  Only READY exposes actions; PARTIAL, ABSTAIN, STALE, and send-time expiry
  hide them. Advice now retains per-field Vision/manual/config/derived/inferred
  provenance, and the UI shows match/source badges with manual input explicitly
  highlighted; older schema-v1 Advice remains readable. Added deterministic
  wire-contract tests and localized status labels, and lazy-loaded realtime
  capture/pipeline exports so serialization tests do not require optional
  OpenCV. An atomic `DesktopFrame` now carries analysis plus optional Advice;
  its WebSocket contract accepts same-state Fast→Slow refinement but converts
  expired or hand/state-mismatched results to STALE before JavaScript sees
  them. All 879 strategy tests and all 1,311 available non-OpenCV tests passed.
  UI-002/UI-003 are complete; UI-001 remains partial until the live capture
  path can construct a full DecisionContext and invoke StrategyOrchestrator.

- **2026-08-22 — complete HU preflop Blueprint parity:** completed PRV-002.
  Ante and action amounts now normalize by arbitrary big-blind units, while
  capability metadata preserves real stack/ante pairs instead of inventing a
  Cartesian product. Expanded the pinned upstream Golden from 4 to 180 direct
  lookups: every 169-class root hand plus 11 action/stack/ante nodes across
  three verified shards. This exposed and fixed Decimal regrouping drift via
  deterministic 1e-26 largest-remainder allocation. Focused Adapter/Router
  tests and direct generation/upstream parity checks passed. Full regression
  passed with all 865 strategy tests and all 1,297 available non-OpenCV tests;
  assets remain optional and upstream exploitability remains unreported.

- **2026-08-22 — canonical action-event reconstruction:** completed ST-005
  with a pure adjacent-state reconciler for fold/check/call/bet/raise/all-in.
  Events carry both additional and total-street amount semantics plus chip
  evidence. Hand/version/card/player/status/chip/current-bet conflicts fail
  closed; multi-meaning all-ins are AMBIGUOUS and emit no event until an
  observed label resolves them. Added 29 focused tests and 8 executable Mock
  transitions, growing the corpus to 237 cases. Live Observation-to-candidate
  state mapping remains under ST-002. All 682 strategy tests and all 1,114
  available non-OpenCV tests passed; fixture check and focused Ruff passed.

- **2026-08-22 — automatic input provenance collection:** completed CTX-002
  with deterministic Vision/manual/config/derived/inferred adapters, stable
  canonical value digests, one resolved provenance record per field, explicit
  same-value consensus, and fail-closed cross-source conflict handling. The
  collector now feeds DecisionContext quality aggregation directly. Added 23
  focused tests and 8 executable provenance fixtures, bringing the synthetic
  corpus to 229 cases. All 652 strategy tests and all 1,084 available
  non-OpenCV tests passed; fixture regeneration/check and focused Ruff passed.

- **2026-08-22 — fail-closed local resolver process adapter:** added a
  no-shell JSON stdin/stdout protocol bound to Provider/version and exact
  hand/state/request identity. Configured timeout and request deadline are
  combined; output size, process exit, malformed JSON, invalid strategy,
  no-strategy, non-convergence, and exploitability thresholds are explicit
  states. A real subprocess test double also proves Slow Path Advice upgrade.
  Added 20 process tests and one fixture; all 625 strategy tests and all 1,057
  available non-OpenCV tests passed. PRV-007 is complete.

- **2026-08-22 — conservative whole-hand debrief aggregation:** added
  `HandReview` to pair Advice and observed actions only by exact
  hand/state/request identity, order matched decisions by observation time,
  sum only known EV losses, report completeness and the largest leak, and
  count action/size deviations. Missing, orphan, duplicate, same-state retry,
  cross-hand, and missing-EV records are never guessed into place. Added 14
  whole-hand tests; all 605 strategy tests and all 1,037 available non-OpenCV
  tests passed. TRN-002 is functionally complete; real Replay remains a
  release-acceptance evidence requirement.

- **2026-08-22 — versioned preflop range-prior selection:** added a bounded
  `PreflopRfiRangePrior` that expands only the reviewed 6/9-handed, 100BB,
  first-in-raise asset into concrete combinations, removes known-card
  collisions, normalizes exactly, and preserves source/version evidence.
  Unsupported counts, BB, stacks, action lines, and fully blocked ranges return
  `NOT_APPLICABLE/UNKNOWN` with no random fallback. Added 41 tests and two
  trace fixtures; all 591 strategy tests and all 1,023 available non-OpenCV
  tests passed. RNG-001 is complete for capability-bounded lookup behavior.

- **2026-08-22 — structured Advice evidence-chain audit:** added deterministic
  input, canonical-state, per-seat range, and Provider references with a
  SHA-256 chain ID and named missing links. Incomplete evidence remains visible
  but caps READY confidence at 0.49 and adds an explicit assumption; complete
  and legacy schema-v1 Advice round-trip safely. Added 19 tests and a broken-
  chain fixture; all 550 strategy tests and all 982 available non-OpenCV tests
  passed. ADV-003 is complete at the in-memory/serialized contract layer.

- **2026-08-22 — KL-bounded opponent adjustment and Decision Fusion:** added
  sample/quality/weight/logit/KL gates around exponential Q-value tilting.
  Weak or incomplete profile evidence returns the identical baseline; applied
  results preserve zero support and per-size conditionals, disclose metadata,
  and are downgraded to `HEURISTIC`. `DecisionFusion` now keeps the Router's
  one baseline, applies at most one adjustment, and is used by Fast and Slow
  Orchestrator Advice. Added 38 focused tests and one trace fixture; all 531
  strategy tests and all 963 available non-OpenCV tests passed. FUS-002 and
  FUS-003 are complete.

- **2026-08-22 — audited 6/9-handed RFI heuristic provider:** added a
  versioned MIT-source asset importer and bundled only the explicit 6-handed
  and 9-handed unopened ranges from `bmorrow10/preflopR` commit `aed511d`.
  The Provider always reports `HEURISTIC`, discloses source/asset hashes and
  limitations, invents neither sizing nor EV, and rejects 3–5/7–8 player and
  BB fallbacks. Thirty-one focused tests cover all 13 explicit ranges × 169
  hand classes, corruption, boundaries, and Exact-over-Heuristic routing.
  All 493 strategy tests and all 925 available non-OpenCV tests passed; a
  local wheel contains the JSON asset and full MIT notice. `PRV-006` is
  complete for this bounded fallback; a true 3–9-player solver-derived
  Provider (`PRV-003`) remains open.

- **2026-08-22 — canonical strategy cache and Provider fast path:** added a
  thread-safe TTL/LRU cache keyed by canonical decision state plus Provider,
  asset, and strategy-engine versions. Cached entries are identity-free
  templates and are materialized against the current hand/state/request, so a
  prior request identity cannot leak. Added a cache-first Provider wrapper,
  context/action payload sensitivity, source/version/asset/engine misses,
  stale re-query, no caching of misses, exact action-option preservation, and
  concurrent lookup tests. Eighteen new tests, all 462 strategy tests, and all
  894 tests in the available non-OpenCV suite passed. RTR-008 is complete;
  RTR-005 remains partial pending real DB/presolved/model layers.

- **2026-08-22 — conservative Advice confidence aggregation:** added named
  quality components and a strict minimum aggregator for input, Provider,
  state-match, range, numerical, or caller-defined factors. Advice now exposes
  factor metadata; an explicitly required but missing factor yields confidence
  zero and ABSTAIN rather than being averaged away. Approximate match scores
  also cap final confidence. Ten new tests, all 444 strategy tests, and all 876
  tests in the available non-OpenCV suite passed. FUS-005 is complete at the
  contract and Advice integration level.

- **2026-08-22 — context quality and request factory:** added conservative
  required-field provenance aggregation that uses the minimum rather than an
  average and emits hard reason codes for missing, UNKNOWN, CONFLICT,
  LOW_CONFIDENCE, below-threshold, duplicate, and state-consistency inputs.
  Integrated the policy into State→DecisionContext and added a thread-safe,
  clock-injected RequestContext factory with aware deadlines, duplicate-ID
  retries, rollback, and concurrent uniqueness. Fifteen new tests, all 434
  strategy tests, and all 866 tests in the available non-OpenCV suite passed.
  CTX-003 and CTX-004 are complete; CTX-002 remains partial until real
  observation/manual/config sources are automatically collected.

- **2026-08-22 — exact EV primitives and completeness gates:** added exact
  Decimal immediate Call EV, explicit Fold/Call/Raise branch aggregation for
  aggressive actions, and UNKNOWN results whenever a positive-probability
  continuation lacks a net value. EV gap now requires every legal action EV;
  incomplete maps cannot leak a misleading best-vs-second gap into Advice.
  Sixteen new EV/Advice tests, all 419 strategy tests, and all 851 tests in the
  available non-OpenCV suite passed. EV-001 through EV-003 are complete at the
  deterministic calculation-contract level; real Provider/solver continuation
  values remain separate evidence requirements.

- **2026-08-22 — training and deterministic explanation contracts:** added
  exact actual-action→Advice identity binding, action and size deviation,
  evidence-preserving debriefs, and EV loss only when both preferred and actual
  counterfactual EVs are present. Added deterministic Chinese/English Advice
  explanations that render the existing Decimal values, source, match, and
  confidence without changing decisions. Eight focused tests, all 403 strategy
  tests, and all 835 tests in the available non-OpenCV suite passed. EXP-001
  and TRN-001 are complete at the domain-contract level; TRN-002 remains
  partial until full-hand Replay/aggregation and real counterfactual Golden
  evidence exist.

- **2026-08-22 — Fast/Slow strategy orchestration:** added immediate Fast
  Advice with optional asynchronous Slow submission, caller-owned threaded
  Provider adaptation, explicit handles, and PENDING/APPLIED/NO_UPDATE/
  DISCARDED/FAILED collection states. Slow results are fail-closed on
  hand/state/request drift, request or candidate expiry, provider identity or
  version mismatch, malformed results, exceptions, and non-improving strategy
  quality. Thirteen focused orchestration tests, all 395 strategy tests, and
  all 827 tests in the available non-OpenCV suite passed. RTR-006 and RTR-007
  are complete at the orchestration-contract level; a real local resolver,
  persistent strategy cache, WebSocket/UI update path, and target-hardware
  latency evidence remain future work.

- **2026-08-22 — real HU preflop Provider baseline:** added an optional
  integrity-checked Adapter for `amaster97/poker_solver` 1.11.0 at commit
  `f78f1b2`, with exact HU/preflop capability boundaries, concrete-card
  to 169-class mapping, authoritative StateEvent→`c/b/r/A` history tokens,
  source action aggregation, per-size `ActionOption` frequencies and legality,
  manifest/shard provenance, and contained NOT_APPLICABLE/NOT_FOUND/REJECTED
  outcomes. Added non-invented 100BB/no-ante Golden root results for AA/AKs/72o
  plus `b300/AA`, an
  explicit upstream parity verifier, Provider→Router→Advice E2E coverage, and
  license/quality research. Direct parity against the checked-out upstream
  assets, 24 focused Provider tests, all 382 strategy tests, and all 814 tests in the
  available non-OpenCV suite passed; strategy-scope Ruff, byte compilation,
  fixture regeneration, and documentation diff checks passed. PRV-002 remains
  partial: non-zero ante capability is not yet BB-normalized and broader action
  tree Golden coverage remains; no multi-player Provider, bundled asset,
  released UI path, or exploitability claim was added.

- **2026-08-22 — equity cache and adaptive-budget baseline:** added canonical
  SHA-256 equity queries over cards, normalized concrete ranges and versions,
  pots/eligibility, method, engine version, trials, and seed; a thread-safe
  bounded TTL/LRU cache with stale state, CI metadata, provenance, and identity
  checks; seeded weighted multiway Monte Carlo with standard error/95% CI; and
  deadline-derived exact/MC selection with COMPLETE/PARTIAL numerical status.
  Thirty-two focused cache/adaptive/multiway tests, all 358 strategy tests,
  and all 790 tests in the available non-OpenCV suite passed; lint, byte
  compilation, fixture regeneration, and documentation diff checks passed.
  EQ-006 is complete at the in-memory contract level. Large concrete ranges now
  bypass joint Cartesian materialization and use seeded independent range draws
  with whole-assignment collision rejection. Monte Carlo checks a monotonic wall
  deadline in bounded intervals and reports actual rather than planned trials.
  EQ-004 remains partial until exact/MC throughput budgets are calibrated on
  target hardware.

- **2026-08-22 — exact derived strategy metrics:** added Decimal pairwise SPR
  over main/side-pot totals, exact immediate pot odds with an explicit
  no-call-cost state, and action-size normalization in BB, pot fraction, and
  raise multiplier using separate additional and total-street amounts. Zero
  bases remain unknown (`None`) instead of being represented as zero. Seven
  focused tests, all 333 strategy tests, and all 765 tests in the available
  non-OpenCV suite passed; lint, byte compilation, fixture regeneration, and
  documentation diff checks passed. The metrics are deterministic domain
  functions and are not yet populated from real recognized stacks/actions.

- **2026-08-22 — exact multiway pot-share equity:** implemented bounded exact
  evaluation over weighted collision-free joint assignments and all legal board
  runouts. Results report Hero win/tie/loss, expected share, and expected chips
  separately for every main/side pot, respecting each pot's eligible seats, plus
  total-pot equity. Tests cover a three-way partial tie, side-pot ineligibility,
  weighted assignments, turn enumeration, missing holdings, card collisions,
  and budget refusal. Seven focused tests, all 324 strategy tests, and all 756
  tests in the available non-OpenCV suite passed; lint, byte compilation,
  fixture regeneration, and documentation diff checks passed. This is an exact
  small-scenario engine, not the adaptive/Monte Carlo budget, confidence
  interval, or cache.

- **2026-08-22 — deterministic range tracker core:** implemented concrete-combo
  parsing, Hero/board blocker filtering with exact renormalization, Bayesian
  action updates with likelihood coverage and missing-data confidence
  degradation, population-prior small-sample shrinkage, and bounded multi-player
  joint assignment enumeration that excludes known-card and cross-player
  collisions. Abstract labels such as `AKs` are intentionally rejected until a
  versioned range asset expands them. Eleven focused range tests, all 317
  strategy tests, and all 749 tests in the available non-OpenCV suite passed;
  lint, byte compilation, fixture regeneration, and documentation diff checks
  passed. No initial range asset/profile event integration or multiway equity
  consumer was added yet.

- **2026-08-22 — strategy state derivation implementation:** added explicit
  legal-action amount semantics (zero/additional/total-street), deterministic
  check/bet/fold/call/raise/short-all-in generation, exact main/side-pot
  allocation with folded-player exclusion, provisional open-betting tranches,
  and settled unmatched-chip returns, plus a
  request-bound PokerState→DecisionContext builder. The builder derives active
  seats, legal actions, pots, pairwise effective stacks, strategy player count,
  missing fields, and consistency hard failures. Synthetic all-in fixtures now
  execute against the implementation with exact pot amounts, eligible seats,
  and chip conservation. The 306 strategy tests, 362 focused compatibility
  tests, and all 738 tests in the available non-OpenCV suite passed; lint, byte
  compilation, fixture regeneration, and documentation diff checks passed.
  The full suite still cannot be collected in this environment because OpenCV
  is unavailable. Real perception does not yet supply calibrated board, pot,
  stacks, actor, or actions, and no real strategy Provider or UI path was added.

- **2026-08-22 — strategy contract/router/advice implementation:** implemented
  the first target-architecture strategy slice as a separate, additive package:
  immutable 2–9-player DecisionContext inputs, provenance/quality/range
  contracts, Provider capabilities and FakeProvider, exact/approximate routing,
  strict player-count and active-count filtering, candidate legality and sizing
  normalization, versioned READY/PARTIAL/ABSTAIN/STALE Advice, request/candidate
  expiry protection, and explicit strategy schema v1 serialization. Extended
  RequestContext additively with expiry/deadline while preserving legacy v1
  deserialization. All 72 preflop player-count/action-line fixtures and all 24
  postflop street/active-count fixtures execute through the new router. The 285
  strategy tests, 341 focused compatibility tests, and 717 available non-OpenCV
  regression tests passed; Ruff/compile checks passed. No real strategy Provider,
  automatic State→Context builder, UI integration, or released behavior changed.

- **2026-08-22 — strategy mock regression corpus:** added a deterministic,
  generated v1 strategy fixture corpus with 213 Synthetic/Benchmark cases, a
  JSON schema, manifest/hash, generator, and automated coverage checks. The
  corpus covers every documented requirement and test ID, 2–9-player preflop
  action families, 2–9-way postflop streets, quality gates, stack/pot/side-pot
  boundaries, malformed payloads, Provider/Router/Fast-Slow failures, all
  Advice states, equity anchors, audit/debrief flows, and benchmark workloads.
  The 143 focused dataset tests and the available 569-test non-OpenCV suite
  passed; Ruff, byte compilation, regeneration check, and documentation diff
  checks passed. The full suite could not be collected in the available
  environment because OpenCV is not installed. Synthetic data remains
  ineligible as real Provider Golden, real capture Replay, hardware performance,
  or clean-install evidence.

- **2026-08-22 — strategy requirements and regression baseline:** reorganized
  the target design into a product requirements document, a strategy
  requirements matrix, and a separate regression test matrix. Added stable
  requirement IDs, input-processing-output gates, 2–9-player scenario
  coverage, deterministic/Golden/Replay/benchmark fixture contracts, CI test
  tiers, change-impact regression sets, release gates, and end-to-end
  traceability. Documentation checks passed; the listed target tests remain to
  be implemented alongside their functions.

- **2026-08-22 — multi-player strategy specification correction:** clarified
  that 6-max is only the first multi-player strategy-asset priority, not the
  product boundary. The post-perception router and test matrix now require
  parameterized 3–9-player preflop coverage, separate 3-way and 4-way+
  postflop coverage, exact player-count capability matching, and explicit
  internal lookup states instead of an ambiguous `miss` result. Documentation
  validation passed; implementation remains future work.

- **2026-08-22 — post-perception strategy specification:** decomposed the v0.3
  strategy path into traceable function IDs, required input and Advice output
  matrices, Provider routing rules, canonical fixtures, unit/property/golden/
  integration/performance/fault tests, and staged acceptance gates. The spec
  distinguishes state facts, range inference, Equity, Strategy, approximations,
  and refusal behavior. Documentation only; no runtime behavior changed.
- **2026-08-22 — multi-scenario/multi-player requirements:** added the product
  requirements baseline for configurable 2–9 seat state, positions, stack and
  game configurations, preflop/postflop action lines, provider capability
  routing, multiway ranges/equity, Advice provenance, refusal rules, and staged
  delivery. Heads-up remains the first demonstrable provider, not a hard-coded
  system boundary. Documentation only; no runtime behavior changed.
- **2026-08-21 — v0.3 target architecture:** replaced the frozen v0.2.1
  plan with a staged architecture for authorized, self-hosted training. The
  design adds trusted temporal perception, State/Event Engine v2, Bayesian
  range tracking, Fast/Slow strategy routing, Decision Fusion, abstention,
  and a live-to-debrief training loop. English and Chinese product docs now
  distinguish current capabilities from the M1–M5 target. The embedded-source
  draw.io SVG passed XML validation, draw.io reopen/export, and visual review;
  the full test suite and flake8 passed. Documentation only; no runtime or
  release artifact changed.
- **2026-08-21 — external project research:** added
  `docs/research-dickreuter-poker.md` after reviewing `dickreuter/Poker`.
  The research records reusable ideas for table calibration, structured
  decision inputs, strategy parameterization, and hand analysis, while
  separating them from the repository's auto-play behavior, known Monte Carlo
  test limitations, fixed-layout assumptions, and GPL-3.0 integration duties.
- **2026-08-21 — v0.1.11:** a real Windows retest showed
  Chrome exposes the table as `WePoker-H5 - Google Chrome`, while the Windows
  backend still required an exact `WePoker-H5` title. Windows matching now
  normalizes titles, accepts generic host suffixes only at explicit separator
  boundaries, supports `window_index`, and honors the existing opt-in primary
  display fallback for the full-screen WePoker calibration. It does not bind
  behavior to a Chrome-specific suffix or use arbitrary substring matching.
  Focused and full local tests, flake8, local packaging checks, macOS/Windows
  CI, both installer builds, and GitHub release upload passed. Source commit
  `f963181`; clean-user Windows retest remains pending.
- **2026-08-21 — v0.1.10:** after v0.1.9 fixed startup,
  a real Windows run exposed the next blocker: the WebView host had already
  established a non-Per-Monitor process DPI mode. `MssBackend` now uses
  Windows mixed-mode DPI and establishes Per-Monitor V2 on every capture
  worker thread before reading physical-pixel coordinates. Focused and full
  local tests, flake8, the local package checks, macOS/Windows CI (including
  Win32-specific DPI tests), both installer builds, and GitHub release upload
  passed. Source commit `94bdc47`; clean-user Windows retest remains pending.
- **2026-08-21 — v0.1.9:** fixed the Windows packaged
  app's missing `mss` dependency, wait for the local uvicorn server before
  opening the webview, and convert capture initialization/runtime failures to
  recoverable UI errors instead of closing the WebSocket. Added focused
  regression tests and a Windows workflow dependency check. Full tests and
  flake8 passed locally and on macOS/Windows CI; the local macOS bundle passed
  version, resource, signature-structure, and HTTP startup checks; both GitHub
  installers built and were published. Source commit `91de955`.
- **2026-08-21 — Windows v0.1.8 startup diagnosis:** the release workflow
  installs `.[desktop,packaging]` but omits the `perceptual` extra that provides
  `mss`.  The frozen Windows app can therefore serve its UI, but opening `/ws`
  raises an uncaught `RuntimeError` while constructing `MssBackend`; the UI
  reports a disconnect and reconnects indefinitely.  The first-launch
  `ERR_CONNECTION_REFUSED` is a separate server-readiness race because the
  webview navigates immediately after starting the server thread.  Windows has
  no macOS-style Screen Recording permission prompt.  Diagnosis was verified
  by tracing the packaging, desktop startup, WebSocket, and capture code paths;
  no product code or release artifact was changed.
- **2026-08-20 — v0.1.8:** fixed stale hero cards across hands. After two
  matching frames show a different pair, the live pipeline closes the active
  capture hand and starts a fresh hand. Regression coverage added; full test
  suite, flake8, local macOS bundle validation, and GitHub macOS/Windows
  release builds passed. Commit `e827f62` is on `main`.
- **2026-08-20 — documentation maintenance:** rewrote English and Simplified
  Chinese READMEs as concise product documentation; audited Markdown files.
  Existing design documents remain in scope and were retained.

## Open work

1. Run the new ADB path against a live Windows LDPlayer instance, measure
   capture latency/reconnect behavior, and package it only after that passes.
2. Register a privacy-reviewed, authorized raw-frame Android Replay covering
   all-in/side-pot, missed multi-action sequences, hand transitions, overlays,
   opponent current-turn evidence, and genuine failures.
3. Keep opponent current actor and every other unmeasured Android field
   `UNKNOWN`; do not infer it from the completed-action glyph.
4. Improve CI coverage so ordinary feature-branch pushes and pull requests run
   the test suite, not only `main` and release tags.
