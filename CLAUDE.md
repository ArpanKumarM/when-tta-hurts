# CLAUDE.md — Rules of engagement for this repository

This file governs how Claude Code (or any AI assistant) works in this repo.
It is binding, not a suggestion. If a request conflicts with this file, stop
and ask the user rather than proceeding.

## Project phase

Check `docs/research_plan.md` for the current phase. As of Phase 0, no
training, no dataset downloads, no dashboard, and no committing/pushing are
permitted. Do not start Phase 1 activities without explicit user approval.

## Absolute rules

1. **No test-set tuning.** Test-set predictions must never be used to choose
   hyperparameters, thresholds, augmentation policies, or model selection.
   Only train/validation data may inform design decisions. See the "test
   firewall" in `docs/experimental_protocol.md`.

2. **No invented citations, results, or statistics.** Every citation must
   come from a primary source that was actually opened and read (not a
   search snippet). Every number reported anywhere (README, docs, paper/)
   must trace to a file under `results/` produced by a script. If a claim
   cannot be verified, mark it `UNVERIFIED` — do not omit the caveat or
   round it away.

3. **No deleting failed runs from the ledger.** `results/ledger.csv` (or
   equivalent) is append-only. Failed, crashed, or abandoned runs get a row
   with their status recorded, not silent removal.

4. **No silently changing hypotheses after seeing results.** H1–H4 in
   `docs/research_plan.md` are pre-registered. If a hypothesis needs to
   change after seeing data, that is an exploratory finding, not a
   confirmatory one — label it as such and keep the original pre-registered
   version visible in the document history.

5. **No publication, README-headline, or LinkedIn/social claims without
   verified results.** Nothing may be described as "showing," "proving," or
   "demonstrating" anything until the corresponding artifact exists in
   `results/` and has passed the statistical plan in
   `docs/statistical_analysis_plan.md`.

6. **Stop and request approval before changing the experimental protocol.**
   Once `docs/experimental_protocol.md` and `configs/experiment_matrix.yaml`
   are marked `status: approved`, do not modify the confirmatory matrix,
   the validation-gated TTA algorithm, or the primary/secondary endpoints
   without flagging the change explicitly to the user and getting sign-off.

7. **No clinical claims, ever.** This project uses MedMNIST, which is
   explicitly not validated for clinical use. Nothing in this repo may
   claim or imply clinical validity, diagnostic utility, or readiness for
   deployment.

8. **No copying third-party source code** without first verifying its
   license is compatible and documenting attribution per-file. When in
   doubt, implement independently from the published method description
   instead of copying.

9. **Do not commit or push** unless the user explicitly asks for it in that
   session. Creating/editing files is fine; `git commit`/`git push` are not
   assumed default actions.

## Working style expected in this repo

- Prefer primary sources over memory or search snippets for any factual or
  citation claim.
- When uncertain, say so explicitly rather than filling gaps plausibly.
- Keep raw per-sample predictions, not just aggregate metrics, so statistics
  can be recomputed later.
- Treat the target preprint (arXiv:2604.09697) as a preprint under review,
  not settled fact — its numbers are a starting point to reproduce, not a
  ground truth to assume.
