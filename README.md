# ECON_5200_PS2
# Problem Set 2: The Underpowered Portfolio

**ECON 5200 · Causal Machine Learning & Applied Analytics · Northeastern University**

A consulting engagement for a client that runs about 200 experiments a year. The head of product reports that about 15% of tests win, but the CFO can't find those gains in revenue. This notebook simulates the client's experiment portfolio with known true effects, audits their p < 0.05 shipping rule, and proposes an alternative.

## Key findings

All effects are in outcome-SD units, per year of about 200 experiments.

- **Real wins:** 24 experiments ship under p < 0.05, and about 62% of them are real.
- **Winner's curse:** shipped experiments claim a total of 6.70 but deliver 4.52 (BCa 95% interval [2.67, 7.47]), an overstatement of 2.18. Inflation is largest in low-powered tests.
- **Recommended rule:** p < 0.05 plus a power floor of 80% at δ = 0.30. It ships 18 experiments with fewer false ships (33% versus 38%) and less inflation (1.19 versus 1.48), at a cost of 0.53 in realised return.
- **Sequential testing:** an always-valid mSPRT holds false positives at 1.5% under continuous monitoring, against 32.9% for naive peeking.

## Files

- `Econ_5200_PS2.ipynb`: the full analysis, Phases 1 to 4.
- `src/experiment_portfolio.py`: the portfolio simulator, decision rules and evaluation.
- `src/msprt.py`: the AI-generated sequential test from Phase 4.
- `ai-appendix.md`: the full AI transcript and change log.

## Running it

Run the notebook top to bottom with *Runtime › Restart and run all*. To run the module's own checks: `python src/experiment_portfolio.py`.

## AI use

Phases 1 to 3 were completed without AI-generated code. AI was used in Phase 4 as required, and the full exchange is documented in `ai-appendix.md`.
