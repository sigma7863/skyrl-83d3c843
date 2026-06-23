# Minimal Reproduction — SkyRL-Agent (arXiv 2511.16108)

End-to-end run of SkyRL-Agent's core mechanism in its smallest configuration:
the **tool-centric ReAct agent loop** + **async dispatcher** +
**transition-based backend bridge**, producing multi-turn rollouts that are
scored by a **verifiable reward** (math grader).

## Configuration
| Component | Value |
|---|---|
| Agent | `ReActAgent` (tool-centric loop) |
| Tools | `["finish"]` |
| Task / verifier | `GeneralReactTask` / `naive_dapo` (math) |
| Backend bridge | `openai_server` -> local vLLM |
| Dispatcher | `async_batch` (init -> run -> eval) |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Problems | 8 |

## Results
| Metric | Value |
|---|---|
| Problems | 8 |
| Correct (reward > 0) | 8 |
| Accuracy | 1.000 |
| Mean reward | 1.000 |
| finish_tool_ratio | 1.0 |
| avg_turns (assistant) | 1.0 |

Per-rollout transcripts: `.openresearch/artifacts/trajectories.jsonl`.

**Verdict:** the pipeline ran end to end — agent rollouts were generated through
the dispatcher, transitions were converted to training data via the backend
bridge, and rewards were computed by the verifier. A non-trivial accuracy
confirms the agent loop and reward path are wired correctly.
