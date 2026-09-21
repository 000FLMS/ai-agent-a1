# Compaction Analysis
| Metric | Baseline | With compaction |
|---|---:|---:|
| Action steps | 30 | 20 |
| Compaction calls | 0 | 3 |
| Total model calls | 30 | 23 |
| Input tokens, including cached | 504,067 | 89,826 |
| Output tokens | 5,892 | 5,509 |
| **Total tokens** | **509,959** | **95,335** |
| Peak input tokens per action | 27,198 | 5,978 |
| Cached input tokens | 474,043 | 60,603 |
| Uncached input tokens | 30,024 | 29,223 |

## Trend observed

With Compaction, agent can have much less total token usage (for my test is 509,959 vs 95,335, reduced about 80%) while both tend to have similar output tokens (5,892 vs 5,509). As compaction can summarize older interactions, reducing a lot of input tokens.
However if consider KV cahce, uncached input tokens can be similar (30,024 vs 29,223)


## Trade-off
Compaction needs summarization of all history, which can lose details needed for future actions, while full history preserves all exact details.
Compaction can also disrupt previous prompt cached.

