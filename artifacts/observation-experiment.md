## Evidence
| Model | Legal moves provided | `play_move` calls | Illegal calls | Invalid-move rate | `game_over` | Outcome |
|---|---|---:|---:|---:|---|---|
| DeepSeek | No | 20 | 1 | 5.00% | `true` | White wins |
| DeepSeek | Yes | 25 | 1 | 4.00% | `true` | Black wins |
| GPT-OSS | No | 25 | 2 | 8.00% | `true` | Black wins |
| GPT-OSS | Yes | 31 | 1 | 3.23% | `true` | Black wins |

## Conclusion
Providing legal moves reduced the invalid move rate for both deepseek and gpt-oss. 
Both models had more play attempts with legal moves.
But due to its single-run, it didn't suggest improved playing results as only deepseek with out legal moves won the game.