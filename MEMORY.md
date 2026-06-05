# FIFA 2026 World Cup Prediction Project Memory

## Competition Requirements

### Predictions Needed
For EVERY match (104 total):
1. **Score**: Exact final scoreline (after 90 mins + extra time, excluding penalties).
2. **Corners**: Number of corner kicks.
3. **Yellow cards**: Number of yellow cards.
4. **Red cards**: Number of red cards.

For **Group Stage** (Matches 1-72):
- **Winning team**: `home`, `away`, or `draw`.

For **Knockout Stage** (Matches 73-104):
- **Matchup**: Both teams playing in the slot.
- **Match winner**: `home` or `away`.
- **Penalties**: Whether match goes to shootout (`True` or `False`).

### Scoring System

| Category | Condition | Points |
|---|---|---|
| Score | Exact scoreline | 25 |
| Score | Correct goal difference, wrong score | 10 |
| Score | Correct total goals, wrong score | 10 |
| Corners | Exact number | 10 |
| Corners | Off by 2 | 5 |
| Yellow cards | Exact number | 10 |
| Yellow cards | Off by 1 | 5 |
| Red cards | Exact number | 5 |
| Winning team (Group) | Correct | 40 |
| Matchup (Knockout) | Both teams correct | 20 |
| Matchup (Knockout) | One team correct | 10 |
| Match winner (Knockout) | Correct | 20 |
| Penalties (Knockout) | Correct | 5 |

### Multipliers
All points for a match are multiplied by the round factor:
- **Group Stage**: x1
- **Round of 32**: x1
- **Round of 16**: x2
- **Quarter-final**: x4
- **Semi-final**: x8
- **Third-place playoff**: x8
- **Final**: x16

## Key Dates
- **Deadline**: June 10, 2026 at 09:00 UTC (Tournament start).
