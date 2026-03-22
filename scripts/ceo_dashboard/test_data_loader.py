"""Quick test: verify data loader works with existing training state."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import friendly_name, compute_portfolio_value, compute_safety_score
from data_loader import load_dashboard_data

data = load_dashboard_data()
print("Portfolio Value:", data["portfolio_value"])
print("Total Return:", data["total_return_pct"], "%")
print("Dollar P&L:", data["dollar_pnl"], "CAD")
print("Best Agent:", data["best_agent"])
print("vs S&P:", data["excess_label"])
print("Safety Score:", data["safety_score"])
print("Agents:", data["total_agents"], "(", data["passed_count"], "passed)")
print("Leaderboard top 5:")
for a in data["leaderboard"][:5]:
    print(f"  {a['name']}: {a['return_pct']:+.2f}% (${a['return_cad']:+.2f})")
print("Alerts:", len(data["alerts"]))
if data["alerts"]:
    for a in data["alerts"][:3]:
        print(f"  [{a['type']}] {a['message']}")
