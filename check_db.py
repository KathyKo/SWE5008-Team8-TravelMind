# check_db.py
import json
from backend.db.database import SessionLocal
from backend.db.models import Plan, PlanItinerary, PlanFlight, PlanHotel, PlanExplain

db = SessionLocal()

plans = db.query(Plan).all()
for p in plans:
    print("=" * 60)
    print(f"plan_id:     {p.plan_id}")
    print(f"route:       {p.origin} -> {p.destination}")
    print(f"dates:       {p.dates} ({p.duration})")
    print(f"budget:      {p.budget}")
    print(f"preferences: {p.preferences}")
    print(f"option_meta: {json.dumps(p.option_meta, indent=2, ensure_ascii=False)}")
    print(f"planner_decision_trace keys: {list((p.planner_decision_trace or {}).keys())}")

    itins = db.query(PlanItinerary).filter(PlanItinerary.plan_id == p.plan_id).all()
    for itin in itins:
        print(f"\n  --- Itinerary {itin.option} ---")
        print(json.dumps(itin.days, indent=2, ensure_ascii=False)[:500])  # 前500字

    flights = db.query(PlanFlight).filter(PlanFlight.plan_id == p.plan_id).all()
    print(f"\n  flights ({len(flights)} 筆):")
    for f in flights[:3]:  # 只印前3筆
        print(f"    [{f.direction}] {json.dumps(f.flight_data, ensure_ascii=False)[:200]}")

    hotels = db.query(PlanHotel).filter(PlanHotel.plan_id == p.plan_id).all()
    print(f"\n  hotels ({len(hotels)} 筆):")
    for h in hotels[:3]:
        print(f"    {json.dumps(h.hotel_data, ensure_ascii=False)[:200]}")

explains = db.query(PlanExplain).all()
print("\n" + "=" * 60)
print(f"plan_explains: {len(explains)} 筆")
for e in explains:
    print(f"  plan_id={e.plan_id}")
    print(json.dumps(e.explain_data, indent=2, ensure_ascii=False)[:500])

db.close()
