import sys
sys.path.append(r"d:\Fpl_winning")
from ortools.sat.python import cp_model

def test_ft_logic(start_ft, transfers):
    model = cp_model.CpModel()
    
    # We want to know what FT for NEXT week would be
    ft_prev = model.NewIntVar(0, 5, "ft_prev")
    buy = model.NewIntVar(0, 15, "buy")
    
    model.Add(ft_prev == start_ft)
    model.Add(buy == transfers)
    
    ft_next = model.NewIntVar(0, 5, "ft_next")
    
    # Logic from multi_gw.py
    prev_rem = model.NewIntVar(-15, 15, "prev_rem")
    model.Add(prev_rem == ft_prev - buy + 1)
    
    capped_ft = model.NewIntVar(1, 15, "capped_ft")
    model.AddMaxEquality(capped_ft, [1, prev_rem])
    model.AddMinEquality(ft_next, [5, capped_ft])
    
    # Hits calculation
    hits = model.NewIntVar(0, 15, "hits")
    diff = model.NewIntVar(-15, 15, "diff")
    model.Add(diff == buy - ft_prev) # diff = transfers_made - FT_t
    model.AddMaxEquality(hits, [0, diff])
    
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Start FT: {start_ft}, Transfers: {transfers} -> Next FT: {solver.Value(ft_next)}, Hits taken: {solver.Value(hits)}")
    else:
        print(f"Start FT: {start_ft}, Transfers: {transfers} -> INFEASIBLE")

if __name__ == '__main__':
    print("Testing FT Rollover Logic (CP-SAT natively)")
    for ft in [0, 1, 4, 5]:
        for tr in [0, 1, 2, 3]:
            test_ft_logic(ft, tr)
