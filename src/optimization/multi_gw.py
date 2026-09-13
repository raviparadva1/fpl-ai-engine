from ortools.sat.python import cp_model
import pandas as pd
import numpy as np

class MultiGWOptimizer:
    def __init__(self, horizon=5, bank=0.0, free_transfers=1):
        self.horizon = horizon
        self.bank = bank
        self.free_transfers = min(5, max(0, free_transfers))  # 24/25 rule max 5
        self.model = cp_model.CpModel()
        
    def build_model(self, player_data_by_gw, current_squad_ids, strategy_config):
        """
        player_data_by_gw: list of H dataframes (one for each GW in horizon).
                           must contain columns: player_code, name, position, team, value, xP, P10, P50, P90, ownership
        strategy_config: dict specifying objective (e.g., 'attack', 'defend', 'neutral', risk_penalty)
        """
        self.players = player_data_by_gw[0].to_dict('records')
        self.player_indices = {p['player_code']: i for i, p in enumerate(self.players)}
        n = len(self.players)
        H = self.horizon
        
        # Decision Variables
        # Indexed by [gw][player]
        self.sq = [[self.model.NewBoolVar(f"sq_{t}_{i}") for i in range(n)] for t in range(H)]
        self.xi = [[self.model.NewBoolVar(f"xi_{t}_{i}") for i in range(n)] for t in range(H)]
        self.cap = [[self.model.NewBoolVar(f"cap_{t}_{i}") for i in range(n)] for t in range(H)]
        self.buy = [[self.model.NewBoolVar(f"buy_{t}_{i}") for i in range(n)] for t in range(H)]
        self.sell = [[self.model.NewBoolVar(f"sell_{t}_{i}") for i in range(n)] for t in range(H)]
        
        # State Variables
        self.ft = [self.model.NewIntVar(0, 5, f"ft_{t}") for t in range(H)]
        self.hits = [self.model.NewIntVar(0, 15, f"hits_{t}") for t in range(H)]
        self.bank_var = [self.model.NewIntVar(0, 1000, f"bank_{t}") for t in range(H)] # value x10
        
        current_sq_arr = [1 if self.players[i]['player_code'] in current_squad_ids else 0 for i in range(n)]
        
        # Constraints per GW
        pos_limits = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        xi_min = {'GK': 1, 'DEF': 3, 'MID': 2, 'FWD': 1}
        xi_max = {'GK': 1, 'DEF': 5, 'MID': 5, 'FWD': 3}
        
        for t in range(H):
            # Squad Size
            self.model.Add(sum(self.sq[t]) == 15)
            self.model.Add(sum(self.xi[t]) == 11)
            self.model.Add(sum(self.cap[t]) == 1)
            
            # Position & Team constraints
            for pos in ['GK', 'DEF', 'MID', 'FWD']:
                p_idx = [i for i, p in enumerate(self.players) if p['position'] == pos]
                self.model.Add(sum(self.sq[t][i] for i in p_idx) == pos_limits[pos])
                self.model.Add(sum(self.xi[t][i] for i in p_idx) >= xi_min[pos])
                self.model.Add(sum(self.xi[t][i] for i in p_idx) <= xi_max[pos])
                
            teams = set(p['team'] for p in self.players)
            for tm in teams:
                t_idx = [i for i, p in enumerate(self.players) if p['team'] == tm]
                self.model.Add(sum(self.sq[t][i] for i in t_idx) <= 3)
                
            # Logical constraints
            for i in range(n):
                self.model.Add(self.xi[t][i] <= self.sq[t][i])
                self.model.Add(self.cap[t][i] <= self.xi[t][i])
                
                # Transitions
                if t == 0:
                    self.model.Add(self.sq[t][i] == current_sq_arr[i] + self.buy[t][i] - self.sell[t][i])
                else:
                    self.model.Add(self.sq[t][i] == self.sq[t-1][i] + self.buy[t][i] - self.sell[t][i])
                    
            # Budget and Transfers
            cost = sum(self.buy[t][i] * int(self.players[i]['value'] * 10) for i in range(n))
            gain = sum(self.sell[t][i] * int(self.players[i]['value'] * 10) for i in range(n)) # simplification: sell = current val
            
            if t == 0:
                self.model.Add(self.bank_var[t] == int(self.bank * 10) + gain - cost)
                transfers_made = sum(self.buy[t])
                # Hits logic (Hits >= transfers - FT)
                # CP-SAT requires intermediate variables for Max(0, X)
                diff = self.model.NewIntVar(-15, 15, f"diff_{t}")
                self.model.Add(diff == transfers_made - self.free_transfers)
                self.model.AddMaxEquality(self.hits[t], [0, diff])
            else:
                self.model.Add(self.bank_var[t] == self.bank_var[t-1] + gain - cost)
                transfers_made = sum(self.buy[t])
                # FT logic for next week: min(5, FT_{t-1} - transfers_{t-1} + 1) if no hits were taken
                # To simplify strictly for CP-SAT without complex non-linear tracking of chips:
                prev_rem = self.model.NewIntVar(-15, 15, f"prev_rem_{t}")
                self.model.Add(prev_rem == self.ft[t-1] - sum(self.buy[t-1]) + 1)
                capped_ft = self.model.NewIntVar(1, 15, f"capped_ft_{t}") # max 5 internally
                self.model.AddMaxEquality(capped_ft, [1, prev_rem])
                self.model.AddMinEquality(self.ft[t], [5, capped_ft])
                
                diff = self.model.NewIntVar(-15, 15, f"diff_{t}")
                self.model.Add(diff == transfers_made - self.ft[t])
                self.model.AddMaxEquality(self.hits[t], [0, diff])

        # Objective Function
        # Note: As requested, not hardcoding Expected Points - 0.2*(P90-P10).
        # Dynamic strategy mapping
        obj_expr = 0
        discount_factor = 0.85
        
        # Scale everything by 1000 to convert floats to ints for CP-SAT
        for t in range(H):
            gw_df = player_data_by_gw[t]
            df_dict = {p['player_code']: p for p in gw_df.to_dict('records')}
            
            for i in range(n):
                code = self.players[i]['player_code']
                p_data = df_dict.get(code, self.players[i]) # fallback
                
                # Base xP
                base_val = p_data.get('xP', 0.0)
                
                # Apply strategy modifications
                if strategy_config.get('mode') == 'attack':
                    # Bonus for differentials (low ownership)
                    own_pct = p_data.get('selected_by_percent', 0.0) / 100.0
                    base_val += strategy_config.get('attack_weight', 0.5) * base_val * (1 - own_pct)
                
                elif strategy_config.get('mode') == 'defend':
                    # Penalty for picking low ownership when defending
                    own_pct = p_data.get('selected_by_percent', 0.0) / 100.0
                    base_val -= strategy_config.get('defend_weight', 0.3) * base_val * (1 - own_pct)
                    
from ortools.sat.python import cp_model
import pandas as pd
import numpy as np

class MultiGWOptimizer:
    def __init__(self, horizon=5, bank=0.0, free_transfers=1):
        self.horizon = horizon
        self.bank = bank
        self.free_transfers = min(5, max(0, free_transfers))  # 24/25 rule max 5
        self.model = cp_model.CpModel()
        
    def build_model(self, player_data_by_gw, current_squad_ids, strategy_config):
        """
        player_data_by_gw: list of H dataframes (one for each GW in horizon).
                           must contain columns: player_code, name, position, team, value, xP, P10, P50, P90, ownership
        strategy_config: dict specifying objective (e.g., 'attack', 'defend', 'neutral', risk_penalty)
        """
        self.players = player_data_by_gw[0].to_dict('records')
        pos_map = {
            'Goalkeeper': 'GK', 'Defender': 'DEF', 'Midfielder': 'MID', 'Forward': 'FWD',
            'G': 'GK', 'GKP': 'GK', 'D': 'DEF', 'M': 'MID', 'F': 'FWD',
            'CB': 'DEF', 'FB': 'DEF', 'CM': 'MID', 'AM': 'MID', 'WM': 'MID', 'CF': 'FWD', 'W': 'FWD'
        }
        for p in self.players:
            p['position'] = pos_map.get(p['position'], p['position'])
            
        self.player_indices = {p['player_code']: i for i, p in enumerate(self.players)}
        n = len(self.players)
        H = self.horizon
        self.H = H
        self.initial_sq = [1 if self.players[i]['player_code'] in current_squad_ids else 0 for i in range(n)]
        
        # Decision Variables
        # Indexed by [gw][player]
        self.sq = [[self.model.NewBoolVar(f"sq_{t}_{i}") for i in range(n)] for t in range(H)]
        self.xi = [[self.model.NewBoolVar(f"xi_{t}_{i}") for i in range(n)] for t in range(H)]
        self.cap = [[self.model.NewBoolVar(f"cap_{t}_{i}") for i in range(n)] for t in range(H)]
        self.buy = [[self.model.NewBoolVar(f"buy_{t}_{i}") for i in range(n)] for t in range(H)]
        self.sell = [[self.model.NewBoolVar(f"sell_{t}_{i}") for i in range(n)] for t in range(H)]
        
        # State Variables
        self.ft = [self.model.NewIntVar(0, 5, f"ft_{t}") for t in range(H)]
        self.hits = [self.model.NewIntVar(0, 15, f"hits_{t}") for t in range(H)]
        self.bank_var = [self.model.NewIntVar(0, 1000, f"bank_{t}") for t in range(H)] # value x10
        
        current_sq_arr = self.initial_sq
        
        # Constraints per GW
        pos_limits = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        xi_min = {'GK': 1, 'DEF': 3, 'MID': 2, 'FWD': 1}
        xi_max = {'GK': 1, 'DEF': 5, 'MID': 5, 'FWD': 3}
        
        self.diff = [None]*H
        self.prev_rem = [None]*H
        self.capped_ft = [None]*H
        
        for t in range(H):
            # Squad Size
            self.model.Add(sum(self.sq[t]) == 15)
            self.model.Add(sum(self.xi[t]) == 11)
            self.model.Add(sum(self.cap[t]) == 1)
            
            # Position & Team constraints
            for pos in ['GK', 'DEF', 'MID', 'FWD']:
                p_idx = [i for i, p in enumerate(self.players) if p['position'] == pos]
                self.model.Add(sum(self.sq[t][i] for i in p_idx) == pos_limits[pos])
                self.model.Add(sum(self.xi[t][i] for i in p_idx) >= xi_min[pos])
                self.model.Add(sum(self.xi[t][i] for i in p_idx) <= xi_max[pos])
                
            teams = set(p['team'] for p in self.players)
            for tm in teams:
                t_idx = [i for i, p in enumerate(self.players) if p['team'] == tm]
                self.model.Add(sum(self.sq[t][i] for i in t_idx) <= 3)
                
            # Logical constraints
            for i in range(n):
                self.model.Add(self.xi[t][i] <= self.sq[t][i])
                self.model.Add(self.cap[t][i] <= self.xi[t][i])
                
                # Transitions
                if t == 0:
                    self.model.Add(self.sq[t][i] == current_sq_arr[i] + self.buy[t][i] - self.sell[t][i])
                else:
                    self.model.Add(self.sq[t][i] == self.sq[t-1][i] + self.buy[t][i] - self.sell[t][i])
                    
            # Budget and Transfers
            cost = sum(self.buy[t][i] * int(self.players[i]['value'] * 10) for i in range(n))
            gain = sum(self.sell[t][i] * int(self.players[i]['value'] * 10) for i in range(n)) # simplification: sell = current val
            
            if t == 0:
                self.model.Add(self.bank_var[t] == int(self.bank * 10) + gain - cost)
                transfers_made = sum(self.buy[t])
                # Hits logic (Hits >= transfers - FT)
                # CP-SAT requires intermediate variables for Max(0, X)
                self.diff[t] = self.model.NewIntVar(-15, 15, f"diff_{t}")
                self.model.Add(self.diff[t] == transfers_made - self.free_transfers)
                self.model.AddMaxEquality(self.hits[t], [0, self.diff[t]])
            else:
                self.model.Add(self.bank_var[t] == self.bank_var[t-1] + gain - cost)
                transfers_made = sum(self.buy[t])
                # FT logic for next week: min(5, FT_{t-1} - transfers_{t-1} + 1) if no hits were taken
                # To simplify strictly for CP-SAT without complex non-linear tracking of chips:
                self.prev_rem[t] = self.model.NewIntVar(-15, 15, f"prev_rem_{t}")
                self.model.Add(self.prev_rem[t] == self.ft[t-1] - sum(self.buy[t-1]) + 1)
                self.capped_ft[t] = self.model.NewIntVar(1, 15, f"capped_ft_{t}") # max 5 internally
                self.model.AddMaxEquality(self.capped_ft[t], [1, self.prev_rem[t]])
                self.model.AddMinEquality(self.ft[t], [5, self.capped_ft[t]])
                
                self.diff[t] = self.model.NewIntVar(-15, 15, f"diff_{t}")
                self.model.Add(self.diff[t] == transfers_made - self.ft[t])
                self.model.AddMaxEquality(self.hits[t], [0, self.diff[t]])

        # Objective Function
        # Note: As requested, not hardcoding Expected Points - 0.2*(P90-P10).
        # Dynamic strategy mapping
        obj_expr = 0
        discount_factor = 0.85
        
        # Scale everything by 1000 to convert floats to ints for CP-SAT
        for t in range(H):
            gw_df = player_data_by_gw[t]
            df_dict = {p['player_code']: p for p in gw_df.to_dict('records')}
            
            for i in range(n):
                code = self.players[i]['player_code']
                p_data = df_dict.get(code, self.players[i]) # fallback
                
                # Base xP
                base_val = p_data.get('xP', 0.0)
                
                # Apply strategy modifications
                if strategy_config.get('mode') == 'attack':
                    # Bonus for differentials (low ownership)
                    own_pct = p_data.get('selected_by_percent', 0.0) / 100.0
                    base_val += strategy_config.get('attack_weight', 0.5) * base_val * (1 - own_pct)
                
                elif strategy_config.get('mode') == 'defend':
                    # Penalty for picking low ownership when defending
                    own_pct = p_data.get('selected_by_percent', 0.0) / 100.0
                    base_val -= strategy_config.get('defend_weight', 0.3) * base_val * (1 - own_pct)
                    
                # Downside risk penalty (Configurable, avoiding hardcoded P90-P10)
                risk_penalty = strategy_config.get('risk_penalty', 0.0)
                if risk_penalty > 0:
                    downside = p_data.get('xP', 0) - p_data.get('P10', 0)
                    base_val -= (risk_penalty * downside)
                
                int_val = int(base_val * 1000 * (discount_factor ** t))
                obj_expr += (self.xi[t][i] + self.cap[t][i]) * int_val
                
            obj_expr -= (self.hits[t] * 4000 * (discount_factor ** t))
            
        self.model.Maximize(obj_expr)
        
    def solve(self):
        # --- WARM START HINTS ---
        # Provide a trivial 0-transfer legal solution to guarantee FEASIBLE baseline
        hint_sq = self.initial_sq
        hint_xi = [0] * len(self.players)
        pos_added = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        for i in range(len(self.players)):
            if hint_sq[i] == 1:
                pos = self.players[i]['position']
                if pos == 'GK' and pos_added['GK'] < 1:
                    hint_xi[i] = 1; pos_added['GK'] += 1
                elif pos == 'DEF' and pos_added['DEF'] < 3:
                    hint_xi[i] = 1; pos_added['DEF'] += 1
                elif pos == 'MID' and pos_added['MID'] < 4:
                    hint_xi[i] = 1; pos_added['MID'] += 1
                elif pos == 'FWD' and pos_added['FWD'] < 3:
                    hint_xi[i] = 1; pos_added['FWD'] += 1
                    
        for t in range(self.H):
            cap_selected = False
            for i in range(len(self.players)):
                self.model.AddHint(self.buy[t][i], 0)
                self.model.AddHint(self.sell[t][i], 0)
                self.model.AddHint(self.sq[t][i], hint_sq[i])
                self.model.AddHint(self.xi[t][i], hint_xi[i])
                if hint_xi[i] == 1 and not cap_selected:
                    self.model.AddHint(self.cap[t][i], 1)
                    cap_selected = True
                else:
                    self.model.AddHint(self.cap[t][i], 0)
            
            # Hint the 46 state variables to ensure a 100% complete solution hint
            self.model.AddHint(self.bank_var[t], int(self.bank * 10))
            self.model.AddHint(self.hits[t], 0)
            
            if t == 0:
                self.model.AddHint(self.diff[t], 0 - self.free_transfers)
            else:
                hint_ft_prev = min(5, self.free_transfers + t - 1)
                hint_prev_rem = hint_ft_prev + 1
                hint_capped_ft = min(5, hint_prev_rem)
                self.model.AddHint(self.prev_rem[t], hint_prev_rem)
                self.model.AddHint(self.capped_ft[t], max(1, hint_prev_rem))
                self.model.AddHint(self.ft[t], min(5, hint_capped_ft))
                self.model.AddHint(self.diff[t], 0 - min(5, hint_capped_ft))
        
        self.model.AddHint(self.ft[0], self.free_transfers)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0
        status = solver.Solve(self.model)
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            buy_codes = [self.players[i]['player_code'] for i in range(len(self.players)) if solver.Value(self.buy[0][i]) == 1]
            sell_codes = [self.players[i]['player_code'] for i in range(len(self.players)) if solver.Value(self.sell[0][i]) == 1]
            xi_codes = [self.players[i]['player_code'] for i in range(len(self.players)) if solver.Value(self.xi[0][i]) == 1]
            cap_code = [self.players[i]['player_code'] for i in range(len(self.players)) if solver.Value(self.cap[0][i]) == 1][0]
            
            print(f"DEBUG: buy={sum(solver.Value(b) for b in self.buy[0])}, FT={self.free_transfers}, diff={solver.Value(self.diff[0])}, hits={solver.Value(self.hits[0])}")
            return {
                'status': 'OPTIMAL' if status == cp_model.OPTIMAL else 'FEASIBLE',
                'buy': buy_codes,
                'sell': sell_codes,
                'xi': xi_codes,
                'cap': cap_code,
                'hits': solver.Value(self.hits[0]),
                'projected_obj': solver.ObjectiveValue() / 1000.0
            }
        elif status == cp_model.UNKNOWN:
            print(f"DEBUG: buy={sum(solver.Value(b) for b in self.buy[0])}, FT={self.free_transfers}, diff={solver.Value(self.diff[0])}, hits={solver.Value(self.hits[0])}")
            return {'status': 'UNKNOWN'}
        else:
            print(f"DEBUG: buy={sum(solver.Value(b) for b in self.buy[0])}, FT={self.free_transfers}, diff={solver.Value(self.diff[0])}, hits={solver.Value(self.hits[0])}")
            return {'status': 'INFEASIBLE'}
