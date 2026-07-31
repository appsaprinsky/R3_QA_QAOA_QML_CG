import pulp

class MasterProblem:
    def __init__(self, customers):
        self.customers = customers
        self.columns = []

    def add_column(self, nodes, cost):
        self.columns.append({'nodes': nodes, 'cost': cost})

    def solve_lp_relaxation(self):
        prob = pulp.LpProblem("VRP_Master_LP", pulp.LpMinimize)
        
        x = [pulp.LpVariable(f"col_{j}", lowBound=0.0, upBound=1.0, cat='Continuous') for j in range(len(self.columns))]
        
        prob += pulp.lpSum(self.columns[j]['cost'] * x[j] for j in range(len(self.columns)))
        
        constraints = {}
        for cust in self.customers:
            # Fixed assignment syntax: add constraint to prob first, then reference it
            constraint_expression = pulp.lpSum(x[j] for j in range(len(self.columns)) if cust in self.columns[j]['nodes'][1:-1]) >= 1.0
            prob += (constraint_expression, f"cov_{cust}")
            constraints[cust] = prob.constraints[f"cov_{cust}"]
            
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if pulp.LpStatus[prob.status] != 'Optimal':
            return None, None
            
        # Extract dual variables (shadow prices)
        duals = {cust: constraints[cust].pi for cust in self.customers}
        return duals, prob.objective.value()

    def solve_integer_fixing(self):
        prob = pulp.LpProblem("VRP_Master_MIP", pulp.LpMinimize)
        
        x = [pulp.LpVariable(f"col_mip_{j}", cat='Binary') for j in range(len(self.columns))]
        
        prob += pulp.lpSum(self.columns[j]['cost'] * x[j] for j in range(len(self.columns)))
        
        for cust in self.customers:
            constraint_expression = pulp.lpSum(x[j] for j in range(len(self.columns)) if cust in self.columns[j]['nodes'][1:-1]) >= 1.0
            prob += (constraint_expression, f"mip_cov_{cust}")
            
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        selected_routes = []
        if pulp.LpStatus[prob.status] == 'Optimal':
            for j, var in enumerate(x):
                if var.varValue > 0.5:
                    selected_routes.append(self.columns[j])
                    
        return selected_routes