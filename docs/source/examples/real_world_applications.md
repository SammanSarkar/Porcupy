# Real-World Applications

This guide demonstrates practical applications of Porcupy in various domains, showcasing its versatility and effectiveness in solving real-world optimization problems.

## Table of Contents
- [Hyperparameter Tuning for Machine Learning](#hyperparameter-tuning-for-machine-learning)
- [Neural Network Architecture Search](#neural-network-architecture-search)
- [Portfolio Optimization](#portfolio-optimization)
- [Engineering Design Optimization](#engineering-design-optimization)
- [Supply Chain Optimization](#supply-chain-optimization)
- [Energy System Optimization](#energy-system-optimization)

## Hyperparameter Tuning for Machine Learning

### Tuning a Random Forest Classifier

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from porcupy import CPO

def tune_random_forest():
    # Load dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define the objective function (to minimize)
    def objective(params):
        # Convert parameters
        params = {
            'n_estimators': int(params[0]),
            'max_depth': int(params[1]) if params[1] > 1 else None,
            'min_samples_split': int(params[2]),
            'min_samples_leaf': int(params[3]),
            'max_features': params[4],
            'bootstrap': params[5] > 0.5
        }
        
        # Create and evaluate model
        model = RandomForestClassifier(
            **params,
            random_state=42,
            n_jobs=-1
        )
        
        # Use cross-validation for robust evaluation
        scores = cross_val_score(
            model, X_train, y_train, 
            cv=5, scoring='neg_log_loss',
            n_jobs=-1, error_score='raise'
        )
        
        return -np.mean(scores)  # Minimize negative log loss
    
    # Define parameter bounds
    bounds = [
        (10, 200),      # n_estimators
        (1, 20),        # max_depth
        (2, 20),        # min_samples_split
        (1, 20),        # min_samples_leaf
        (0.1, 1.0),     # max_features (fraction of features)
        (0, 1)          # bootstrap (0-1 as float, will be converted to bool)
    ]
    
    # Initialize and run optimizer
    optimizer = CPO(
        dimensions=len(bounds),
        bounds=bounds,
        pop_size=30,
        max_iter=30,
        verbose=True
    )
    
    best_params, best_score, _ = optimizer.optimize(objective)
    
    # Train final model with best parameters
    final_params = {
        'n_estimators': int(best_params[0]),
        'max_depth': int(best_params[1]) if best_params[1] > 1 else None,
        'min_samples_split': int(best_params[2]),
        'min_samples_leaf': int(best_params[3]),
        'max_features': best_params[4],
        'bootstrap': best_params[5] > 0.5,
        'random_state': 42,
        'n_jobs': -1
    }
    
    final_model = RandomForestClassifier(**final_params)
    final_model.fit(X_train, y_train)
    
    # Evaluate on test set
    test_score = final_model.score(X_test, y_test)
    
    return {
        'best_params': final_params,
        'best_score': -best_score,  # Convert back to positive log loss
        'test_accuracy': test_score
    }
```

## Neural Network Architecture Search

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def neural_architecture_search():
    # Generate synthetic data
    X = np.random.randn(1000, 20)
    y = (X[:, 0] + 2*X[:, 1] - 1.5*X[:, 2] > 0).astype(int)
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # Split into train and test
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Define search space
    def create_model(params, input_dim=20, output_dim=2):
        # Extract architecture parameters
        n_layers = int(params[0])  # 1-3 hidden layers
        layer1 = int(2 ** (4 + params[1]))  # 16-512 neurons
        layer2 = int(2 ** (4 + params[2])) if n_layers > 1 else 0
        layer3 = int(2 ** (4 + params[3])) if n_layers > 2 else 0
        dropout = params[4]  # 0-0.5
        lr = 10 ** (-2 - 2 * params[5])  # 1e-2 to 1e-4
        
        # Build model
        layers = []
        prev_dim = input_dim
        
        for dim in [layer1, layer2, layer3]:
            if dim == 0:
                continue
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers), lr
    
    # Training function
    def train_model(params, num_epochs=20):
        model, learning_rate = create_model(params)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        model.train()
        for epoch in range(num_epochs):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Evaluate on test set
        model.eval()
        test_loader = DataLoader(test_dataset, batch_size=32)
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        accuracy = correct / total
        return 1 - accuracy  # Minimize error rate
    
    # Define parameter bounds
    bounds = [
        (1, 3),      # Number of hidden layers (1-3)
        (0, 5),      # Layer 1 size (16-512)
        (0, 5),      # Layer 2 size (16-512)
        (0, 5),      # Layer 3 size (16-512)
        (0, 0.5),    # Dropout rate (0-0.5)
        (0, 1)       # Learning rate (1e-2 to 1e-4)
    ]
    
    # Run optimization
    optimizer = CPO(
        dimensions=len(bounds),
        bounds=bounds,
        pop_size=20,
        max_iter=15,
        verbose=True
    )
    
    best_params, best_error, _ = optimizer.optimize(train_model)
    
    # Get best model architecture
    best_model, best_lr = create_model(best_params)
    
    return {
        'best_params': best_params,
        'best_accuracy': 1 - best_error,
        'model_architecture': str(best_model)
    }
```

## Portfolio Optimization

```python
import pandas as pd
import yfinance as yf

def portfolio_optimization():
    # Download stock data
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK-B', 'JPM', 'V', 'JNJ']
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    
    # Get adjusted close prices
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    # Calculate daily returns
    returns = data.pct_change().dropna()
    
    # Calculate expected returns and covariance
    mu = returns.mean() * 252  # Annualized returns
    sigma = returns.cov() * 252  # Annualized covariance
    
    # Define objective function (negative Sharpe ratio)
    def objective(weights):
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Ensure weights sum to 1
        
        # Portfolio return
        port_return = np.sum(weights * mu)
        
        # Portfolio volatility
        port_vol = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
        
        # Risk-free rate (approximate)
        rf_rate = 0.02  # 2%
        
        # Sharpe ratio (negative because we minimize)
        sharpe_ratio = (port_return - rf_rate) / (port_vol + 1e-8)
        return -sharpe_ratio
    
    # Define constraints (weights sum to 1, no short selling)
    def constraint(weights):
        return np.sum(weights) - 1  # sum(weights) = 1
    
    # Run optimization
    optimizer = CPO(
        dimensions=len(tickers),
        bounds=[(0, 1)] * len(tickers),  # No short selling
        pop_size=50,
        max_iter=100,
        ftol=1e-6
    )
    
    best_weights, best_sharpe, _ = optimizer.optimize(
        objective_func=objective,
        f_eqcons=constraint
    )
    
    # Normalize weights to sum to 1
    best_weights = np.array(best_weights)
    best_weights = best_weights / np.sum(best_weights)
    
    # Calculate portfolio metrics
    port_return = np.sum(best_weights * mu)
    port_vol = np.sqrt(np.dot(best_weights.T, np.dot(sigma, best_weights)))
    sharpe_ratio = (port_return - 0.02) / port_vol
    
    return {
        'tickers': tickers,
        'weights': dict(zip(tickers, best_weights)),
        'expected_return': port_return,
        'volatility': port_vol,
        'sharpe_ratio': sharpe_ratio
    }
```

## Engineering Design Optimization

### Truss Structure Optimization

```python
def truss_optimization():
    """
    Optimize a simple truss structure to minimize weight while
    satisfying stress and displacement constraints.
    """
    # Material properties (steel)
    E = 200e9  # Young's modulus (Pa)
    rho = 7850  # Density (kg/m^3)
    sigma_max = 250e6  # Maximum allowable stress (Pa)
    delta_max = 0.01  # Maximum allowable displacement (m)
    
    # Geometry (2D truss with 10 bars)
    # Node coordinates (x, y) in meters
    nodes = np.array([
        [0, 0],    # Node 0
        [1, 0],    # Node 1
        [2, 0],    # Node 2
        [3, 0],    # Node 3
        [1, 1],    # Node 4
        [2, 1]     # Node 5
    ])
    
    # Element connectivity (start node, end node)
    elements = [
        (0, 1), (1, 2), (2, 3),  # Bottom chord
        (4, 5),                    # Top chord
        (0, 4), (1, 4), (1, 5), (2, 5), (2, 3), (3, 5)  # Diagonals and verticals
    ]
    
    # External loads (node, direction: 0=x, 1=y, force in N)
    loads = [
        (4, 1, -1e5),  # 100 kN downward at node 4
        (5, 1, -1e5)   # 100 kN downward at node 5
    ]
    
    # Fixed nodes (node, direction: 0=x, 1=y)
    supports = [
        (0, 0), (0, 1),  # Fixed at node 0 (x and y)
        (3, 1)           # Roller at node 3 (y only)
    ]
    
    def analyze_truss(areas):
        """
        Analyze the truss and return weight, max stress, and max displacement.
        areas: Cross-sectional areas of each element (m^2)
        """
        # This is a placeholder for the actual finite element analysis
        # In a real implementation, you would solve the truss equations here
        
        # For demonstration, we'll use simplified calculations
        total_length = sum(
            np.linalg.norm(nodes[j] - nodes[i]) 
            for i, j in elements
        )
        
        # Simplified calculations (not actual FEA)
        weight = total_length * np.mean(areas) * rho
        
        # These would come from FEA in a real implementation
        max_stress = 100e6 + np.random.normal(0, 10e6)  # Simulated
        max_disp = 0.005 + np.random.normal(0, 0.001)   # Simulated
        
        return weight, max_stress, max_disp
    
    # Objective function for optimization
    def objective(areas):
        weight, max_stress, max_disp = analyze_truss(areas)
        
        # Penalty for constraints
        stress_penalty = max(0, (max_stress / sigma_max) - 1) ** 2
        disp_penalty = max(0, (max_disp / delta_max) - 1) ** 2
        
        # Total cost (minimize weight plus penalties)
        cost = weight * (1 + 1e6 * (stress_penalty + disp_penalty))
        return cost
    
    # Bounds for cross-sectional areas (m^2)
    n_elements = len(elements)
    bounds = [(1e-4, 1e-2)] * n_elements  # 1 cm² to 100 cm²
    
    # Run optimization
    optimizer = CPO(
        dimensions=n_elements,
        bounds=bounds,
        pop_size=50,
        max_iter=100,
        ftol=1e-4
    )
    
    best_areas, best_cost, _ = optimizer.optimize(objective)
    
    # Analyze final design
    weight, max_stress, max_disp = analyze_truss(best_areas)
    
    return {
        'best_areas': best_areas,
        'weight': weight,
        'max_stress': max_stress,
        'max_displacement': max_disp,
        'constraints_satisfied': (
            max_stress <= sigma_max * 1.001 and  # Allow small numerical tolerance
            max_disp <= delta_max * 1.001
        )
    }
```

## Supply Chain Optimization

```python
def supply_chain_optimization():
    """
    Optimize a multi-echelon supply chain network to minimize total cost
    while meeting demand and capacity constraints.
    """
    # Problem parameters
    n_suppliers = 3
    n_warehouses = 4
    n_customers = 5
    n_products = 2
    n_periods = 12  # Monthly planning for a year
    
    # Generate random data (in a real application, this would come from your data)
    np.random.seed(42)
    
    # Production capacity at each supplier (units per period)
    supplier_capacity = np.random.randint(1000, 5000, (n_suppliers, n_products))
    
    # Holding capacity at each warehouse (units)
    warehouse_capacity = np.random.randint(5000, 10000, n_warehouses)
    
    # Transportation costs
    # supplier -> warehouse cost per unit per km
    supplier_warehouse_cost = np.random.uniform(0.1, 0.5, (n_suppliers, n_warehouses, n_products))
    # warehouse -> customer cost per unit per km
    warehouse_customer_cost = np.random.uniform(0.2, 0.8, (n_warehouses, n_customers, n_products))
    
    # Holding cost at warehouses ($/unit/period)
    holding_cost = np.random.uniform(0.5, 2.0, (n_warehouses, n_products))
    
    # Demand at each customer for each product in each period
    demand = np.random.randint(50, 200, (n_periods, n_customers, n_products))
    
    # Distances (simplified as random values)
    supplier_warehouse_dist = np.random.uniform(10, 100, (n_suppliers, n_warehouses))
    warehouse_customer_dist = np.random.uniform(5, 50, (n_warehouses, n_customers))
    
    def evaluate_solution(solution):
        """
        Evaluate a supply chain solution.
        solution: [supplier_warehouse_flow, warehouse_customer_flow]
        """
        # Reshape solution
        sw_flow = solution[:n_suppliers * n_warehouses * n_products * n_periods]
        sw_flow = sw_flow.reshape((n_suppliers, n_warehouses, n_products, n_periods))
        
        wc_flow = solution[n_suppliers * n_warehouses * n_products * n_periods:]
        wc_flow = wc_flow.reshape((n_warehouses, n_customers, n_products, n_periods))
        
        total_cost = 0
        
        # Calculate transportation costs
        # Supplier to warehouse
        transport_cost_sw = np.sum(
            sw_flow * 
            supplier_warehouse_cost[:, :, :, np.newaxis] * 
            supplier_warehouse_dist[:, :, np.newaxis, np.newaxis]
        )
        
        # Warehouse to customer
        transport_cost_wc = np.sum(
            wc_flow * 
            warehouse_customer_cost[:, :, :, np.newaxis] * 
            warehouse_customer_dist[:, :, np.newaxis, np.newaxis]
        )
        
        # Calculate holding costs
        # Inventory at warehouses (simplified)
        inventory = np.cumsum(
            np.sum(sw_flow, axis=0) - np.sum(wc_flow, axis=1),
            axis=2
        )
        holding_costs = np.sum(inventory * holding_cost[:, :, np.newaxis])
        
        total_cost = transport_cost_sw + transport_cost_wc + holding_costs
        
        # Calculate constraint violations
        # 1. Supplier capacity
        supplier_utilization = np.sum(sw_flow, axis=(1, 3))  # Sum over warehouses and periods
        supplier_violation = np.sum(np.maximum(0, supplier_utilization - supplier_capacity[:, :, np.newaxis]))
        
        # 2. Warehouse capacity
        max_inventory = np.max(np.sum(inventory, axis=1), axis=1)  # Max inventory per warehouse
        warehouse_violation = np.sum(np.maximum(0, max_inventory - warehouse_capacity))
        
        # 3. Demand satisfaction
        demand_fulfilled = np.sum(wc_flow, axis=0)  # Sum over warehouses
        demand_violation = np.sum(np.maximum(0, demand - demand_fulfilled))
        
        # Add penalty terms to cost
        penalty = 1e6 * (supplier_violation + warehouse_violation + demand_violation)
        
        return total_cost + penalty
    
    # Define optimization problem
    n_vars = (
        n_suppliers * n_warehouses * n_products * n_periods +  # Supplier-warehouse flow
        n_warehouses * n_customers * n_products * n_periods     # Warehouse-customer flow
    )
    
    # Bounds (non-negative flows)
    bounds = [(0, None)] * n_vars
    
    # Run optimization
    optimizer = CPO(
        dimensions=n_vars,
        bounds=bounds,
        pop_size=30,
        max_iter=50,
        verbose=True
    )
    
    # Note: For large problems, consider using a more efficient encoding
    # or decomposition approach
    best_solution, best_cost, _ = optimizer.optimize(evaluate_solution)
    
    # Decode solution
    sw_flow = best_solution[:n_suppliers * n_warehouses * n_products * n_periods]
    sw_flow = sw_flow.reshape((n_suppliers, n_warehouses, n_products, n_periods))
    
    wc_flow = best_solution[n_suppliers * n_warehouses * n_products * n_periods:]
    wc_flow = wc_flow.reshape((n_warehouses, n_customers, n_products, n_periods))
    
    return {
        'total_cost': best_cost,
        'supplier_warehouse_flow': sw_flow,
        'warehouse_customer_flow': wc_flow,
        'transportation_cost': np.sum(sw_flow * supplier_warehouse_cost[:, :, :, np.newaxis]) + 
                             np.sum(wc_flow * warehouse_customer_cost[:, :, :, np.newaxis])
    }
```

## Energy System Optimization

### Microgrid Design and Operation

```python
def microgrid_optimization():
    """
    Optimize the design and operation of a renewable energy microgrid.
    """
    # Time parameters
    n_hours = 24  # 24-hour optimization horizon
    time_step = 1  # hours
    
    # Component parameters
    pv_capex = 800  # $/kW
    wind_capex = 1200  # $/kW
    battery_capex = 300  # $/kWh
    diesel_capex = 800  # $/kW
    
    pv_opex = 20  # $/kW/year
    wind_opex = 30  # $/kW/year
    battery_opex = 10  # $/kWh/year
    diesel_opex = 0.1  # $/kWh
    diesel_fuel_cost = 1.0  # $/L
    
    # Technical parameters
    battery_eff = 0.95  # Round-trip efficiency
    battery_dod_max = 0.8  # Maximum depth of discharge
    diesel_min_load = 0.3  # Minimum load ratio
    
    # Load and renewable profiles (normalized)
    load_profile = np.array([
        0.5, 0.4, 0.4, 0.4, 0.5, 0.7,  # 00:00-05:00
        0.9, 1.0, 0.9, 0.8, 0.8, 0.8,  # 06:00-11:00
        0.9, 1.0, 1.0, 0.9, 0.9, 0.9,  # 12:00-17:00
        1.0, 1.1, 1.2, 1.1, 0.9, 0.7,  # 18:00-23:00
        0.6, 0.5  # 22:00-23:00
    ])
    
    pv_profile = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Night
        0.1, 0.3, 0.6, 0.8, 0.9, 1.0,  # Morning
        1.0, 0.9, 0.8, 0.7, 0.6, 0.5,  # Afternoon
        0.4, 0.2, 0.0, 0.0, 0.0, 0.0,  # Evening
        0.0, 0.0  # Night
    ])
    
    wind_profile = np.random.uniform(0.2, 1.0, n_hours)  # Simplified
    
    # Scale to actual values
    peak_load = 1000  # kW
    load = load_profile * peak_load  # kW
    
    def evaluate_design(x):
        """
        x = [pv_capacity, wind_capacity, battery_capacity, diesel_capacity]
        """
        pv_cap = x[0]
        wind_cap = x[1]
        batt_cap = x[2]
        diesel_cap = x[3]
        
        # Initialize variables
        soc = 0.5 * batt_cap  # Start at 50% state of charge
        total_diesel = 0
        load_curtailment = 0
        
        # Simulate operation
        for t in range(n_hours):
            # Calculate available generation
            pv_gen = pv_cap * pv_profile[t]
            wind_gen = wind_cap * wind_profile[t]
            
            # Calculate net load
            net_load = load[t] - pv_gen - wind_gen
            
            # Battery operation
            if net_load > 0:
                # Discharge battery
                discharge = min(net_load, soc * battery_eff, batt_cap * battery_dod_max)
                soc -= discharge / battery_eff
                net_load -= discharge
            else:
                # Charge battery
                charge = min(-net_load, (batt_cap - soc) * battery_eff)
                soc += charge
                net_load += charge
            
            # Diesel generator
            if net_load > 0:
                diesel_gen = min(max(net_load, diesel_min_load * diesel_cap), diesel_cap)
                total_diesel += diesel_gen * time_step
                net_load -= diesel_gen
            
            # Load curtailment (unmet load)
            if net_load > 0:
                load_curtailment += net_load * time_step
        
        # Calculate costs
        capital_cost = (
            pv_cap * pv_capex +
            wind_cap * wind_capex +
            batt_cap * battery_capex +
            diesel_cap * diesel_capex
        )
        
        opex = (
            pv_cap * pv_opex +
            wind_cap * wind_opex +
            batt_cap * battery_opex +
            total_diesel * diesel_opex
        )
        
        fuel_cost = total_diesel * diesel_fuel_cost
        
        # Penalty for load curtailment
        penalty = 1e6 * load_curtailment / np.sum(load)
        
        total_cost = capital_cost + opex + fuel_cost + penalty
        
        return total_cost
    
    # Define optimization bounds
    bounds = [
        (0, 2000),    # PV capacity (kW)
        (0, 2000),    # Wind capacity (kW)
        (0, 5000),    # Battery capacity (kWh)
        (0, 2000)     # Diesel capacity (kW)
    ]
    
    # Run optimization
    optimizer = CPO(
        dimensions=len(bounds),
        bounds=bounds,
        pop_size=30,
        max_iter=50,
        verbose=True
    )
    
    best_design, best_cost, _ = optimizer.optimize(evaluate_design)
    
    return {
        'pv_capacity_kw': best_design[0],
        'wind_capacity_kw': best_design[1],
        'battery_capacity_kwh': best_design[2],
        'diesel_capacity_kw': best_design[3],
        'total_cost': best_cost
    }
```

## Tips for Real-World Applications

1. **Start Simple**: Begin with a simplified version of your problem and gradually add complexity.

2. **Data Preparation**: Ensure your input data is properly scaled and preprocessed.

3. **Constraint Handling**: Use appropriate constraint handling techniques (penalty functions, repair operators, etc.).

4. **Parallelization**: For computationally expensive problems, leverage parallel evaluation of solutions.

5. **Visualization**: Create visualizations to understand the optimization process and results.

6. **Sensitivity Analysis**: Perform sensitivity analysis to understand how changes in parameters affect the solution.

7. **Validation**: Always validate optimization results with domain knowledge and real-world constraints.
