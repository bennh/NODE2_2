# Car Obstacle Avoidance — Direct Multiple Shooting OCP

This project solves an **obstacle avoidance manoeuvre problem** for a car using **Direct Multiple Shooting (DMS)** and **Sequential Quadratic Programming (SQP)**. It reproduces and extends the benchmark in Gerdts (2005), formulated as a continuous optimal control problem (OCP) with fixed gear and nonlinear track constraints.

---

## Project Objectives

- Model realistic car dynamics using a single-track model.
- Formulate the OCP with track boundaries via Bézier constraints.
- Discretize using **Direct Multiple Shooting**.
- Solve using:
  - CasADi’s built-in NLP solvers (IPOPT),
  - A custom implementation of **SQP with exact Hessian**.
- Visualize and compare results.

---

## Project Structure

```text
project2_car_ocp/
│
├── main.ipynb                    # Main notebook (problem setup, solve, visualize)
│
├── car_model.py                  # Car dynamics + CasADi integrator
├── track_constraints.py          # Bézier track constraints (Pl, Pu)
├── multiple_shooting.py          # DMS structure and constraints (F2, F3)
├── ocp_formulation.py            # Assembles full CasADi NLP problem
├── sqp_solver.py                 # Self-written SQP solver
│
└── utils.py                      # Plotting, solution unpacking, warm start
```

---

## Team Contributions

| Member           | Contribution                                                                                                                                     |
|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| **Yuefeiyang Li**   | Implemented the car dynamics model and RK4 integrator in `car_model.py`, and formulated the Bézier-based track constraints in `track_constraints.py` for Milestone 2. |
| **Binheng Zheng**   | Developed the multiple shooting structure in `multiple_shooting.py` and assembled the full OCP in `ocp_formulation.py` for Milestone 3, including in-notebook markdown and documentation. |
| **Siqing Fan**      | Designed and implemented the custom SQP solver in `sqp_solver.py` for Milestone 4, and conducted warm-start tuning, final testing, and trajectory visualization in `main.ipynb`. |
