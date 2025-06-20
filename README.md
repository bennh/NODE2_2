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

---

## Team Contributions

| Member     | Contribution                                                                                     |
|------------|--------------------------------------------------------------------------------------------------|
| **Fan, Siqing** | Built the custom SQP solver (`sqp_solver.py`) for Milestone 4, and handled initial guess generation, final testing, and result visualization in `main.ipynb`. |
| **Li, Yuefeiyang** | Developed the vehicle dynamics model and RK4 integrator (`car_model.py`), and implemented Bézier-based track constraints (`track_constraints.py`) for Milestone 2. |
| **Zheng, Binheng** | Implemented the multiple shooting scheme (`multiple_shooting.py`) and formulated the OCP structure (`ocp_formulation.py`) for Milestone 3. |
