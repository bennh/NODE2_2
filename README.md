# NODE2_2
```text
project2_car_ocp/
│
├── README.md                     # 项目说明文档
│
├── main.ipynb                    # Jupyter主Notebook，结果展示与交互
│
├── car_model.py                  # 汽车动力学模型与积分器
├── track_constraints.py          # 赛道Bezier边界约束
├── multiple_shooting.py          # 多重打靶离散化
├── ocp_formulation.py            # OCP组装与CasADi问题描述
├── sqp_solver.py                 # 自实现的SQP算法
│
├── utils.py                      # 通用工具函数（如初始化、可视化）
│
├── data/                         # 存放参数表、赛道数据等
├── results/                      # 保存仿真结果、图片等
└── figures/                      # 保存用于presentation的图片
