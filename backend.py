from fastapi import FastAPI
import numpy as np

app = FastAPI()

@app.post("/fk")
def forward_kinematics(data: dict):
    dh = data["dh"]
    T = np.eye(4)

    for a, alpha, d, theta in dh:
        T = T @ np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])

    return {"matrix": T.tolist()}
