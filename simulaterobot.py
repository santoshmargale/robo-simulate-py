import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go

st.set_page_config(page_title="Robot Kinematics Expert System", layout="wide")
st.title("🤖 Advanced Robot Kinematics Expert System")

# ---------------- DH Transform ----------------
def dh_transform(a, alpha, d, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])

def forward_kinematics(dh_params):
    T = np.eye(4)
    positions = [(0,0,0)]
    matrices = []
    for p in dh_params:
        T = T @ dh_transform(*p)
        matrices.append(T.copy())
        positions.append((T[0,3], T[1,3], T[2,3]))
    return positions, T, matrices

# ---------------- 3D Robot Plot ----------------
def plot_robot_3d(positions, trail=None):
    x, y, z = zip(*positions)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines+markers', marker=dict(size=5), line=dict(width=6)))
    fig.add_trace(go.Scatter3d(x=[x[-1]], y=[y[-1]], z=[z[-1]], mode='markers', marker=dict(size=8)))

    if trail:
        tx, ty, tz = zip(*trail)
        fig.add_trace(go.Scatter3d(x=tx, y=ty, z=tz, mode='lines', line=dict(dash='dot')))

    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                      margin=dict(l=0, r=0, b=0, t=30), height=600)
    return fig

# ---------------- Trajectory Plot ----------------
def plot_trajectory(trail):
    if len(trail) < 2:
        return None
    tx, ty, tz = zip(*trail)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=tx, y=ty, z=tz, mode='lines+markers'))
    fig.update_layout(title="End Effector Trajectory")
    return fig

# ---------------- Robot Setup ----------------
st.sidebar.header("⚙ Robot Configuration")
dof = st.sidebar.slider("Select DOF", 2, 12, 3)

if "prev_dof" not in st.session_state or st.session_state.prev_dof != dof:
    st.session_state.prev_dof = dof
    st.session_state.joint_types = ["Revolute"] * dof
    st.session_state.dh_table = [[1.0, 0.0, 0.5, 0.0] for _ in range(dof)]
    st.session_state.recording = []

joint_types = []
dh_table = []

st.sidebar.subheader("Define Joints")
for i in range(dof):
    jt = st.sidebar.selectbox(f"Joint {i+1} Type", ["Revolute", "Prismatic"], key=f"type{i}")
    joint_types.append(jt)

st.sidebar.subheader("DH Parameters")
for i in range(dof):
    a = st.sidebar.number_input(f"a{i+1}", value=st.session_state.dh_table[i][0], key=f"a{i}")
    alpha = st.sidebar.number_input(f"alpha{i+1} (deg)", value=np.rad2deg(st.session_state.dh_table[i][1]), key=f"alpha{i}")
    d = st.sidebar.number_input(f"d{i+1}", value=st.session_state.dh_table[i][2], key=f"d{i}")
    theta = st.sidebar.number_input(f"theta{i+1} (deg)", value=np.rad2deg(st.session_state.dh_table[i][3]), key=f"theta{i}")
    st.session_state.dh_table[i] = [a, np.deg2rad(alpha), d, np.deg2rad(theta)]
    dh_table.append(st.session_state.dh_table[i].copy())

# ---------------- Joint Sliders ----------------
st.header("🎮 Move Robot Joints")
joint_values = []

for i in range(dof):
    if joint_types[i] == "Revolute":
        val = st.slider(f"θ{i+1} (deg)", -180, 180, 0, key=f"slider_theta_{i}")
        st.session_state.dh_table[i][3] = np.deg2rad(val)
    else:
        val = st.slider(f"d{i+1}", 0.0, 5.0, 0.5, key=f"slider_d_{i}")
        st.session_state.dh_table[i][2] = val
    joint_values.append(val)

positions, T, matrices = forward_kinematics(st.session_state.dh_table)

# ---------------- Visualizations ----------------
st.subheader("🦾 3D Robot Visualization")
trail = [step["End Effector"] for step in st.session_state.get("recording", [])]
st.plotly_chart(plot_robot_3d(positions, trail), use_container_width=True)

st.subheader("📍 End Effector Position")
st.write(f"X={positions[-1][0]:.2f}, Y={positions[-1][1]:.2f}, Z={positions[-1][2]:.2f}")

st.subheader("📐 End Effector Matrix")
st.dataframe(pd.DataFrame(T))

# ---------------- Recording ----------------
if "recording" not in st.session_state:
    st.session_state.recording = []

if st.button("🔴 Record Step"):
    st.session_state.recording.append({"Joint Values": joint_values.copy(), "End Effector": positions[-1], "Matrix": T.tolist()})
    st.success("Step Recorded!")

# ---------------- Trajectory Plot ----------------
if st.session_state.recording:
    st.subheader("📈 End Effector Trajectory")
    st.plotly_chart(plot_trajectory(trail), use_container_width=True)

# ---------------- Export CSV ----------------
if st.button("💾 Export Motion to CSV"):
    df = pd.DataFrame(st.session_state.recording)
    df.to_csv("robot_motion_data.csv", index=False)
    st.success("Motion data saved as robot_motion_data.csv")

# ---------------- Workspace Sampling ----------------
if st.button("🧭 Show Workspace Sample"):
    samples = []
    for t1 in np.linspace(-90, 90, 10):
        for t2 in np.linspace(-90, 90, 10):
            test_dh = [[1,0,0.5,np.deg2rad(t1)], [1,0,0.5,np.deg2rad(t2)]]
            pos,_,_ = forward_kinematics(test_dh)
            samples.append(pos[-1])
    st.plotly_chart(plot_trajectory(samples), use_container_width=True)

# ---------------- Simple Inverse Kinematics Demo ----------------
st.subheader("🎯 Simple 2-Link Inverse Kinematics Demo")
x_target = st.number_input("Target X", value=1.0)
y_target = st.number_input("Target Y", value=1.0)

L1, L2 = 1.0, 1.0
D = (x_target**2 + y_target**2 - L1**2 - L2**2) / (2*L1*L2)

if abs(D) <= 1:
    theta2 = np.arccos(D)
    theta1 = np.arctan2(y_target, x_target) - np.arctan2(L2*np.sin(theta2), L1+L2*np.cos(theta2))
    st.write(f"θ1 = {np.rad2deg(theta1):.2f}°, θ2 = {np.rad2deg(theta2):.2f}°")
else:
    st.warning("Target is outside reachable workspace.")

# ---------------- Expert System ----------------
st.subheader("🧠 Expert System Analysis")
feedback = []

if dof < 3: feedback.append("⚠ Limited workspace (<3 DOF)")
if dof >= 6: feedback.append("✅ Suitable for complex tasks")

for i, jt in enumerate(joint_types):
    if jt == "Revolute" and abs(joint_values[i]) > 170:
        feedback.append(f"⚠ Revolute Joint {i+1} near limit")
    if jt == "Prismatic" and joint_values[i] > 4.5:
        feedback.append(f"⚠ Prismatic Joint {i+1} near extension")

for f in feedback:
    st.write(f)
