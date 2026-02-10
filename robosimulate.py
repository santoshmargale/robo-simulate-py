import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Robotics Expert System", layout="wide")

# -------------------------------------------------
# Session State Initialization
# -------------------------------------------------
if "dh_params" not in st.session_state:
    st.session_state.dh_params = None
if "fk_result" not in st.session_state:
    st.session_state.fk_result = None
if "ik_result" not in st.session_state:
    st.session_state.ik_result = None
if "fd_result" not in st.session_state:
    st.session_state.fd_result = None
if "id_result" not in st.session_state:
    st.session_state.id_result = None

# ---------------- Navigation ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Module",
    [
        "Robot Simulator",
        "Forward Kinematics",
        "Inverse Kinematics",
        "Forward Dynamics",
        "Inverse Dynamics",
    ],
)

# ---------------- DH Functions ----------------
def dh_transform(a, alpha, d, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),               np.cos(alpha),               d],
        [0,              0,                           0,                           1],
    ])

def forward_kinematics(dh_params):
    T = np.eye(4)
    positions = [(0, 0, 0)]
    matrices = []
    for p in dh_params:
        T = T @ dh_transform(*p)
        matrices.append(T.copy())
        positions.append((T[0, 3], T[1, 3], T[2, 3]))
    return positions, T, matrices

def plot_robot_3d(positions, matrices):
    fig = go.Figure()
    x, y, z = zip(*positions)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines+markers"))

    for T in matrices:
        origin = T[:3, 3]
        scale = 0.25
        for axis, color in zip([T[:3,0], T[:3,1], T[:3,2]], ["red","green","blue"]):
            end = origin + axis * scale
            fig.add_trace(go.Scatter3d(
                x=[origin[0], end[0]],
                y=[origin[1], end[1]],
                z=[origin[2], end[2]],
                mode="lines",
                line=dict(color=color),
                showlegend=False
            ))
    fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"), height=550)
    return fig

# =========================================================
# ROBOT SIMULATOR (NO JOINT INPUT HERE)
# =========================================================
if page == "Robot Simulator":

    st.title("Robot Simulator")

    if st.session_state.dh_params is None:
        st.warning("No robot configuration available. Please define joints in FK / IK modules.")
    else:
        positions, _, matrices = forward_kinematics(st.session_state.dh_params)
        st.plotly_chart(plot_robot_3d(positions, matrices), use_container_width=True)

    st.subheader("Imported Results")

    if st.session_state.fk_result is not None:
        st.markdown("Forward Kinematics")
        st.dataframe(pd.DataFrame(st.session_state.fk_result))

    if st.session_state.ik_result is not None:
        st.markdown("Inverse Kinematics")
        st.json(st.session_state.ik_result)

    if st.session_state.fd_result is not None:
        st.markdown("Forward Dynamics")
        st.json(st.session_state.fd_result)

    if st.session_state.id_result is not None:
        st.markdown("Inverse Dynamics")
        st.json(st.session_state.id_result)

# =========================================================
# FORWARD KINEMATICS
# =========================================================
elif page == "Forward Kinematics":

    st.title("Forward Kinematics")

    dof = st.slider("Number of Joints", 2, 10, 3)
    dh_params = []

    for i in range(dof):
        joint_type = st.selectbox(f"Joint {i+1} Type", ["Revolute", "Prismatic"], key=f"fk_type{i}")
        a = st.number_input(f"a{i+1}", 1.0, key=f"fka{i}")
        alpha = np.deg2rad(st.number_input(f"alpha{i+1} (deg)", 0.0, key=f"fkal{i}"))

        if joint_type == "Revolute":
            d = st.number_input(f"d{i+1}", 0.5, key=f"fkd{i}")
            theta = np.deg2rad(st.number_input(f"theta{i+1} (deg)", 0.0, key=f"fkt{i}"))
        else:
            d = st.slider(f"d{i+1}", 0.0, 5.0, 0.5, key=f"fkp{i}")
            theta = 0.0

        dh_params.append([a, alpha, d, theta])

    _, T, _ = forward_kinematics(dh_params)
    st.session_state.dh_params = dh_params
    st.session_state.fk_result = T
    st.dataframe(pd.DataFrame(T))

# =========================================================
# INVERSE KINEMATICS (2-LINK)
# =========================================================
elif page == "Inverse Kinematics":

    st.title("Inverse Kinematics (2-Link Planar)")

    L1 = st.number_input("Link 1 Length", 1.0)
    L2 = st.number_input("Link 2 Length", 1.0)
    x = st.number_input("Target X", 1.0)
    y = st.number_input("Target Y", 1.0)

    D = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)

    if abs(D) <= 1:
        theta2 = np.arccos(D)
        theta1 = np.arctan2(y, x) - np.arctan2(L2*np.sin(theta2), L1+L2*np.cos(theta2))

        st.session_state.ik_result = {
            "theta1_deg": np.rad2deg(theta1),
            "theta2_deg": np.rad2deg(theta2),
            "x": x,
            "y": y,
        }

        st.success("IK Solution Found")
    else:
        st.error("Target outside workspace")

# =========================================================
# FORWARD DYNAMICS
# =========================================================
elif page == "Forward Dynamics":

    st.title("Forward Dynamics")

    joint_type = st.selectbox("Joint Type", ["Revolute", "Prismatic"])
    m = st.number_input("Mass (kg)", 1.0)
    input_val = st.number_input("Input (Torque / Force)", 10.0)

    if m > 0:
        acc = input_val / m
        st.session_state.fd_result = {
            "joint_type": joint_type,
            "mass": m,
            "input": input_val,
            "acceleration": acc,
        }
        st.success(f"Acceleration = {acc:.2f}")

# =========================================================
# INVERSE DYNAMICS
# =========================================================
elif page == "Inverse Dynamics":

    st.title("Inverse Dynamics")

    joint_type = st.selectbox("Joint Type", ["Revolute", "Prismatic"])
    m = st.number_input("Mass (kg)", 1.0)
    acc = st.number_input("Desired Acceleration", 5.0)

    if m > 0:
        force = m * acc
        st.session_state.id_result = {
            "joint_type": joint_type,
            "mass": m,
            "acceleration": acc,
            "required_input": force,
        }
        st.success(f"Required Input = {force:.2f}")
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Robotics Expert System", layout="wide")

# -------------------------------------------------
# Session State Initialization
# -------------------------------------------------
if "dh_params" not in st.session_state:
    st.session_state.dh_params = None
if "fk_result" not in st.session_state:
    st.session_state.fk_result = None
if "ik_result" not in st.session_state:
    st.session_state.ik_result = None
if "fd_result" not in st.session_state:
    st.session_state.fd_result = None
if "id_result" not in st.session_state:
    st.session_state.id_result = None

# ---------------- Navigation ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Module",
    [
        "Robot Simulator",
        "Forward Kinematics",
        "Inverse Kinematics",
        "Forward Dynamics",
        "Inverse Dynamics",
    ],
)

# ---------------- DH Functions ----------------
def dh_transform(a, alpha, d, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),               np.cos(alpha),               d],
        [0,              0,                           0,                           1],
    ])

def forward_kinematics(dh_params):
    T = np.eye(4)
    positions = [(0, 0, 0)]
    matrices = []
    for p in dh_params:
        T = T @ dh_transform(*p)
        matrices.append(T.copy())
        positions.append((T[0, 3], T[1, 3], T[2, 3]))
    return positions, T, matrices

def plot_robot_3d(positions, matrices):
    fig = go.Figure()
    x, y, z = zip(*positions)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines+markers"))

    for T in matrices:
        origin = T[:3, 3]
        scale = 0.25
        for axis, color in zip([T[:3,0], T[:3,1], T[:3,2]], ["red","green","blue"]):
            end = origin + axis * scale
            fig.add_trace(go.Scatter3d(
                x=[origin[0], end[0]],
                y=[origin[1], end[1]],
                z=[origin[2], end[2]],
                mode="lines",
                line=dict(color=color),
                showlegend=False
            ))
    fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"), height=550)
    return fig

# =========================================================
# ROBOT SIMULATOR (NO JOINT INPUT HERE)
# =========================================================
if page == "Robot Simulator":

    st.title("Robot Simulator")

    if st.session_state.dh_params is None:
        st.warning("No robot configuration available. Please define joints in FK / IK modules.")
    else:
        positions, _, matrices = forward_kinematics(st.session_state.dh_params)
        st.plotly_chart(plot_robot_3d(positions, matrices), use_container_width=True)

    st.subheader("Imported Results")

    if st.session_state.fk_result is not None:
        st.markdown("Forward Kinematics")
        st.dataframe(pd.DataFrame(st.session_state.fk_result))

    if st.session_state.ik_result is not None:
        st.markdown("Inverse Kinematics")
        st.json(st.session_state.ik_result)

    if st.session_state.fd_result is not None:
        st.markdown("Forward Dynamics")
        st.json(st.session_state.fd_result)

    if st.session_state.id_result is not None:
        st.markdown("Inverse Dynamics")
        st.json(st.session_state.id_result)

# =========================================================
# FORWARD KINEMATICS
# =========================================================
elif page == "Forward Kinematics":

    st.title("Forward Kinematics")

    dof = st.slider("Number of Joints", 2, 10, 3)
    dh_params = []

    for i in range(dof):
        joint_type = st.selectbox(f"Joint {i+1} Type", ["Revolute", "Prismatic"], key=f"fk_type{i}")
        a = st.number_input(f"a{i+1}", 1.0, key=f"fka{i}")
        alpha = np.deg2rad(st.number_input(f"alpha{i+1} (deg)", 0.0, key=f"fkal{i}"))

        if joint_type == "Revolute":
            d = st.number_input(f"d{i+1}", 0.5, key=f"fkd{i}")
            theta = np.deg2rad(st.number_input(f"theta{i+1} (deg)", 0.0, key=f"fkt{i}"))
        else:
            d = st.slider(f"d{i+1}", 0.0, 5.0, 0.5, key=f"fkp{i}")
            theta = 0.0

        dh_params.append([a, alpha, d, theta])

    _, T, _ = forward_kinematics(dh_params)
    st.session_state.dh_params = dh_params
    st.session_state.fk_result = T
    st.dataframe(pd.DataFrame(T))

# =========================================================
# INVERSE KINEMATICS (2-LINK)
# =========================================================
elif page == "Inverse Kinematics":

    st.title("Inverse Kinematics (2-Link Planar)")

    L1 = st.number_input("Link 1 Length", 1.0)
    L2 = st.number_input("Link 2 Length", 1.0)
    x = st.number_input("Target X", 1.0)
    y = st.number_input("Target Y", 1.0)

    D = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)

    if abs(D) <= 1:
        theta2 = np.arccos(D)
        theta1 = np.arctan2(y, x) - np.arctan2(L2*np.sin(theta2), L1+L2*np.cos(theta2))

        st.session_state.ik_result = {
            "theta1_deg": np.rad2deg(theta1),
            "theta2_deg": np.rad2deg(theta2),
            "x": x,
            "y": y,
        }

        st.success("IK Solution Found")
    else:
        st.error("Target outside workspace")

# =========================================================
# FORWARD DYNAMICS
# =========================================================
elif page == "Forward Dynamics":

    st.title("Forward Dynamics")

    joint_type = st.selectbox("Joint Type", ["Revolute", "Prismatic"])
    m = st.number_input("Mass (kg)", 1.0)
    input_val = st.number_input("Input (Torque / Force)", 10.0)

    if m > 0:
        acc = input_val / m
        st.session_state.fd_result = {
            "joint_type": joint_type,
            "mass": m,
            "input": input_val,
            "acceleration": acc,
        }
        st.success(f"Acceleration = {acc:.2f}")

# =========================================================
# INVERSE DYNAMICS
# =========================================================
elif page == "Inverse Dynamics":

    st.title("Inverse Dynamics")

    joint_type = st.selectbox("Joint Type", ["Revolute", "Prismatic"])
    m = st.number_input("Mass (kg)", 1.0)
    acc = st.number_input("Desired Acceleration", 5.0)

    if m > 0:
        force = m * acc
        st.session_state.id_result = {
            "joint_type": joint_type,
            "mass": m,
            "acceleration": acc,
            "required_input": force,
        }
        st.success(f"Required Input = {force:.2f}")
