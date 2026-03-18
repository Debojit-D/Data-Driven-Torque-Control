# https://mujoco.readthedocs.io/en/stable/computation/index.html#actuation-model

import mujoco
import mujoco.viewer
import numpy as np

from utils.data_driven_control.preprocess import hankel, check_gpe

XML_FILE_PATH = "robot_descriptions/franka_emika_panda/scene.xml"

# Constant user torques for the 7 Panda arm joints
tau_user = np.array([
    0.1,  # joint1
    -0.1,  # joint2
    0.2,  # joint3
    -0.2,  # joint4
    0.15,  # joint5
    -0.05,  # joint6
    0.1,   # joint7
    0.1
], dtype=np.float64)


def apply_torque_with_bias_comp(model, data, tau_user):
    """
    Apply MuJoCo bias compensation + constant user torque
    to the 7 Panda arm actuators.
    """
    data.ctrl[:] = 0.0

    for aid in range(8):
        joint_id = model.actuator_trnid[aid, 0]
        dof_id = model.jnt_dofadr[joint_id]

        tau_bias = data.qfrc_bias[dof_id]
        data.ctrl[aid] = tau_bias + tau_user[aid]


# Load model and data
model = mujoco.MjModel.from_xml_path(XML_FILE_PATH)
data = mujoco.MjData(model)

dt = model.opt.timestep
T_final = 10.0   # seconds
N = int(T_final / dt)
L = 100  # depth of Hankel matrix

m = model.nu          # number of actuators
p = model.nq          # number of joint positions

U = np.zeros((m, N))  # inputs
Y = np.zeros((p, N))  # outputs
time = np.zeros(N)

# Reset to home keyframe if present
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
if key_id != -1:
    mujoco.mj_resetDataKeyframe(model, data, key_id)

# Forward once to initialize dynamics
mujoco.mj_forward(model, data)

# Launch viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    for k in range(N):
        # Update dynamics for current state
        mujoco.mj_forward(model, data)

        # Apply bias compensation + user torque
        apply_torque_with_bias_comp(model, data, tau_user)

        U[:, k] = data.ctrl.copy()
        Y[:, k] = data.qpos.copy()
        time[k] = data.time

        # Step simulation
        mujoco.mj_step(model, data)

        # Refresh viewer
        viewer.sync()

print("Simulation completed.")
print("Input data shape (U):", U.shape)
print("Output data shape (Y):", Y.shape)

print("Checking persistence of excitation...")

H_U, H_Y = hankel(U, Y, L)
gpe_satisfied, rank_H_U = check_gpe(H_U)
print("GPE satisfied:", gpe_satisfied)
print("Rank of Hankel matrix for inputs (H_U):", rank_H_U)