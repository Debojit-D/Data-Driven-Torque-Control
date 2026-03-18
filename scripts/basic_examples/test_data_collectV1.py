import time
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

XML_FILE_PATH = "/home/iitgn-robotics/Debojit_WS/Data-Driven-Torque-Control/robot_descriptions/franka_emika_panda/scene.xml"

# ==============================
# User settings
# ==============================
# Target joint configuration in degrees
TARGET_Q_DEG = np.array([
    107.18,
    -68.6479,
    11.4592,
    -114.5916,
    85.7296,
    97.4028,
    45.8366
], dtype=np.float64).reshape(7)
TARGET_Q = np.deg2rad(TARGET_Q_DEG)

TRAJ_DURATION = 5.0   # seconds

# PD gains (joint-space)
KP = np.array([20.0, 20.0, 20.0, 15.0, 10.0, 8.0, 5.0], dtype=np.float64)
KD = np.array([2.0,  2.0,  2.0,  1.5,  1.0,  0.8,  0.5], dtype=np.float64)

# Safety torque saturation
TAU_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float64)


def get_joint_state(model, data):
    """Return the current 7 joint positions and velocities."""
    q = np.zeros(7, dtype=np.float64)
    qd = np.zeros(7, dtype=np.float64)

    for aid in range(7):
        joint_id = model.actuator_trnid[aid, 0]
        dof_id = model.jnt_dofadr[joint_id]
        q[aid] = data.qpos[dof_id]
        qd[aid] = data.qvel[dof_id]

    return q, qd


# ==============================
# Quintic trajectory planner
# ==============================
class QuinticJointPlanner:
    """
    Quintic planner for joint-space motion:
    q(0)   = q0
    q(T)   = qf
    qd(0)  = 0
    qd(T)  = 0
    qdd(0) = 0
    qdd(T) = 0
    """

    def __init__(self, q0, qf, T):
        self.q0 = np.asarray(q0, dtype=np.float64).reshape(7)
        self.qf = np.asarray(qf, dtype=np.float64).reshape(7)
        self.T = float(T)

        if self.T <= 0.0:
            raise ValueError("Trajectory duration T must be > 0")

        self.dq = self.qf - self.q0

    def evaluate(self, t):
        """
        Returns:
            q_des   : desired joint position (7,)
            qd_des  : desired joint velocity (7,)
            qdd_des : desired joint acceleration (7,)
        """
        if t <= 0.0:
            return self.q0.copy(), np.zeros(7), np.zeros(7)

        if t >= self.T:
            return self.qf.copy(), np.zeros(7), np.zeros(7)

        s = t / self.T

        # Quintic polynomial in normalized time
        s2 = s * s
        s3 = s2 * s
        s4 = s3 * s
        s5 = s4 * s

        pos_scale = 10.0 * s3 - 15.0 * s4 + 6.0 * s5
        vel_scale = (30.0 * s2 - 60.0 * s3 + 30.0 * s4) / self.T
        acc_scale = (60.0 * s - 180.0 * s2 + 120.0 * s3) / (self.T ** 2)

        q_des = self.q0 + self.dq * pos_scale
        qd_des = self.dq * vel_scale
        qdd_des = self.dq * acc_scale

        return q_des, qd_des, qdd_des


# ==============================
# Controller
# ==============================
def apply_joint_pd_control(model, data, q_des, qd_des):
    """
    PD torque control with gravity/bias compensation:
        tau = qfrc_bias + Kp(q_des - q) + Kd(qd_des - qd)

    Assumes first 7 actuators correspond to first 7 joints.
    """
    data.ctrl[:] = 0.0
    tau_bias_all = np.zeros(7, dtype=np.float64)
    tau_kp_all = np.zeros(7, dtype=np.float64)
    tau_kd_all = np.zeros(7, dtype=np.float64)
    tau_cmd_all = np.zeros(7, dtype=np.float64)

    for aid in range(7):
        joint_id = model.actuator_trnid[aid, 0]
        dof_id = model.jnt_dofadr[joint_id]

        q = data.qpos[dof_id]
        qd = data.qvel[dof_id]

        tau_bias = data.qfrc_bias[dof_id]

        pos_err = q_des[aid] - q
        vel_err = qd_des[aid] - qd

        tau_kp = KP[aid] * pos_err
        tau_kd = KD[aid] * vel_err
        tau_cmd = tau_bias + tau_kp + tau_kd

        # Saturation for safety
        tau_cmd = np.clip(tau_cmd, -TAU_LIMITS[aid], TAU_LIMITS[aid])

        data.ctrl[aid] = tau_cmd
        tau_bias_all[aid] = tau_bias
        tau_kp_all[aid] = tau_kp
        tau_kd_all[aid] = tau_kd
        tau_cmd_all[aid] = tau_cmd

    return tau_bias_all, tau_kp_all, tau_kd_all, tau_cmd_all


def plot_joint_tracking(time_hist, q_hist, q_des_hist):
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    axes = axes.flatten()

    for j in range(7):
        axes[j].plot(time_hist, np.rad2deg(q_hist[:, j]), label="Tracked", linewidth=2.0)
        axes[j].plot(
            time_hist,
            np.rad2deg(q_des_hist[:, j]),
            "--",
            label="Reference",
            linewidth=1.5,
        )
        axes[j].set_ylabel(f"q{j + 1} (deg)")
        axes[j].grid(True, alpha=0.3)
        axes[j].legend(loc="best")

    axes[7].axis("off")
    axes[6].set_xlabel("Time (s)")
    fig.suptitle("Joint Angle Tracking")
    fig.tight_layout()


def plot_joint_velocity_tracking(time_hist, qd_hist, qd_des_hist):
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    axes = axes.flatten()

    for j in range(7):
        axes[j].plot(time_hist, np.rad2deg(qd_hist[:, j]), label="Tracked", linewidth=2.0)
        axes[j].plot(
            time_hist,
            np.rad2deg(qd_des_hist[:, j]),
            "--",
            label="Reference",
            linewidth=1.5,
        )
        axes[j].set_ylabel(f"qd{j + 1} (deg/s)")
        axes[j].grid(True, alpha=0.3)
        axes[j].legend(loc="best")

    axes[7].axis("off")
    axes[6].set_xlabel("Time (s)")
    fig.suptitle("Joint Velocity Tracking")
    fig.tight_layout()


def plot_torque_components(time_hist, tau_cmd_hist, tau_bias_hist, tau_kp_hist, tau_kd_hist):
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    axes = axes.flatten()

    for j in range(7):
        axes[j].plot(time_hist, tau_cmd_hist[:, j], label="Applied torque", linewidth=2.0)
        axes[j].plot(time_hist, tau_bias_hist[:, j], label="qfrc_bias", linewidth=1.5)
        axes[j].plot(time_hist, tau_kp_hist[:, j], label="Kp term", linewidth=1.5)
        axes[j].plot(time_hist, tau_kd_hist[:, j], label="Kd term", linewidth=1.5)
        axes[j].set_ylabel(f"tau{j + 1} (Nm)")
        axes[j].grid(True, alpha=0.3)
        axes[j].legend(loc="best")

    axes[7].axis("off")
    axes[6].set_xlabel("Time (s)")
    fig.suptitle("Torque Breakdown")
    fig.tight_layout()


# ==============================
# Main
# ==============================
def main():
    model = mujoco.MjModel.from_xml_path(XML_FILE_PATH)
    data = mujoco.MjData(model)

    # Reset to home pose if available
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)

    mujoco.mj_forward(model, data)

    # Initial joint configuration = planner start
    q0, _ = get_joint_state(model, data)

    planner = QuinticJointPlanner(q0=q0, qf=TARGET_Q, T=TRAJ_DURATION)

    print("Initial q0 (rad):", q0)
    print("Target  qf (deg):", TARGET_Q_DEG)
    print("Target  qf (rad):", TARGET_Q)
    print("Trajectory duration:", TRAJ_DURATION, "s")

    start_wall_time = time.time()
    time_hist = []
    q_hist = []
    q_des_hist = []
    qd_hist = []
    qd_des_hist = []
    tau_cmd_hist = []
    tau_bias_hist = []
    tau_kp_hist = []
    tau_kd_hist = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t = time.time() - start_wall_time

            # Desired trajectory from quintic planner
            q_des, qd_des, qdd_des = planner.evaluate(t)

            # Compute dynamics terms at current state
            mujoco.mj_forward(model, data)

            # Apply controller
            q, qd = get_joint_state(model, data)
            tau_bias, tau_kp, tau_kd, tau_cmd = apply_joint_pd_control(model, data, q_des, qd_des)

            time_hist.append(t)
            q_hist.append(q.copy())
            q_des_hist.append(q_des.copy())
            qd_hist.append(qd.copy())
            qd_des_hist.append(qd_des.copy())
            tau_cmd_hist.append(tau_cmd.copy())
            tau_bias_hist.append(tau_bias.copy())
            tau_kp_hist.append(tau_kp.copy())
            tau_kd_hist.append(tau_kd.copy())

            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()

            # Optional: print final status once trajectory completes
            if t >= TRAJ_DURATION:
                pass

    if not time_hist:
        print("No simulation data recorded, skipping plots.")
        return

    time_hist = np.asarray(time_hist, dtype=np.float64)
    q_hist = np.asarray(q_hist, dtype=np.float64)
    q_des_hist = np.asarray(q_des_hist, dtype=np.float64)
    qd_hist = np.asarray(qd_hist, dtype=np.float64)
    qd_des_hist = np.asarray(qd_des_hist, dtype=np.float64)
    tau_cmd_hist = np.asarray(tau_cmd_hist, dtype=np.float64)
    tau_bias_hist = np.asarray(tau_bias_hist, dtype=np.float64)
    tau_kp_hist = np.asarray(tau_kp_hist, dtype=np.float64)
    tau_kd_hist = np.asarray(tau_kd_hist, dtype=np.float64)

    plot_joint_tracking(time_hist, q_hist, q_des_hist)
    plot_joint_velocity_tracking(time_hist, qd_hist, qd_des_hist)
    plot_torque_components(time_hist, tau_cmd_hist, tau_bias_hist, tau_kp_hist, tau_kd_hist)
    plt.show()


if __name__ == "__main__":
    main()
