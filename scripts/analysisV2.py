import json
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.data_driven_control.preprocess import check_gpe, hankel

XML_FILE_PATH = "/home/iitgn-robotics/Debojit_WS/Data-Driven-Torque-Control/robot_descriptions/franka_emika_panda/scene.xml"

# ============================================================
# User settings
# ============================================================

# Total duration of multisine excitation
EXCITATION_DURATION = 40.0  # seconds

# Smooth ramp-in / ramp-out duration
RAMP_TIME = 3.0  # seconds

# Joint-space PD gains
KP = np.array([20.0, 20.0, 20.0, 15.0, 10.0, 8.0, 5.0], dtype=np.float64)
KD = np.array([2.0,  2.0,  2.0,  1.5,  1.0,  0.8,  0.5], dtype=np.float64)

# Safety torque saturation
TAU_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float64)

# Multi-sine amplitudes in degrees about q_home
MULTISINE_AMPLITUDE_DEG = np.array([8.0, 8.0, 6.0, 6.0, 4.0, 4.0, 4.0], dtype=np.float64)

# Distinct non-harmonic frequencies in Hz for better persistence of excitation
MULTISINE_FREQUENCIES_HZ = np.array([0.17, 0.23, 0.31, 0.41, 0.53, 0.67, 0.83], dtype=np.float64)

# Phases in degrees
MULTISINE_PHASE_DEG = np.array([0.0, 35.0, 80.0, 125.0, 170.0, 215.0, 260.0], dtype=np.float64)

# Optional constant offsets from q_home in degrees
JOINT_CENTER_OFFSET_DEG = np.zeros(7, dtype=np.float64)

# Save dataset
SAVE_DATA = True
SAVE_PATH = "run_data.npz"
PLOTS_ROOT = os.path.join(PROJECT_ROOT, "plots")

# GPE / Hankel settings
RUN_GPE_CHECK = True
GPE_HANKEL_DEPTH = 200
PLOT_GPE = True

# Output choice for data-driven pipeline:
# False -> Y = q
# True  -> Y = [q; qd]
USE_VELOCITY_IN_OUTPUT = False


# ============================================================
# Utilities
# ============================================================
def get_joint_state(model, data):
    """Return the current 7 joint positions and velocities."""
    q = np.zeros(7, dtype=np.float64)
    qd = np.zeros(7, dtype=np.float64)

    for aid in range(7):
        joint_id = model.actuator_trnid[aid, 0]
        qpos_id = model.jnt_qposadr[joint_id]
        dof_id = model.jnt_dofadr[joint_id]

        q[aid] = data.qpos[qpos_id]
        qd[aid] = data.qvel[dof_id]

    return q, qd


def smooth_envelope(t, T, T_ramp):
    """
    Smooth amplitude envelope e(t) with smoothstep ramp-in and ramp-out.
    Returns:
        e   : envelope
        de  : first derivative
        dde : second derivative
    """
    if T_ramp <= 0.0:
        return 1.0, 0.0, 0.0

    if t <= 0.0:
        return 0.0, 0.0, 0.0

    if t < T_ramp:
        s = t / T_ramp
        e = 3.0 * s**2 - 2.0 * s**3
        de = (6.0 * s - 6.0 * s**2) / T_ramp
        dde = (6.0 - 12.0 * s) / (T_ramp**2)
        return e, de, dde

    if t <= T - T_ramp:
        return 1.0, 0.0, 0.0

    if t < T:
        s = (T - t) / T_ramp
        e = 3.0 * s**2 - 2.0 * s**3
        de = -(6.0 * s - 6.0 * s**2) / T_ramp
        dde = (6.0 - 12.0 * s) / (T_ramp**2)
        return e, de, dde

    return 0.0, 0.0, 0.0


class MultiSineJointPlanner:
    """
    Simultaneous multisine excitation around q_home:
        q_des(t) = q_home + q_center_offset + e(t) * A * sin(w t + phi)

    where e(t) is a smooth ramp-in/ramp-out envelope.
    """

    def __init__(self, q_home, amplitudes_deg, frequencies_hz, phases_deg, T, T_ramp,
                 center_offset_deg=None):
        self.q_home = np.asarray(q_home, dtype=np.float64).reshape(7)
        self.A = np.deg2rad(np.asarray(amplitudes_deg, dtype=np.float64).reshape(7))
        self.f = np.asarray(frequencies_hz, dtype=np.float64).reshape(7)
        self.w = 2.0 * np.pi * self.f
        self.phi = np.deg2rad(np.asarray(phases_deg, dtype=np.float64).reshape(7))
        self.T = float(T)
        self.T_ramp = float(T_ramp)

        if center_offset_deg is None:
            self.q_center_offset = np.zeros(7, dtype=np.float64)
        else:
            self.q_center_offset = np.deg2rad(
                np.asarray(center_offset_deg, dtype=np.float64).reshape(7)
            )

        if self.T <= 0.0:
            raise ValueError("EXCITATION_DURATION must be > 0")

        if self.T_ramp < 0.0:
            raise ValueError("RAMP_TIME must be >= 0")

    def evaluate(self, t):
        if t <= 0.0:
            q_des = self.q_home + self.q_center_offset
            return q_des.copy(), np.zeros(7), np.zeros(7)

        if t >= self.T:
            q_des = self.q_home + self.q_center_offset
            return q_des.copy(), np.zeros(7), np.zeros(7)

        e, de, dde = smooth_envelope(t, self.T, self.T_ramp)

        theta = self.w * t + self.phi
        s = np.sin(theta)
        c = np.cos(theta)

        q_offset = e * self.A * s
        qd_offset = self.A * (de * s + e * self.w * c)
        qdd_offset = self.A * (dde * s + 2.0 * de * self.w * c - e * (self.w**2) * s)

        q_des = self.q_home + self.q_center_offset + q_offset
        qd_des = qd_offset
        qdd_des = qdd_offset

        return q_des, qd_des, qdd_des


def apply_joint_pd_control(model, data, q_des, qd_des):
    """
    PD torque control with gravity/bias compensation:
        tau = qfrc_bias + Kp(q_des - q) + Kd(qd_des - qd)
    """
    data.ctrl[:] = 0.0

    tau_bias_all = np.zeros(7, dtype=np.float64)
    tau_kp_all = np.zeros(7, dtype=np.float64)
    tau_kd_all = np.zeros(7, dtype=np.float64)
    tau_cmd_all = np.zeros(7, dtype=np.float64)

    for aid in range(7):
        joint_id = model.actuator_trnid[aid, 0]
        qpos_id = model.jnt_qposadr[joint_id]
        dof_id = model.jnt_dofadr[joint_id]

        q = data.qpos[qpos_id]
        qd = data.qvel[dof_id]

        tau_bias = data.qfrc_bias[dof_id]

        pos_err = q_des[aid] - q
        vel_err = qd_des[aid] - qd

        tau_kp = KP[aid] * pos_err
        tau_kd = KD[aid] * vel_err
        tau_cmd = tau_bias + tau_kp + tau_kd
        tau_cmd = np.clip(tau_cmd, -TAU_LIMITS[aid], TAU_LIMITS[aid])

        data.ctrl[aid] = tau_cmd
        tau_bias_all[aid] = tau_bias
        tau_kp_all[aid] = tau_kp
        tau_kd_all[aid] = tau_kd
        tau_cmd_all[aid] = tau_cmd

    return tau_bias_all, tau_kp_all, tau_kd_all, tau_cmd_all


# ============================================================
# Plotting
# ============================================================
def plot_joint_tracking(time_hist, q_hist, q_des_hist):
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    axes = axes.flatten()

    for j in range(7):
        axes[j].plot(time_hist, np.rad2deg(q_hist[:, j]), label="Tracked", linewidth=2.0)
        axes[j].plot(time_hist, np.rad2deg(q_des_hist[:, j]), "--", label="Reference", linewidth=1.5)
        axes[j].set_ylabel(f"q{j + 1} (deg)")
        axes[j].grid(True, alpha=0.3)
        axes[j].legend(loc="best")

    axes[7].axis("off")
    axes[6].set_xlabel("Time (s)")
    fig.suptitle("Simultaneous Multi-Sine Excitation: Position Tracking")
    fig.tight_layout()
    return fig


def plot_joint_velocity_tracking(time_hist, qd_hist, qd_des_hist):
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    axes = axes.flatten()

    for j in range(7):
        axes[j].plot(time_hist, np.rad2deg(qd_hist[:, j]), label="Tracked", linewidth=2.0)
        axes[j].plot(time_hist, np.rad2deg(qd_des_hist[:, j]), "--", label="Reference", linewidth=1.5)
        axes[j].set_ylabel(f"qd{j + 1} (deg/s)")
        axes[j].grid(True, alpha=0.3)
        axes[j].legend(loc="best")

    axes[7].axis("off")
    axes[6].set_xlabel("Time (s)")
    fig.suptitle("Simultaneous Multi-Sine Excitation: Velocity Tracking")
    fig.tight_layout()
    return fig


def plot_torque_components(time_hist, tau_cmd_hist, tau_bias_hist, tau_kp_hist, tau_kd_hist):
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    axes = axes.flatten()

    for j in range(7):
        axes[j].plot(time_hist, tau_cmd_hist[:, j], label="Applied torque", linewidth=2.0)
        axes[j].plot(time_hist, tau_bias_hist[:, j], label="qfrc_bias", linewidth=1.2)
        axes[j].plot(time_hist, tau_kp_hist[:, j], label="Kp term", linewidth=1.2)
        axes[j].plot(time_hist, tau_kd_hist[:, j], label="Kd term", linewidth=1.2)
        axes[j].set_ylabel(f"tau{j + 1} (Nm)")
        axes[j].grid(True, alpha=0.3)
        axes[j].legend(loc="best")

    axes[7].axis("off")
    axes[6].set_xlabel("Time (s)")
    fig.suptitle("Simultaneous Multi-Sine Excitation: Torque Breakdown")
    fig.tight_layout()
    return fig


def plot_reference_spectrum_info():
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].bar(np.arange(1, 8), MULTISINE_AMPLITUDE_DEG)
    axes[0].set_ylabel("Amplitude (deg)")
    axes[0].set_title("Multi-Sine Amplitudes per Joint")
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(np.arange(1, 8), MULTISINE_FREQUENCIES_HZ)
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_xlabel("Joint index")
    axes[1].set_title("Multi-Sine Frequencies per Joint")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_gpe_singular_values(H_Y):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("Singular Values of H_Y")
    ax.loglog(np.linalg.svd(H_Y, compute_uv=False), marker="o")
    fig.tight_layout()
    return fig


def create_output_dir(script_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(PLOTS_ROOT, f"{script_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_figure(fig, output_dir, filename):
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")


def save_metadata(output_dir, metadata):
    path = os.path.join(output_dir, "run_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {path}")


# ============================================================
# Main
# ============================================================
def main():
    model = mujoco.MjModel.from_xml_path(XML_FILE_PATH)
    data = mujoco.MjData(model)

    # Reset to home pose if available
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)

    mujoco.mj_forward(model, data)

    q_home, _ = get_joint_state(model, data)

    planner = MultiSineJointPlanner(
        q_home=q_home,
        amplitudes_deg=MULTISINE_AMPLITUDE_DEG,
        frequencies_hz=MULTISINE_FREQUENCIES_HZ,
        phases_deg=MULTISINE_PHASE_DEG,
        T=EXCITATION_DURATION,
        T_ramp=RAMP_TIME,
        center_offset_deg=JOINT_CENTER_OFFSET_DEG,
    )

    print("Home configuration (deg):", np.rad2deg(q_home))
    print("Multi-sine amplitudes (deg):", MULTISINE_AMPLITUDE_DEG)
    print("Multi-sine frequencies (Hz):", MULTISINE_FREQUENCIES_HZ)
    print("Multi-sine phases (deg):", MULTISINE_PHASE_DEG)
    print("Center offsets (deg):", JOINT_CENTER_OFFSET_DEG)
    print("Excitation duration (s):", EXCITATION_DURATION)
    print("Ramp time (s):", RAMP_TIME)

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
        while viewer.is_running() and data.time <= EXCITATION_DURATION:
            t = data.time

            q_des, qd_des, qdd_des = planner.evaluate(t)

            mujoco.mj_forward(model, data)
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

            mujoco.mj_step(model, data)
            viewer.sync()

    if not time_hist:
        print("No simulation data recorded.")
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

    # Data-driven arrays
    # U shape: (m, N)
    # Y shape: (p, N)
    U = tau_cmd_hist.T  # (7, N)

    if USE_VELOCITY_IN_OUTPUT:
        Y = np.vstack((q_hist.T, qd_hist.T))  # (14, N)
    else:
        Y = q_hist.T  # (7, N)

    print("Collected data:")
    print("  U shape:", U.shape)
    print("  Y shape:", Y.shape)
    print("  Number of samples:", U.shape[1])

    output_dir = create_output_dir("analysisV2")
    gpe_result = {
        "checked": False,
        "hankel_depth": int(GPE_HANKEL_DEPTH),
        "plot_saved": False,
        "gpe_satisfied": None,
        "rank_H_U": None,
        "rank_H_Y": None,
        "message": "GPE check disabled.",
    }

    if RUN_GPE_CHECK:
        print("Checking persistence of excitation...")
        L = GPE_HANKEL_DEPTH
        num_samples = U.shape[1]

        if L > num_samples:
            message = f"Skipping GPE check: Hankel depth L={L} exceeds number of samples N={num_samples}."
            print(message)
            gpe_result["message"] = message
        else:
            H_U, H_Y = hankel(U, Y, L)
            gpe_satisfied, rank_H_U, rank_H_Y = check_gpe(H_U, H_Y, plot=False)
            print("GPE satisfied:", gpe_satisfied)
            print("Rank of Hankel matrix for inputs (H_U):", rank_H_U)
            print("Rank of Hankel matrix for outputs (H_Y):", rank_H_Y)
            gpe_result = {
                "checked": True,
                "hankel_depth": int(L),
                "plot_saved": bool(PLOT_GPE),
                "gpe_satisfied": bool(gpe_satisfied),
                "rank_H_U": int(rank_H_U),
                "rank_H_Y": int(rank_H_Y),
                "message": "GPE check completed.",
            }

            if PLOT_GPE:
                save_figure(plot_gpe_singular_values(H_Y), output_dir, "gpe_singular_values.png")

    if SAVE_DATA:
        save_path = os.path.join(output_dir, SAVE_PATH)
        np.savez(
            save_path,
            time=time_hist,
            U=U,
            Y=Y,
            q_hist=q_hist,
            q_des_hist=q_des_hist,
            qd_hist=qd_hist,
            qd_des_hist=qd_des_hist,
            tau_cmd_hist=tau_cmd_hist,
            tau_bias_hist=tau_bias_hist,
            tau_kp_hist=tau_kp_hist,
            tau_kd_hist=tau_kd_hist,
            q_home=q_home,
            amplitudes_deg=MULTISINE_AMPLITUDE_DEG,
            frequencies_hz=MULTISINE_FREQUENCIES_HZ,
            phases_deg=MULTISINE_PHASE_DEG,
            center_offset_deg=JOINT_CENTER_OFFSET_DEG,
            excitation_duration=EXCITATION_DURATION,
            ramp_time=RAMP_TIME,
            use_velocity_in_output=USE_VELOCITY_IN_OUTPUT,
            kp=KP,
            kd=KD,
            tau_limits=TAU_LIMITS,
        )
        print(f"Saved dataset to: {os.path.abspath(save_path)}")

    metadata = {
        "script": "analysisV2.py",
        "output_dir": output_dir,
        "xml_file_path": XML_FILE_PATH,
        "settings": {
            "excitation_duration": float(EXCITATION_DURATION),
            "ramp_time": float(RAMP_TIME),
            "multisine_amplitude_deg": MULTISINE_AMPLITUDE_DEG.tolist(),
            "multisine_frequencies_hz": MULTISINE_FREQUENCIES_HZ.tolist(),
            "multisine_phase_deg": MULTISINE_PHASE_DEG.tolist(),
            "joint_center_offset_deg": JOINT_CENTER_OFFSET_DEG.tolist(),
            "use_velocity_in_output": bool(USE_VELOCITY_IN_OUTPUT),
            "kp": KP.tolist(),
            "kd": KD.tolist(),
            "tau_limits": TAU_LIMITS.tolist(),
            "run_gpe_check": bool(RUN_GPE_CHECK),
            "gpe_hankel_depth": int(GPE_HANKEL_DEPTH),
            "save_data": bool(SAVE_DATA),
        },
        "results": {
            "num_samples": int(U.shape[1]),
            "U_shape": list(U.shape),
            "Y_shape": list(Y.shape),
            "home_configuration_deg": np.rad2deg(q_home).tolist(),
            "gpe": gpe_result,
        },
    }
    save_metadata(output_dir, metadata)

    save_figure(plot_joint_tracking(time_hist, q_hist, q_des_hist), output_dir, "position_tracking.png")
    save_figure(plot_joint_velocity_tracking(time_hist, qd_hist, qd_des_hist), output_dir, "velocity_tracking.png")
    save_figure(plot_torque_components(time_hist, tau_cmd_hist, tau_bias_hist, tau_kp_hist, tau_kd_hist), output_dir, "torque_breakdown.png")
    save_figure(plot_reference_spectrum_info(), output_dir, "reference_spectrum.png")


if __name__ == "__main__":
    main()
