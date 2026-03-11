import numpy as np
import mujoco
from utils.mj_velocity_control.mj_velocity_ctrl import JointVelocityController


class DLSVelocityPlanner:
    """
    Joint-torque planner based on damped least-squares IK (Levenberg–Marquardt).
    Public API
    ----------
    reach_pose(target_pos, target_quat=None)
        – Drive EE to `target_pos`; if `target_quat` is supplied (x y z w),
          also track the full orientation.  Otherwise only align local z
          with world −z.

    track_twist(v_cart, w_cart=None)
        – Map a desired 6-D twist to torques.

    All MuJoCo quaternions (stored as [w x y z]) are automatically converted
    to [x y z w] internally.
    """

    # --------------------------- static helpers --------------------------- #
    @staticmethod
    def _wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
        """Convert MuJoCo order [w x y z] → [x y z w]."""
        q = np.asarray(q_wxyz)
        return np.array([q[1], q[2], q[3], q[0]])

    @staticmethod
    def _quat_log_error(q_t: np.ndarray, q_c: np.ndarray) -> np.ndarray:
        """
        Quaternion logarithmic error (axis-angle 3-vector) between target
        q_t and current q_c.  Convention: quaternions are [x y z w].
        """
        # q_err = q_t ⊗ q_c⁻¹
        q_c_inv = np.array([-q_c[0], -q_c[1], -q_c[2], q_c[3]])
        x1, y1, z1, w1 = q_t
        x2, y2, z2, w2 = q_c_inv
        q_e = np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])

        # Hemisphere continuity
        if q_e[3] < 0.0:
            q_e *= -1.0

        ang = 2.0 * np.arccos(np.clip(q_e[3], -1.0, 1.0))
        if ang < 1e-6:
            return np.zeros(3)
        axis = q_e[:3] / np.sin(ang / 2.0)
        return ang * axis

    # ------------------------------ init ---------------------------------- #
    def __init__(self,
                 model,
                 data,
                 kd: float = 5.0,
                 site_name: str = "right_center",
                 damping: float = 1e-2,
                 gripper_cfg: list[dict] | None = None):
        self.model     = model
        self.data      = data
        self.site_name = site_name
        self.damping   = damping
        self.site_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                           site_name)

        # Extract actuator IDs from gripper_cfg
        if gripper_cfg is not None:
            gripper_ids = {g["actuator_id"] for g in gripper_cfg}
        else:
            gripper_ids = set()

        self.ctrl = JointVelocityController(model, data, kd=kd, gripper_ids=gripper_ids)

        # Re-usable buffers
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    # --------------------------- public methods --------------------------- #
    def reach_pose(self,
                target_pos: np.ndarray,
                target_quat: np.ndarray | None = None,
                pos_gain: float = 120.0,
                ori_gain: float = 120.0):
        """
        Drive EE to target_pos; if target_quat (x y z w) is supplied,
        also track orientation.  Otherwise only align local z with world –z.
        """
        # ── current EE pose ────────────────────────────────────────────────
        ee_pos = self.data.site_xpos[self.site_id]
        ee_R   = self.data.site_xmat[self.site_id].reshape(3, 3)
        pos_err = target_pos - ee_pos

        # ── orientation error ──────────────────────────────────────────────
        if target_quat is None:
            # Align local Z to global –Z
            cur_z   = ee_R[:, 2]
            ori_err = np.cross(cur_z, np.array([0, 0, -1]))
        else:
            # 1) normalise caller-supplied quaternion
            q_t = np.asarray(target_quat, float)
            q_t /= np.linalg.norm(q_t)              # ← NEW

            # 2) current orientation (w x y z  →  x y z w)
            q_wxyz = np.zeros(4)
            mujoco.mju_mat2Quat(q_wxyz, self.data.site_xmat[self.site_id])
            q_c = self._wxyz_to_xyzw(q_wxyz)

            # 3) ensure both in same hemisphere
            if np.dot(q_t, q_c) < 0.0:
                q_t = -q_t                          # ← NEW

            # 4) logarithmic error
            ori_err = self._quat_log_error(q_t, q_c)

        # ── stack & solve ──────────────────────────────────────────────────
        task_err = np.concatenate([pos_gain * pos_err,
                                ori_gain * ori_err])
        return self._error_to_torque(task_err, pos_gain, ori_gain)


    def track_twist(self,
                    v_cart: np.ndarray,
                    w_cart: np.ndarray | None = None,
                    lin_gain: float = 1.0,
                    ang_gain: float = 1.0,
                    damping: float | None = None):
        """Map desired twist to torques."""
        if w_cart is None:
            w_cart = np.zeros(3)
        if damping is None:
            damping = self.damping

        twist = np.concatenate([lin_gain * v_cart,
                                ang_gain * w_cart])
        return self._twist_to_torque(twist, lin_gain, ang_gain, damping)

    # -------------------------- private helpers --------------------------- #
    def _compute_jac(self):
        mujoco.mj_jacSite(self.model, self.data,
                          self._jacp, self._jacr, self.site_id)
        return self._jacp, self._jacr

    def _dls(self, J, vec, lam):
        JT = J.T
        reg = lam * np.eye(J.shape[0])
        dq_full = JT @ np.linalg.inv(J @ JT + reg) @ vec
        return dq_full[: self.model.nu]

    def _send_torque(self, dq):
        self.ctrl.set_velocity_target(dq)

        tau = np.zeros(self.model.nu)
        for i in range(self.ctrl.num_actuators):
            if i in self.ctrl.gripper_ids:
                continue
            dof_i = self.ctrl.dof_indices[i]
            v_act = self.data.qvel[dof_i]
            v_tar = self.ctrl.v_targets[i]
            torque_d = -self.ctrl.kd[i] * (v_act - v_tar)
            tau[i] = self.data.qfrc_bias[dof_i] + torque_d
            self.data.ctrl[i] = tau[i]  # 👈 assign torque
        return tau
    
    def _secondary_task(self, J, q_preferred, epsilon=10 ):
        """
        Compute the secondary task gradient for joint centering
        """
        
        J = J [:,:self.model.nu]
        JT = J.T

        I = np.eye(JT.shape[0])

        J_pseudo_inv = JT @ np.linalg.inv(J @ JT)
        J_pseudo_inv = np.linalg.pinv(J)

        q = self.data.qpos[:self.model.nu]
        qvel = self.data.qvel[:self.model.nu]
        grad = epsilon * (q-q_preferred)
        
        print(f"[DEBUG] J.shape = {J.shape}") 
        print(f"[DEBUG] I.shape = {I.shape}") 
        print(f"[DEBUG] q.shape = {q.shape}") 
        print(f"[DEBUG] grad.shape = {grad.shape}") 
        print(f"[DEBUG] J_pseudo_inv @ J.shape = {(J_pseudo_inv @ J).shape}")

        q_secondary = (I - J_pseudo_inv @ J) @ grad

        # # Combine gradients
        # grad_total += grad_centering + grad_posture + grad_damping

        return q_secondary

    
    def _error_to_torque(self, err, lin_gain, ang_gain, lam=None,q_preferred=None,):

        q_preferred=[0.011, -0.7, -0.01, -2.3, -0.01, 1.56, 0.8, 0]

        if lam is None:
            lam = self.damping
        jacp, jacr = self._compute_jac()

        # 1️⃣ Primary Task
        J1 = np.vstack([lin_gain * jacp, ang_gain * jacr])
        dq1 = self._dls(J1, err, lam)

        # 2️⃣ Secondary Task Gradient
        dq2 = self._secondary_task(J1, q_preferred )

        # 4️⃣ Total Command
        dq_total = dq1 - dq2
        return self._send_torque(dq_total)

    def _twist_to_torque(self, twist, lin_gain, ang_gain, lam):
        jacp, jacr = self._compute_jac()
        J = np.vstack([lin_gain * jacp,
                       ang_gain * jacr])
        dq = self._dls(J, twist, lam)
        return self._send_torque(dq)

class MultiDLSVelocityPlanner:

    """
    Wraps multiple DLSVelocityPlanner instances (one per end-effector site)
    and combines their torques into a single data.ctrl array.
    """

    def __init__(self, model, data, arm_sites, kd=5.0, damping=1e-2):
        """
        Args:
            model, data: your mujoco.MjModel/MjData
            arm_sites: list of site_name strings (e.g. ["left_center","right_center"])
            kd: joint‐velocity D‐gain
            damping: DLS damping for IK
        """
        self.model = model
        self.data  = data
        self.planners = []
        for site in arm_sites:
            p = DLSVelocityPlanner(
                model=model,
                data=data,
                kd=kd,
                site_name=site,
                damping=damping
            )
            self.planners.append(p)

    def apply(self, target_positions):
        """
        Compute & sum torques from each arm.
        
        Args:
            target_positions: list of np.array target positions, same order as arm_sites
        """
        # sanity check
        assert len(target_positions) == len(self.planners), \
               "Need one target per planner"

        # accumulate each planner's torque vector
        total_tau = np.zeros(self.model.nu)
        for planner, targ in zip(self.planners, target_positions):
            tau = planner.get_torque_command(targ)
            total_tau += tau

        # write the combined torques back into sim
        self.data.ctrl[:] = total_tau
        print(total_tau)

    def apply_cartesian_velocity(self, v_carts, damping=None, ori_gain=1.0):
        """
        Apply Cartesian velocity commands for each arm.
        v_carts: list of 3‑vectors, one per planner

        This method avoids double-applying gravity bias by adding it only once,
        then accumulating the D‑term deltas from each planner.
        """
        assert len(v_carts) == len(self.planners), "Need one v_cart per planner"

        # 1) apply gravity bias once
        self.hold()
        bias = self.data.ctrl.copy()

        # 2) accumulate D‑term deltas for each planner
        total_delta = np.zeros(self.model.nu)
        for planner, v_cart in zip(self.planners, v_carts):
            tau_i = planner.get_torque_for_cartesian_velocity(
                v_cart, damping=damping, ori_gain=ori_gain
            )
            total_delta += (tau_i - bias)

        # 3) write final combined torques = bias + deltas
        self.data.ctrl[:] = bias + total_delta

    def hold(self):
        """Apply gravity compensation on all arms without any targets."""
        # zero out
        self.data.ctrl[:] = 0
        
        # just call one planner’s controller callback
        # (they all do the same gravity comp)
        self.planners[0].ctrl.control_callback(self.model, self.data)
