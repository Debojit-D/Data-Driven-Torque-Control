import numpy as np
import mujoco
import cvxpy as cp
from utils.mj_velocity_control.mj_velocity_ctrl import JointVelocityController


class QPVelocityPlanner:
    """
    Joint-torque planner using QP-based inverse kinematics with a JointVelocityController.

    Public API
    ----------
    reach_pose(target_pos, target_quat=None)
        – Track position and orientation (quaternion [x y z w] format).
    """

    @staticmethod
    def _wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
        """Convert MuJoCo order [w x y z] → [x y z w]."""
        q = np.asarray(q_wxyz)
        return np.array([q[1], q[2], q[3], q[0]])

    @staticmethod
    def _quat_log_error(q_t: np.ndarray, q_c: np.ndarray) -> np.ndarray:
        """Logarithmic orientation error between quaternions q_t and q_c (both [x y z w])."""
        q_c_inv = np.array([-q_c[0], -q_c[1], -q_c[2], q_c[3]])
        x1, y1, z1, w1 = q_t
        x2, y2, z2, w2 = q_c_inv
        q_e = np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])
        if q_e[3] < 0.0:
            q_e *= -1.0
        ang = 2.0 * np.arccos(np.clip(q_e[3], -1.0, 1.0))
        if ang < 1e-6:
            return np.zeros(3)
        axis = q_e[:3] / np.sin(ang / 2.0)
        return ang * axis

    def __init__(self,
                 model,
                 data,
                 kd: float = 5.0,
                 site_name: str = "right_center",
                 damping: float = 1e-2,
                 gripper_cfg: list[dict] | None = None):
        self.model = model
        self.data = data
        self.site_name = site_name
        self.damping = damping
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

        if gripper_cfg is not None:
            gripper_ids = {g["actuator_id"] for g in gripper_cfg}
        else:
            gripper_ids = set()

        self.ctrl = JointVelocityController(model, data, kd=kd, gripper_ids=gripper_ids)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def reach_pose(self, target_pos, target_quat=None, pos_gain=300.0, ori_gain=300.0):
        """
        Drive EE to target_pos; if target_quat (x y z w) is supplied,
        also track orientation.  Otherwise only align local z with world –z.
        """
        ee_pos = self.data.site_xpos[self.site_id]
        ee_R = self.data.site_xmat[self.site_id].reshape(3, 3)
        pos_err = target_pos - ee_pos

        if target_quat is None:
            cur_z = ee_R[:, 2]
            ori_err = np.cross(cur_z, np.array([0, 0, -1]))
        else:
            q_t = np.asarray(target_quat, float)
            q_t /= np.linalg.norm(q_t)
            q_wxyz = np.zeros(4)
            mujoco.mju_mat2Quat(q_wxyz, self.data.site_xmat[self.site_id])
            q_c = self._wxyz_to_xyzw(q_wxyz)
            if np.dot(q_t, q_c) < 0.0:
                q_t = -q_t
            ori_err = self._quat_log_error(q_t, q_c)

        task_err = np.concatenate([pos_gain * pos_err, ori_gain * ori_err])
        return self._error_to_torque(task_err, pos_gain, ori_gain)

    def _compute_jac(self):
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
        return self._jacp, self._jacr

    def _error_to_torque(self, err, lin_gain, ang_gain):
        jacp, jacr = self._compute_jac()
        J = np.vstack([lin_gain * jacp, ang_gain * jacr])
        dq_var = cp.Variable(self.model.nv)
        cost = cp.sum_squares(J @ dq_var - err) + self.damping * cp.sum_squares(dq_var)
        prob = cp.Problem(cp.Minimize(0.5 * cost))
        prob.solve(solver=cp.OSQP, eps_abs=1e-5, eps_rel=1e-5, max_iter=1000)
        if dq_var.value is None:
            raise RuntimeError("QP failed to solve.")
        dq = dq_var.value[:self.model.nu]
        return self._send_torque(dq)

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
            self.data.ctrl[i] = tau[i]
        return tau