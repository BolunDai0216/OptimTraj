import numpy as np
import tensorflow as tf


def cross2D(v, w):
    v = v[:, 0].tolist()
    w = w[:, 0].tolist()
    result = np.cross(v, w) * 1

    return result


class FiveLinkBiped:
    def __init__(self):
        # ------ Model Parameters ------
        self.m1 = 3.2  # kg
        self.m2 = 6.8  # kg
        self.m3 = 20.0  # kg
        self.m4 = 6.8  # kg
        self.m5 = 3.2  # kg

        self.I1 = 0.93  # kg•m2
        self.I2 = 1.08  # kg•m2
        self.I3 = 1.22  # kg•m2
        self.I4 = 1.08  # kg•m2
        self.I5 = 0.93  # kg•m2

        self.l1 = 0.4  # m
        self.l2 = 0.4  # m
        self.l3 = 0.625  # m
        self.l4 = 0.4  # m
        self.l5 = 0.4  # m

        self.c1 = 0.128  # m
        self.c2 = 0.163  # m
        self.c3 = 0.2  # m
        self.c4 = 0.163  # m
        self.c5 = 0.128  # m

        # ------ Simulation Parameters ------
        self.dt = 0.01  # sec
        self.batch_size = 64

        # Link Orientations
        self.q1 = 0.0  # rad
        self.q2 = 0.0  # rad
        self.q3 = 0.0  # rad
        self.q4 = 0.0  # rad
        self.q5 = 0.0  # rad

        # Joint Positions
        self.p0 = 0.0  # rad
        self.p1 = 0.0  # rad
        self.p2 = 0.0  # rad
        self.p3 = 0.0  # rad
        self.p4 = 0.0  # rad
        self.p5 = 0.0  # rad

        # CoM Positions
        self.g1 = 0.0  # rad
        self.g2 = 0.0  # rad
        self.g3 = 0.0  # rad
        self.g4 = 0.0  # rad
        self.g5 = 0.0  # rad

        # Mass Vectors
        self.g = 9.81
        j_vec = tf.constant([[0.], [1.], [0.]])
        self.w1 = -self.m1*self.g*j_vec
        self.w2 = -self.m2*self.g*j_vec
        self.w3 = -self.m3*self.g*j_vec
        self.w4 = -self.m4*self.g*j_vec
        self.w5 = -self.m5*self.g*j_vec

    def update_joint_angles(self, angles):
        a1, a2, a3, a4, a5 = angles
        self.q1 = a1
        self.q2 = a2
        self.q3 = a3
        self.q4 = a4
        self.q5 = a5

    def get_drift(self, states):
        qs = states[:, :, :5]
        dqs = states[:, :, 5:]
        self.update_joint_n_com_pos(qs)
        # have size [batch_size, 5, 1]
        A = self.get_mass_torque()
        # have size [batch_size, 5, 5] and [batch_size, 5, 5]
        B, C = self.get_angular_momentum_change()

        # have size [batch_size, 5, 1]
        drift = tf.linalg.inv(C) @ (
            A - B @ tf.square(tf.transpose(dqs, perm=[0, 2, 1]))
        )

        # have size [batch_size, 1, 5]
        return tf.transpose(drift, perm=[0, 2, 1])

    def get_control_influence(self, states):
        # have [batch_size, 5, 5]
        _, C = self.get_angular_momentum_change()

        # have size [batch_size, 5, 1]
        control_influence = tf.linalg.inv(C)

        # have size [batch_size, 1, 5]
        return control_influence

    def get_mass_torque(self):
        mass_list = [
            self.m1, self.m2, self.m3, self.m4, self.m5
        ]
        com_list = [
            self.g1, self.g2, self.g3, self.g4, self.g5
        ]
        joint_list = [
            self.p0, self.p1, self.p2, self.p2, self.p4
        ]

        _j_vec = tf.constant([[0., 1., 0.]])
        # have size [batch_size, 3]
        j_vec = tf.tile(_j_vec, [self.batch_size, 1])

        mass_torque_mat = tf.zeros([self.batch_size, 0])
        for j in range(5):
            # have size [batch_size, 3]
            mass_torque = tf.zeros_like(j_vec)
            for i in range(j, 5):
                # have size [batch_size, 3]
                com2point = tf.squeeze(com_list[i] - joint_list[j])
                weight_vec = -mass_list[i] * 9.81 * j_vec
                # have size [batch_size, 3]
                mass_torque += tf.linalg.cross(com2point, weight_vec)
            # Has shape [batch_size, j+1]
            mass_torque_mat = tf.concat(
                [mass_torque_mat, mass_torque[:, 2:]], 1)

        # Has shape [batch_size, 5, 1]
        return mass_torque_mat[:, :, tf.newaxis]

    def get_angular_momentum_change(self):
        # each has shape [batch_size, 2, 10]
        ddG_mat = self.get_com_derivatives()

        mass_list = [
            self.m1, self.m2, self.m3, self.m4, self.m5
        ]
        com_list = [
            self.g1, self.g2, self.g3, self.g4, self.g5
        ]
        joint_list = [
            self.p0, self.p1, self.p2, self.p2, self.p4
        ]

        angular_momentum_mat = tf.zeros([self.batch_size, 0, 10])
        for j in range(5):
            mass_torque = tf.zeros([self.batch_size, 1, 10])
            for i in range(j, 5):
                # have size [batch_size, 3, 1]
                _gp = com_list[i] - joint_list[j]
                # have size [batch_size, 1, 2]
                gp = tf.concat(
                    [-_gp[:, 1, :], _gp[:, 0, :]], axis=1
                )[:, tf.newaxis, :]
                mass_torque += mass_list[i] * (gp @ ddG_mat[i])
            # have size [batch_size, j+1, 10]
            angular_momentum_mat = tf.concat(
                [angular_momentum_mat, mass_torque], 1)

        I_mat = tf.constant([
            [self.I1, self.I2, self.I3, self.I4, self.I5],
            [0.0, self.I2, self.I3, self.I4, self.I5],
            [0.0, 0.0, self.I3, self.I4, self.I5],
            [0.0, 0.0, 0.0, self.I4, self.I5],
            [0.0, 0.0, 0.0, 0.0, self.I5]
        ])

        # have size [batch_size, 5, 5]
        angular_momentum_mat_dq = angular_momentum_mat[:, :, :5]
        # have size [batch_size, 5, 5]
        angular_momentum_mat_ddq = angular_momentum_mat[:, :, 5:] + I_mat

        return angular_momentum_mat_dq, angular_momentum_mat_ddq

    def get_com_derivatives(self):
        # ddG1 =
        #   (l1*sin(q1) - c1*sin(q1))*dq1^2 + ddq1*(c1*cos(q1) - l1*cos(q1))
        #   (c1*cos(q1) - l1*cos(q1))*dq1^2 + ddq1*(c1*sin(q1) - l1*sin(q1))

        # has shape [batch_size, 1, 1]
        ddG1_mat_11 = self.l1 * tf.sin(self.q1) - self.c1 * tf.sin(self.q1)
        # has shape [batch_size, 1, 1]
        ddG1_mat_12 = self.c1 * tf.cos(self.q1) - self.l1 * tf.cos(self.q1)
        # has shape [batch_size, 1, 1]
        ddG1_mat_21 = self.c1 * tf.cos(self.q1) - self.l1 * tf.cos(self.q1)
        # has shape [batch_size, 1, 1]
        ddG1_mat_22 = self.c1 * tf.sin(self.q1) - self.l1 * tf.sin(self.q1)

        mat_zeros_1 = tf.zeros([self.batch_size, 1, 4])
        # has shape [batch_size, 1, 10]
        ddG1_mat_1 = tf.concat(
            [ddG1_mat_11, mat_zeros_1, ddG1_mat_12, mat_zeros_1], axis=2)
        # has shape [batch_size, 1, 10]
        ddG1_mat_2 = tf.concat(
            [ddG1_mat_21, mat_zeros_1, ddG1_mat_22, mat_zeros_1], axis=2)
        # has shape [batch_size, 2, 10]
        ddG1_mat = tf.concat([ddG1_mat_1, ddG1_mat_2], axis=1)

        # ddG2 =
        #   l1*sin(q1)*dq1^2 + (l2*sin(q2) - c2*sin(q2))*dq2^2 + ddq2*(c2*cos(q2) - l2*cos(q2)) - ddq1*l1*cos(q1)
        # - l1*cos(q1)*dq1^2 + (c2*cos(q2) - l2*cos(q2))*dq2^2 + ddq2*(c2*sin(q2) - l2*sin(q2)) - ddq1*l1*sin(q1)

        # has shape [batch_size, 1, 1]
        ddG2_mat_111 = self.l1 * tf.sin(self.q1)
        ddG2_mat_112 = self.l2 * tf.sin(self.q2) - self.c2 * tf.sin(self.q2)
        # has shape [batch_size, 1, 1]
        ddG2_mat_121 = -self.l1 * tf.cos(self.q1)
        ddG2_mat_122 = self.c2 * tf.cos(self.q2) - self.l2 * tf.cos(self.q2)
        # has shape [batch_size, 1, 1]
        ddG2_mat_211 = -self.l1 * tf.cos(self.q1)
        ddG2_mat_212 = self.c2 * tf.cos(self.q2) - self.l2 * tf.cos(self.q2)
        # has shape [batch_size, 1, 1]
        ddG2_mat_221 = -self.l1 * tf.sin(self.q1)
        ddG2_mat_222 = self.c2 * tf.sin(self.q2) - self.l2 * tf.sin(self.q2)

        mat_zeros_2 = tf.zeros([self.batch_size, 1, 3])
        # has shape [batch_size, 1, 10]
        ddG2_mat_1 = tf.concat([ddG2_mat_111, ddG2_mat_112, mat_zeros_2,
                                ddG2_mat_121, ddG2_mat_122, mat_zeros_2], axis=2)
        # has shape [batch_size, 1, 10]
        ddG2_mat_2 = tf.concat([ddG2_mat_211, ddG2_mat_212, mat_zeros_2,
                                ddG2_mat_221, ddG2_mat_222, mat_zeros_2], axis=2)
        # has shape [batch_size, 2, 10]
        ddG2_mat = tf.concat([ddG2_mat_1, ddG2_mat_2], axis=1)

        # ddG3 =
        #   l1*sin(q1)*dq1^2 + l2*sin(q2)*dq2^2 + (l3*sin(q3) - c3*sin(q3))*dq3^2 + ddq3*(c3*cos(q3) - l3*cos(q3)) - ddq1*l1*cos(q1) - ddq2*l2*cos(q2)
        # - l1*cos(q1)*dq1^2 - l2*cos(q2)*dq2^2 + (c3*cos(q3) - l3*cos(q3))*dq3^2 + ddq3*(c3*sin(q3) - l3*sin(q3)) - ddq1*l1*sin(q1) - ddq2*l2*sin(q2)

        # has shape [batch_size, 1, 1]
        ddG3_mat_111 = self.l1 * tf.sin(self.q1)
        ddG3_mat_112 = self.l2 * tf.sin(self.q2)
        ddG3_mat_113 = self.l3 * tf.sin(self.q3) - self.c3 * tf.sin(self.q3)
        # has shape [batch_size, 1, 1]
        ddG3_mat_121 = -self.l1 * tf.cos(self.q1)
        ddG3_mat_122 = -self.l2 * tf.cos(self.q2)
        ddG3_mat_123 = self.c3 * tf.cos(self.q3) - self.l3 * tf.cos(self.q3)
        # has shape [batch_size, 1, 1]
        ddG3_mat_211 = -self.l1 * tf.cos(self.q1)
        ddG3_mat_212 = -self.l2 * tf.cos(self.q2)
        ddG3_mat_213 = self.c3 * tf.cos(self.q3) - self.l3 * tf.cos(self.q3)
        # has shape [batch_size, 1, 1]
        ddG3_mat_221 = -self.l1 * tf.sin(self.q1)
        ddG3_mat_222 = -self.l2 * tf.sin(self.q2)
        ddG3_mat_223 = self.c3 * tf.sin(self.q3) - self.l3 * tf.sin(self.q3)

        mat_zeros_3 = tf.zeros([self.batch_size, 1, 2])
        # has shape [batch_size, 1, 10]
        ddG3_mat_1 = tf.concat([ddG3_mat_111, ddG3_mat_112, ddG3_mat_113, mat_zeros_3,
                                ddG3_mat_121, ddG3_mat_122, ddG3_mat_123, mat_zeros_3], axis=2)
        # has shape [batch_size, 1, 10]
        ddG3_mat_2 = tf.concat([ddG3_mat_211, ddG3_mat_212, ddG3_mat_213, mat_zeros_3,
                                ddG3_mat_221, ddG3_mat_222, ddG3_mat_223, mat_zeros_3], axis=2)
        # has shape [batch_size, 2, 10]
        ddG3_mat = tf.concat([ddG3_mat_1, ddG3_mat_2], axis=1)

        # ddG4 =
        #   l1*sin(q1)*dq1^2 + l2*sin(q2)*dq2^2 - c4*sin(q4)*dq4^2 - ddq1*l1*cos(q1) - ddq2*l2*cos(q2) + c4*ddq4*cos(q4)
        # - l1*cos(q1)*dq1^2 - l2*cos(q2)*dq2^2 + c4*cos(q4)*dq4^2 + c4*ddq4*sin(q4) - ddq1*l1*sin(q1) - ddq2*l2*sin(q2)

        # has shape [batch_size, 1, 1]
        ddG4_mat_111 = self.l1 * tf.sin(self.q1)
        ddG4_mat_112 = self.l2 * tf.sin(self.q2)
        ddG4_mat_113 = -self.c4 * tf.sin(self.q4)
        # has shape [batch_size, 1, 1]
        ddG4_mat_121 = -self.l1 * tf.cos(self.q1)
        ddG4_mat_122 = -self.l2 * tf.cos(self.q2)
        ddG4_mat_123 = self.c4 * tf.cos(self.q4)
        # has shape [batch_size, 1, 1]
        ddG4_mat_211 = -self.l1 * tf.cos(self.q1)
        ddG4_mat_212 = -self.l2 * tf.cos(self.q2)
        ddG4_mat_213 = self.c4 * tf.cos(self.q4)
        # has shape [batch_size, 1, 1]
        ddG4_mat_221 = -self.l1 * tf.sin(self.q1)
        ddG4_mat_222 = -self.l2 * tf.sin(self.q2)
        ddG4_mat_223 = self.c4 * tf.sin(self.q4)

        mat_zeros_4 = tf.zeros([self.batch_size, 1, 1])
        # has shape [batch_size, 1, 10]
        ddG4_mat_1 = tf.concat([ddG4_mat_111, ddG4_mat_112, mat_zeros_4, ddG4_mat_113, mat_zeros_4,
                                ddG4_mat_121, ddG4_mat_122, mat_zeros_4, ddG4_mat_123, mat_zeros_4], axis=2)
        # has shape [batch_size, 1, 10]
        ddG4_mat_2 = tf.concat([ddG4_mat_211, ddG4_mat_212, mat_zeros_4, ddG4_mat_213, mat_zeros_4,
                                ddG4_mat_221, ddG4_mat_222, mat_zeros_4, ddG4_mat_223, mat_zeros_4], axis=2)
        # has shape [batch_size, 2, 10]
        ddG4_mat = tf.concat([ddG4_mat_1, ddG4_mat_2], axis=1)

        # ddG5 =
        #   l1*sin(q1)*dq1^2 + l2*sin(q2)*dq2^2 - l4*sin(q4)*dq4^2 - c5*sin(q5)*dq5^2 - ddq1*l1*cos(q1) - ddq2*l2*cos(q2) + ddq4*l4*cos(q4) + c5*ddq5*cos(q5)
        # - l1*cos(q1)*dq1^2 - l2*cos(q2)*dq2^2 + l4*cos(q4)*dq4^2 + c5*cos(q5)*dq5^2 + c5*ddq5*sin(q5) - ddq1*l1*sin(q1) - ddq2*l2*sin(q2) + ddq4*l4*sin(q4)

        # has shape [batch_size, 1, 1]
        ddG5_mat_111 = self.l1 * tf.sin(self.q1)
        ddG5_mat_112 = self.l2 * tf.sin(self.q2)
        ddG5_mat_113 = -self.l4 * tf.sin(self.q4)
        ddG5_mat_114 = -self.c5 * tf.sin(self.q5)
        # has shape [batch_size, 1, 1]
        ddG5_mat_121 = -self.l1 * tf.cos(self.q1)
        ddG5_mat_122 = -self.l2 * tf.cos(self.q2)
        ddG5_mat_123 = self.l4 * tf.cos(self.q4)
        ddG5_mat_124 = self.c5 * tf.cos(self.q5)
        # has shape [batch_size, 1, 1]
        ddG5_mat_211 = -self.l1 * tf.cos(self.q1)
        ddG5_mat_212 = -self.l2 * tf.cos(self.q2)
        ddG5_mat_213 = self.l4 * tf.cos(self.q4)
        ddG5_mat_214 = self.c5 * tf.cos(self.q5)
        # has shape [batch_size, 1, 1]
        ddG5_mat_221 = -self.l1 * tf.sin(self.q1)
        ddG5_mat_222 = -self.l2 * tf.sin(self.q2)
        ddG5_mat_223 = self.l4 * tf.sin(self.q4)
        ddG5_mat_224 = self.c5 * tf.sin(self.q5)

        mat_zeros_5 = tf.zeros([self.batch_size, 1, 1])
        # has shape [batch_size, 1, 10]
        ddG5_mat_1 = tf.concat([ddG5_mat_111, ddG5_mat_112, mat_zeros_5, ddG5_mat_113, ddG5_mat_114,
                                ddG5_mat_121, ddG5_mat_122, mat_zeros_5, ddG5_mat_123, ddG5_mat_124], axis=2)
        # has shape [batch_size, 1, 10]
        ddG5_mat_2 = tf.concat([ddG5_mat_211, ddG5_mat_212, mat_zeros_5, ddG5_mat_213, ddG5_mat_214,
                                ddG5_mat_221, ddG5_mat_222, mat_zeros_5, ddG5_mat_223, ddG5_mat_224], axis=2)
        # has shape [batch_size, 2, 10]
        ddG5_mat = tf.concat([ddG5_mat_1, ddG5_mat_2], axis=1)

        return [ddG1_mat, ddG2_mat, ddG3_mat, ddG4_mat, ddG5_mat]

    def update_joint_n_com_pos(self, qs):
        """
        qs: shape [batch_size, 1, state_dim]
        """
        _i_vec = tf.constant([[1.], [0.], [0.]])
        _j_vec = tf.constant([[0.], [1.], [0.]])

        # Has shape [batch_size, 3, 1]
        i_vec = tf.tile(_i_vec[tf.newaxis, :, :], [self.batch_size, 1, 1])
        j_vec = tf.tile(_j_vec[tf.newaxis, :, :], [self.batch_size, 1, 1])

        # get angles, has shape [batch_size, 1, 1]
        self.q1 = qs[:, :, 0][:, :, tf.newaxis]
        self.q2 = qs[:, :, 1][:, :, tf.newaxis]
        self.q3 = qs[:, :, 2][:, :, tf.newaxis]
        self.q4 = qs[:, :, 3][:, :, tf.newaxis]
        self.q5 = qs[:, :, 4][:, :, tf.newaxis]

        # All ei's have size [batch_size, 3, 1]
        # unit vector from P0 -> P1, (contact point to stance knee)
        e1 = (j_vec) @ tf.cos(self.q1) + (-i_vec) @ tf.sin(self.q1)
        # unit vector from P1 -> P2, (stance knee to hip)
        e2 = (j_vec) @ tf.cos(self.q2) + (-i_vec) @ tf.sin(self.q2)
        # unit vector from P2 -> P3, (hip to shoulders)
        e3 = (j_vec) @ tf.cos(self.q3) + (-i_vec) @ tf.sin(self.q3)
        # unit vector from P2 -> P4, (hip to swing knee)
        e4 = -(j_vec) @ tf.cos(self.q4) - (-i_vec) @ tf.sin(self.q4)
        # unit vector from P4 -> P5, (swing knee to swing foot)
        e5 = -(j_vec) @ tf.cos(self.q5) - (-i_vec) @ tf.sin(self.q5)

        # All pi's have size [batch_size, 3, 1]
        # stance foot = Contact point = origin
        self.p0 = 0 * i_vec + 0 * i_vec
        self.p1 = self.p0 + self.l1 * e1  # stance knee
        self.p2 = self.p1 + self.l2 * e2  # hip
        self.p3 = self.p2 + self.l3 * e3  # shoulders
        self.p4 = self.p2 + self.l4 * e4  # swing knee
        self.p5 = self.p4 + self.l5 * e5  # swing foot

        # All gi's have size [batch_size, 3, 1]
        self.g1 = self.p1 - self.c1 * e1  # CoM stance leg tibia
        self.g2 = self.p2 - self.c2 * e2  # CoM stance leg febur
        self.g3 = self.p3 - self.c3 * e3  # CoM torso
        self.g4 = self.p2 + self.c4 * e4  # CoM swing leg femur
        self.g5 = self.p4 + self.c5 * e5  # CoM swing leg tibia
