using Rotations

quat = rand(UnitQuaternion)

q = [quat.q.s, quat.q.v1, quat.q.v2, quat.q.v3]
R = [
(1 - 2*q[3]*q[3] - 2*q[4]*q[4]),
(    2*q[2]*q[3] - 2*q[1]*q[4]),
(    2*q[2]*q[4] + 2*q[1]*q[3]),
(    2*q[2]*q[3] + 2*q[1]*q[4]),
(1 - 2*q[2]*q[2] - 2*q[4]*q[4]),
(    2*q[3]*q[4] - 2*q[1]*q[2]),
(    2*q[2]*q[4] - 2*q[1]*q[3]),
(    2*q[3]*q[4] + 2*q[1]*q[2]),
(1 - 2*q[2]*q[2] - 2*q[3]*q[3]),
]
R = reshape(R, 3, 3)'

R - quat