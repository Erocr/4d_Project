from math import cos, sin, pi

x1, y1, z1, w1 = 0, 0, 0, 0
x2, y2, z2, w2 = 1, 1, 1, 1
hypercube_points = {"p0": [x1, y1, z1, w1],
                    "p1": [x2, y1, z1, w1],
                    "p2": [x1, y2, z1, w1],
                    "p3": [x2, y2, z1, w1],
                    "p4": [x1, y1, z2, w1],
                    "p5": [x2, y1, z2, w1],
                    "p6": [x1, y2, z2, w1],
                    "p7": [x2, y2, z2, w1],
                    "p8": [x1, y1, z1, w2],
                    "p9": [z2, y1, z1, w2],
                    "p10": [x1, y2, z1, w2],
                    "p11": [x2, y2, z1, w2],
                    "p12": [x1, y1, z2, w2],
                    "p13": [x2, y1, z2, w2],
                    "p14": [x1, y2, z2, w2],
                    "p15": [x2, y2, z2, w2]}


def rotation_90degrees(p, axis1, axis2):
    """(p[0],
           p[1] * cos(-pi / 2) + p[2] * sin(-pi / 2),
           -p[1] * sin(-pi / 2) + p[2] * cos(-pi / 2),
           p[3])"""
    p[0] -= 0.5
    p[1] -= 0.5
    p[2] -= 0.5
    p[3] -= 0.5
    p[axis1], p[axis2] = (p[axis1] * cos(-pi/2) + p[axis2] * sin(-pi/2),
                          -p[axis1] * sin(-pi/2) + p[axis2] * cos(-pi/2))
    p[0] = int(p[0] + 0.5)
    p[1] = int(p[1] + 0.5)
    p[2] = int(p[2] + 0.5)
    p[3] = int(p[3] + 0.5)
    return p


def draw_good_order(points):
    good_points1 = ((x1, y1, z1, w1), (x2, y1, z1, w1), (x1, y2, z1, w1), (x2, y2, z1, w1), (x1, y1, z2, w1),
                    (x2, y1, z2, w1), (x1, y2, z2, w1), (x2, y2, z2, w1))
    good_points2 = ((x1, y1, z1, w2), (z2, y1, z1, w2), (x1, y2, z1, w2), (x2, y2, z1, w2), (x1, y1, z2, w2),
                    (x2, y1, z2, w2), (x1, y2, z2, w2), (x2, y2, z2, w2))
    result1 = [""] * 8
    result2 = [""] * 8
    for elt in points:
        p = tuple(points[elt])
        if p in good_points1:
            result1[good_points1.index(p)] = elt
        if p in good_points2:
            result2[good_points2.index(p)] = elt
    print(result1)
    print(result2)


draw_good_order(hypercube_points)
for axis1 in range(2):
    for axis2 in range(axis1+1, 4):
        rotated_hypercube = hypercube_points.copy()
        for point in rotated_hypercube:
            rotated_hypercube[point] = rotation_90degrees(rotated_hypercube[point], axis1, axis2)
        draw_good_order(rotated_hypercube)

