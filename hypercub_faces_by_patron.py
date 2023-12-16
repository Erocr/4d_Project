from math import cos, sin, radians
from copy import deepcopy


def rotation(p, center, axis1, axis2, angle):
    """(p[0],
           p[1] * cos(-pi / 2) + p[2] * sin(-pi / 2),
           -p[1] * sin(-pi / 2) + p[2] * cos(-pi / 2),
           p[3])"""
    new = p.copy()
    new[0] -= center[0]
    new[1] -= center[1]
    new[2] -= center[2]
    new[3] -= center[3]
    new[axis1], new[axis2] = (new[axis1] * cos(radians(angle)) + new[axis2] * sin(radians(angle)),
                              -new[axis1] * sin(radians(angle)) + new[axis2] * cos(radians(angle)))
    new[0] = int(new[0] + center[0])
    new[1] = int(new[1] + center[1])
    new[2] = int(new[2] + center[2])
    new[3] = int(new[3] + center[3])
    return new


def rotate_cube(cube, center, axis1, axis2, angle):
    cube = deepcopy(cube)
    for i in range(len(cube)):
        cube[i] = rotation(cube[i], center, axis1, axis2, angle)
    return cube


def good_cube(cube):
    for point in cube:
        if point[3] < 0:
            return False
    return True


center_cube_points = [[0, 0, 0, 0],
                      [10, 0, 0, 0],
                      [0, 10, 0, 0],
                      [10, 10, 0, 0],
                      [0, 0, 10, 0],
                      [10, 0, 10, 0],
                      [0, 10, 10, 0],
                      [10, 10, 10, 0]]

x1, y1, z1, w1 = 0, 0, 0, 0
x2, y2, z2, w2 = 10, 10, 10, 10
points_order = {(x1, y1, z1, w1): 0,
                (x2, y1, z1, w1): 1,
                (x1, y2, z1, w1): 2,
                (x2, y2, z1, w1): 3,
                (x1, y1, z2, w1): 4,
                (x2, y1, z2, w1): 5,
                (x1, y2, z2, w1): 6,
                (x2, y2, z2, w1): 7,
                (x1, y1, z1, w2): 8,
                (z2, y1, z1, w2): 9,
                (x1, y2, z1, w2): 10,
                (x2, y2, z1, w2): 11,
                (x1, y1, z2, w2): 12,
                (x2, y1, z2, w2): 13,
                (x1, y2, z2, w2): 14,
                (x2, y2, z2, w2): 15}


def draw_cube_points(cube):
    to_print = ""
    for elt in cube:
        to_print += str(points_order[tuple(elt)]) + ", "
    print(to_print)


for axis in range(3):
    cube1 = deepcopy(center_cube_points)
    cube2 = deepcopy(center_cube_points)
    for i in range(len(cube1)):
        cube1[i][axis] += 10
    center1 = [5, 5, 5, 0]
    center1[axis] += 5
    for i in range(len(cube2)):
        cube2[i][axis] -= 10
    center2 = [5, 5, 5, 0]
    center2[axis] -= 5
    c1 = rotate_cube(cube1, center1, axis, 3, 90)
    if good_cube(c1):
        #print(f"{axis}: {c1}")
        draw_cube_points(c1)
    else:
        #print(f"{axis}: {rotate_cube(cube1, center1, axis, 3, -90)}")
        draw_cube_points(rotate_cube(cube1, center1, axis, 3, -90))
    c2 = rotate_cube(cube2, center2, axis, 3, 90)
    if good_cube(c2):
        #print(f"{axis}: {rotate_cube(cube2, center2, axis, 3, 90)}")
        draw_cube_points(c2)
    else:
        #print(f"{axis}: {rotate_cube(cube2, center2, axis, 3, -90)}")
        draw_cube_points(rotate_cube(cube2, center2, axis, 3, -90))
