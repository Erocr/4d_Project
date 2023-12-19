from math import *
import pygame as pg
from random import random


def cart_to_pol(x, y, z):
    d = sqrt(x**2+y**2+z**2)
    if d == 0: return 0, 0, atan2(z, x)
    theta = acos(y/d)  # angle vertical partant de y vers le bas en radians
    phi = atan2(z, x)  # angle horizontal partant de x vers z
    return d, theta, phi


def pol_to_cart(d, theta, phi):
    x = d * sin(theta) * cos(phi)
    y = d * cos(theta)
    z = d * sin(theta) * sin(phi)
    return x, y, z


def equation(x1, y1, x2, y2) -> tuple:
    a = y2 - y1
    b = x1 - x2
    c = -x1 * y2 + x1 * y1 + y1 * x2 - y1 * x1
    return a, b, c


def sigmoid(x: float):
    return 1/(1+e**(-x))


def moy(x: tuple):
    res = 0
    if len(x) == 0: return 0
    for elt in x:
        res += elt
    return res/len(x)


def plane_line_inter(plane_points, plane_normal, line_start, line_end):
    plane_normal = normalise(plane_normal)
    plane_d = -dot_product(plane_normal, plane_points)
    ad = dot_product(line_start, plane_normal)
    bd = dot_product(line_end, plane_normal)
    t = (-plane_d - ad) / (bd - ad)
    line_vector = (line_end[0]-line_start[0], line_end[1]-line_start[1], line_end[2]-line_start[2])
    line_vector = (line_vector[0]*t, line_vector[1]*t, line_vector[2]*t)
    return line_start[0]+line_vector[0], line_start[1]+line_vector[1], line_start[2]+line_vector[2]


def add(v1, v2):
    v = ()
    for i in range(len(v1)):
        v += (v1[i]+v2[i],)
    return v


def inter(line, obs):
    a, b, c = line
    if a == 0:
        if b != 0 and obs.y < -c / b < obs.y + obs.height:
            return True
        else:
            return False
    elif b == 0:
        if obs.x < -c / a < obs.x + obs.width:
            return True
        else:
            return False
    else:
        y1 = (obs.x + obs.width + c / a) / (-b / a)
        y2 = (obs.x + c / a) / (-b / a)
        if (-b / a) < 0:
            y3 = y1
            y1 = y2
            y2 = y3
        return not (obs.y + obs.height <= y2 or obs.y >= y1)


def complete_inter(x1, y1, x2, y2, obs):
    line = equation(x1, y1, x2, y2)
    if inter(line, obs):
        vx = x2 - x1
        vy = y2 - y1
        if vx > 0:
            if obs.x > x2 or obs.x + obs.width < x1: return False
        elif vx < 0:
            if obs.x + obs.width < x2 or obs.x > x1: return False
        if vy > 0:
            if obs.y > y2 or obs.y + obs.height < y1: return False
        elif vy < 0:
            if obs.y + obs.height < y2 or obs.y > y1: return False
        return True


def in_triangle(ax, ay, bx, by, cx, cy, mx, my):
  det = bx*cy-ay*bx-ax*cy-by*cx+by*ax+ay*cx
  if det == 0: return False
  t1 = (cy*mx-cy*ax-ay*mx+ax*my-cx*my+cx*ay)/det
  t2 = (ay*mx-by*mx+by*ax+bx*my-bx*ay-ax*my)/det
  return t1 > 0 and t2 > 0 and t1+t2 < 1


def intersection_point(px, py, ang, x1, y1, x2, y2) -> tuple or bool:
    vx, vy = rotate(0, 1, ang)
    x1, y1, x2, y2, x3, y3, x4, y4 = px, py, px+vx*3000, py+vy*3000, x1, y1, x2, y2
    t = ((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4)) / (((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))+0.0000000001)
    u = ((x1-x3)*(y1-y2)-(y1-y3)*(x1-x2)) / (((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))+0.0000000001)
    if 0 <= t <= 1 and 0 <= u <= 1:
        return x1 + t*(x2-x1), y1 + t*(y2-y1)
    return False


def inter_segment(x1, y1, x2, y2, x3, y3, x4, y4):
    t = ((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4)) / (((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))+0.0000000001)
    u = ((x1-x3)*(y1-y2)-(y1-y3)*(x1-x2)) / (((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))+0.0000000001)
    if 0 <= t <= 1 and 0 <= u <= 1:
        return x1 + t*(x2-x1), y1 + t*(y2-y1)
    return False


def intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    if (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) == 0:
        return x1, y1
    x = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
    y = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
    return int(x), int(y)


def rotate(vx: float, vy: float, alpha: float) -> tuple:
    alpha = -radians(alpha)
    return cos(alpha) * vx - sin(alpha) * vy, sin(alpha) * vx + cos(alpha) * vy


def angle_from_vect(vx1, vy1, vx2, vy2):
    norm1 = sqrt(vx1 ** 2 + vy1 ** 2)
    norm2 = sqrt(vx2 ** 2 + vy2 ** 2)
    if (vx1 * vx2 + vy1 * vy2) / (norm1 * norm2) > 1:
        return degrees(acos(1))
    return degrees(acos((vx1 * vx2 + vy1 * vy2) / (norm1 * norm2)))


def angle(xa, ya, xb, yb, xc, yc):
    a = dist(xc, yc, xb, yb)
    b = dist(xc, yc, xa, ya)
    c = dist(xa, ya, xb, yb)
    if (xc == xa and yc == ya) or (xc == xb and yc == yb):
        return acos((a ** 2 + b ** 2 - c ** 2))
    if abs((a ** 2 + b ** 2 - c ** 2) / (2 * a * b)) > 1:
        return 0
    return degrees(acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b)))


def dist(x1, y1, x2, y2):
    """Retourne la distance entre le point x1 y1 et le point x2 y2"""
    return int(sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))


def dist2(p1, p2):
    """Retourne la distance entre le point x1 y1 et le point x2 y2"""
    return int(sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


def dist3(obj1, obj2):
    pos_x = pos_y = None
    if obj2.x > obj1.x + obj1.width:
        pos_x = "left"
    elif obj1.x > obj2.x + obj2.width:
        pos_x = "right"
    elif obj2.y > obj1.y + obj1.height:
        pos_y = "up"
    elif obj1.y > obj2.y + obj2.height:
        pos_y = "down"
    if pos_x is not None and pos_y is not None:
        return dist(obj1.x+obj1.width*int(pos_x == "left"), obj1.y+obj1.height*int(pos_y == "up"),
                    obj2.x+obj2.width*int(pos_x == "right"), obj2.y+obj2.height*int(pos_y == "down"))
    elif pos_x is not None and pos_y is None:
        return abs((obj1.x+obj1.width*int(pos_x == "left")) - (obj2.x+obj2.width*int(pos_x == "right")))
    elif pos_x is None and pos_y is not None:
        return abs((obj1.y+obj1.height*int(pos_y == "up")) - (obj2.y+obj2.height*int(pos_y == "down")))
    else:
        return 0


def normal_3d(v1, v2) -> tuple:
    return (v1[1]*v2[2]-v1[2]*v2[1],
            v1[2]*v2[0]-v1[0]*v2[2],
            v1[0]*v2[1]-v1[1]*v2[0])


def normal_p_3d(p1, p2, p3) -> tuple:
    v1 = (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])
    v2 = (p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2])
    return normal_3d(v1, v2)


def normalise(vector):
    length = sqrt(vector[0]**2+vector[1]**2+vector[2]**2)+0.0001
    return vector[0]/length, vector[1]/length, vector[2]/length


def dot_product(v1, v2):
    assert len(v1) == len(v2), "Jaj"
    return sum([v1[i]*v2[i] for i in range(len(v1))])


def vector1(x1, y1, x2, y2):
    """ return the vector between two points """
    return x2 - x1, y2 - y1


def vector2(p1, p2):
    """ return the vector between two points """
    return p2[0] - p1[0], p2[1] - p1[1]


def short_vect(x, y, speed):
    if speed == 0:
        return 0, 0
    a = sqrt(x ** 2 + y ** 2) / speed
    if a < 0.1: return 0, 0
    return x / a, y / a


def f_sum(forces, without=()):
    s = 0
    for force in forces:
        if force not in without:
            s += forces[force]
    return s


white = (255, 255, 255)
black = (0, 0, 0)


def triangulation(face):
    faces = []
    for i in range(1, len(face) - 1):
        faces.append((face[0], face[i], face[i + 1]))
    return faces


class Model4dv2:
    def __init__(self, points, faces, color=(255, 255, 255), tetrahedrons=None):
        centre = (0, 0, 0, 0)
        self.mini_map_p1, self.mini_map_p2 = (None, None)
        self.color = color
        self.center3d = ()
        for point in points:
            if self.mini_map_p1 is None or point[3] < self.mini_map_p1[1]:
                self.mini_map_p1 = (point[0], point[3])
            if self.mini_map_p2 is None or point[3] > self.mini_map_p1[1]:
                self.mini_map_p2 = (point[0], point[3])
            centre = (centre[0] + point[0], centre[1] + point[1], centre[2] + point[2], centre[3] + point[3])
        self.center = (centre[0] / len(points), centre[1] / len(points), centre[2] / len(points), centre[3] / len(points))
        if tetrahedrons is None:
            self.tetrahedrons = ()
            for face in faces:
                self.tetrahedrons += (Tetrahedron(points[face[0]], points[face[1]], points[face[2]], centre),)
        else:
            self.tetrahedrons = tetrahedrons
        self.vert_start = None
        self.vertices = None

    def get_elements(self, v_start, cam):
        # prend les bouts de tétraèdres d'intersection entre les tétraèdres de l'objet et l'espace dans lequel on regarde
        self.vert_start = v_start
        faces = []
        self.vertices = []
        tetra2triangles = ((0, 1, 2), (0, 3, 2), (1, 2, 3), (1, 3, 0))
        #tetra2triangles = ((0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 1, 5), (0, 2, 3), (0, 2, 4), (0, 2, 5), (0, 3, 4),
        #                   (0, 3, 5), (0, 4, 5), (1, 2, 3), (1, 2, 4), (1, 2, 5), (1, 3, 4), (1, 3, 5), (1, 4, 5),
        #                   (2, 3, 4), (2, 3, 5), (2, 4, 5), (3, 4, 5))
        #tetra2triangles = ((0, 1, 2), (4, 5, 3))
        for tetrahedron in self.tetrahedrons:
            points = tetrahedron.spaceinter(cam.visible_space)
            for i in range(len(points)-1, -1, -1):
                if points[i] == (): points.pop(i)
                else:
                    points[i] = points[i][0]
            triangles = triangulation(points)
            for triangle in triangles:
                self.vertices.append(
                    (triangle[0][0] + cam.pos[0], triangle[0][1] + cam.pos[1], triangle[0][2] + cam.pos[2]))
                self.vertices.append(
                    (triangle[1][0] + cam.pos[0], triangle[1][1] + cam.pos[1], triangle[1][2] + cam.pos[2]))
                self.vertices.append(
                    (triangle[2][0] + cam.pos[0], triangle[2][1] + cam.pos[1], triangle[2][2] + cam.pos[2]))
                index = len(self.vertices) + v_start
                faces.append((index - 1, index - 2, index - 3))
        self.refresh_center3d()
        return self.vertices, faces

    def refresh_center3d(self):
        self.center3d = (0, 0, 0)
        if len(self.vertices) == 0:
            return None
        for vertex in self.vertices:
            self.center3d = add(self.center3d, vertex)
        self.center3d = (self.center3d[0]/len(self.vertices),
                         self.center3d[1]/len(self.vertices),
                         self.center3d[2]/len(self.vertices))

    def get_xbox(self):
        # le nom de cette fonction provient d'un jeu de mot avec hitbox
        max_x = max_y = max_z = -100000
        min_x = min_y = min_z = 100000
        for vert in self.vertices:
            if vert[0] > max_x:
                max_x = vert[0]
            elif vert[0] < min_x:
                min_x = vert[0]
            if vert[1] > max_y:
                max_y = vert[1]
            elif vert[1] < min_y:
                min_y = vert[1]
            if vert[2] > max_z:
                max_z = vert[2]
            elif vert[2] < min_z:
                min_z = vert[2]
        return (min_x, min_y, min_z), (max_x, max_y, max_z)

    def get_color(self):
        return self.color

    def get_center(self):
        return self.center


class Hypercube(Model4dv2):  # un model4d spécial: un hyper pavé
    def __init__(self, pos1, pos2, col):
        x1, y1, z1, w1 = pos1
        x2, y2, z2, w2 = pos2
        hypercube_points = [(x1, y1, z1, w1), (x2, y1, z1, w1), (x1, y2, z1, w1), (x2, y2, z1, w1), (x1, y1, z2, w1),
                            (x2, y1, z2, w1), (x1, y2, z2, w1), (x2, y2, z2, w1),  (x1, y1, z1, w2), (z2, y1, z1, w2),
                            (x1, y2, z1, w2), (x2, y2, z1, w2), (x1, y1, z2, w2), (x2, y1, z2, w2), (x1, y2, z2, w2),
                            (x2, y2, z2, w2)]
        hypercube_faces = self.cube_faces(9, 13, 11, 15, 1, 5, 3, 7)
        hypercube_faces += self.cube_faces(12, 8, 14, 10, 4, 0, 6, 2)
        hypercube_faces += self.cube_faces(8, 9, 10, 11, 0, 1, 2, 3)
        hypercube_faces += self.cube_faces(13, 12, 15, 14, 5, 4, 7, 6)
        hypercube_faces += self.cube_faces(13, 12, 9, 8, 5, 4, 1, 0)
        hypercube_faces += self.cube_faces(7, 6, 3, 2, 15, 14, 11, 10)
        hypercube_tetrahedrons = []
        hypercube_tetrahedrons += self.cube_tetrahedrons_optimised(0, 1, 2, 3, 4, 5, 6, 7, hypercube_points)
        hypercube_tetrahedrons += self.cube_tetrahedrons_optimised(8, 9, 10, 11, 12, 13, 14, 15, hypercube_points)
        hypercube_tetrahedrons += self.cube_tetrahedrons_optimised(1, 9, 3, 11, 5, 13, 7, 15, hypercube_points)
        hypercube_tetrahedrons += self.cube_tetrahedrons_optimised(8, 0, 10, 2, 12, 4, 14, 6, hypercube_points)
        hypercube_tetrahedrons += self.cube_tetrahedrons_optimised(2, 3, 10, 11, 6, 7, 14, 15, hypercube_points)
        hypercube_tetrahedrons += self.cube_tetrahedrons_optimised(8, 9, 0, 1, 12, 13, 4, 5, hypercube_points)
        hypercube_tetrahedrons += self.cube_tetrahedrons_optimised(4, 5, 6, 7, 12, 13, 14, 15, hypercube_points)
        hypercube_tetrahedrons += self.cube_tetrahedrons_optimised(8, 9, 10, 11, 0, 1, 2, 3, hypercube_points)
        Model4dv2.__init__(self, hypercube_points, hypercube_faces, col, hypercube_tetrahedrons)

    @staticmethod
    def cube_faces(p1, p2, p3, p4, p5, p6, p7, p8):
        return [(p1, p2, p3), (p4, p3, p2), (p4, p2, p6), (p8, p4, p6), (p1, p5, p6), (p2, p1, p6), (p7, p5, p1),
                (p3, p7, p1), (p8, p6, p5), (p7, p8, p5), (p7, p3, p4), (p8, p7, p4)]

    def cube_tetrahedrons(self, p1, p2, p3, p4, p5, p6, p7, p8, points_pos):
        faces = self.cube_faces(p1, p2, p3, p4, p5, p6, p7, p8)
        centre = (0, 0, 0, 0)
        for elt in (p1, p2, p3, p4, p5, p6, p7, p8):
            centre = (centre[0]+points_pos[elt][0], centre[1]+points_pos[elt][1], centre[2]+points_pos[elt][2],
                      centre[3]+points_pos[elt][3])
        centre = (centre[0]/8, centre[1]/8, centre[2]/8, centre[3]/8)
        tetrahedrons = ()
        for face in faces:
            tetrahedrons += (Tetrahedron(points_pos[face[0]], points_pos[face[1]], points_pos[face[2]], centre),)
        return tetrahedrons

    @staticmethod
    def cube_tetrahedrons_optimised(p1, p2, p3, p4, p5, p6, p7, p8, hypercube_points):
        p1 = hypercube_points[p1]
        p2 = hypercube_points[p2]
        p3 = hypercube_points[p3]
        p4 = hypercube_points[p4]
        p5 = hypercube_points[p5]
        p6 = hypercube_points[p6]
        p7 = hypercube_points[p7]
        p8 = hypercube_points[p8]
        return (Tetrahedron(p3, p1, p7, p4),
                Tetrahedron(p1, p2, p4, p6),
                Tetrahedron(p8, p7, p6, p4),
                Tetrahedron(p1, p7, p4, p6),
                Tetrahedron(p7, p5, p1, p6))


class Space:
    def __init__(self, point):
        # un espace défini par un point, un vecteur normal et un angle alpha
        assert len(point) == 4
        self.point = point
        self.normal = (0, 0, 0, 1)
        self.alpha = 0
        self.cartesian_e = -self.point[3]  # The e from the cartesian equation ax+by+cz+e = 0
        # The same as eval the cartesian_e by update_cartesian_e

    def update_cartesian_e(self):
        self.cartesian_e = -self.normal[0]*self.point[0] - self.normal[1]*self.point[1] - \
                           self.normal[2]*self.point[2] - self.normal[3]*self.point[3]

    def update(self, alpha):
        # change l'angle
        self.alpha = alpha
        self.normal = (-sin(alpha), 0, 0, cos(alpha))
        self.update_cartesian_e()

    def point_side(self, point):
        return self.normal[0]*point[0] + self.normal[1]*point[1] + self.normal[2]*point[2] + \
            self.normal[3]*point[3] + self.cartesian_e > 0

    def from4d_to_3d(self, points):
        new_points = ()
        for p in points:
            point = (p[0] - self.point[0], p[1] - self.point[1], p[2] - self.point[2], p[3] - self.point[3])
            new_points += ((cos(-self.alpha) * point[0] - sin(-self.alpha) * point[3],
                            point[1],
                            point[2],
                            sin(-self.alpha) * point[0] + cos(-self.alpha) * point[3]),)
        return new_points


class Line:
    def __init__(self, point1, point2):
        # une ligne définie par 2 points et par un vecteur directeur
        self.point1 = point1
        self.point2 = point2
        self.vecdir = (point2[0] - point1[0], point2[1] - point1[1], point2[2] - point1[2], point2[3] - point1[3])

    def spaceinter(self, space, segment=True):
        # point d'intersection entre un espace et le segment
        denom = self.vecdir[0] * space.normal[0] + self.vecdir[1] * space.normal[1] + self.vecdir[2] * space.normal[2] + \
                self.vecdir[3] * space.normal[3]
        num = (space.point[0] - self.point1[0]) * space.normal[0] + (space.point[1] - self.point1[1]) * space.normal[
            1] + (space.point[2] - self.point1[2]) * space.normal[2] + (space.point[3] - self.point1[3]) * space.normal[
                  3]
        if denom != 0 and num != 0:
            k = num / denom
            if 0 <= k <= 1 or not segment:
                return ((self.point1[0] + self.vecdir[0] * k, self.point1[1] + self.vecdir[1] * k,
                         self.point1[2] + self.vecdir[2] * k, self.point1[3] + self.vecdir[3] * k),)
        elif denom == num == 0:
            return self.point1, self.point2
        return ()


class Tetrahedron:
    def __init__(self, point1, point2, point3, point4):
        # un tétrahèdre défini par 6 segments
        self.points = (point1, point2, point3, point4)
        self.lines = (Line(point1, point2), Line(point2, point3), Line(point3, point1),
                      Line(point4, point1), Line(point2, point4), Line(point3, point4))
        self.points2line_index = {(0, 1): self.lines[0],
                                  (1, 2): self.lines[1],
                                  (2, 0): self.lines[2],
                                  (3, 0): self.lines[3],
                                  (1, 3): self.lines[4],
                                  (2, 3): self.lines[5]}
        self.useful_lines = {(False, False, False, False): (),
                             (False, False, False, True): (self.lines[3], self.lines[4], self.lines[5]),
                             (False, False, True, False): (self.lines[1], self.lines[2], self.lines[5]),
                             (False, False, True, True): (self.lines[1], self.lines[2], self.lines[3], self.lines[4]),
                             (False, True, False, False): (self.lines[0], self.lines[1], self.lines[4]),
                             (False, True, False, True): (self.lines[0], self.lines[1], self.lines[5], self.lines[3]),
                             (False, True, True, False): (self.lines[0], self.lines[2], self.lines[5], self.lines[4]),
                             (False, True, True, True): (self.lines[0], self.lines[2], self.lines[3]),
                             (True, False, False, False): (self.lines[0], self.lines[2], self.lines[3]),
                             (True, False, False, True): (self.lines[0], self.lines[2], self.lines[5], self.lines[4]),
                             (True, False, True, False): (self.lines[0], self.lines[1], self.lines[5], self.lines[3]),
                             (True, False, True, True): (self.lines[0], self.lines[1], self.lines[4]),
                             (True, True, False, False): (self.lines[1], self.lines[2], self.lines[3], self.lines[4]),
                             (True, True, False, True): (self.lines[1], self.lines[2], self.lines[5]),
                             (True, True, True, False): (self.lines[3], self.lines[4], self.lines[5]),
                             (True, True, True, True): ()}

    def spaceinter(self, space: Space):
        """ intersection entre chacun de ses segments et l'espace entré en paramètre """
        points = []
        sides = (space.point_side(self.points[0]),
                 space.point_side(self.points[1]),
                 space.point_side(self.points[2]),
                 space.point_side(self.points[3]))
        for line in self.useful_lines[sides]:
            points.append(space.from4d_to_3d(line.spaceinter(space)))
        return points


class Camera:  # le joueur défini par sa position, les angles d'orientation de son regard, etc
    def __init__(self):
        self.pos = [0, 0, -1.5, 0.1]
        self.radius_x = 0.4
        self.radius_y = 0.7
        self.fov = 90  # fov ist for "field of controller"
        self.factor = 1 / tan(radians(self.fov) / 2)
        self.z_near = 0.3
        self.z_far = 100
        self.q = self.z_far / (self.z_far - self.z_near)
        self.angle_z = 0
        self.angle_y = 0
        self.angle_x = 0
        self.dir = 0, 0, 1
        self.visible_space = Space(self.pos)

    def update(self):
        # change le vecteur du joueur selon son angle
        if self.angle_x > 1.1708:
            self.angle_x = 1.1708
        elif self.angle_x < -1.1708:
            self.angle_x = -1.1708
        v = (0, 0, 1)
        # Rotation in X-axis
        v = (v[0],
             v[1] * cos(-self.angle_x) + v[2] * sin(-self.angle_x),
             -v[1] * sin(-self.angle_x) + v[2] * cos(-self.angle_x),)
        v = (v[0] * cos(-self.angle_y) + v[2] * sin(-self.angle_y),
             v[1],
             -v[0] * sin(-self.angle_y) + v[2] * cos(-self.angle_y))
        self.dir = v

    def set_angle_wx(self, new_angle):
        """The new angle have to be in radiants"""
        self.visible_space.update(new_angle)


def key_sorted(distances):
    def distance_of_face(face):
        return distances[face]
    return distance_of_face


class Controller:
    def __init__(self):
        # le code commence par exécuter ceci!!
        # création de la fenêtre et initialisations
        self.screen_size = (1280, 660)
        self.screen = pg.display.set_mode(self.screen_size)  # , pg.FULLSCREEN)
        self.objects_v_pos = []
        self.objects = []
        pg.font.init()
        self.basic_font = pg.font.SysFont("Comic Sans MS", 20)
        pg.mouse.set_visible(False)
        pg.event.set_grab(True)
        self.debug = 0
        self.add_infos_ath = False
        self.screen_size = pg.display.get_surface().get_size()
        self.aspect_ratio = self.screen_size[1] / self.screen_size[0]
        self.light_dir = normalise((1, 1, 0))
        self.vertices = []
        self.faces = []
        self.objects.append(Hypercube((0, 0, 0, 0), (1, 1, 1, 2), (255, 10, 10)))

    def create_hypercube(self):
        x = int(random()*10)
        y = int(random()*10)
        z = int(random()*10)
        w = int(random()*10)
        self.objects.append(Hypercube((x, y, z, w), (x+1, y+1, z+1, w+1), (255, 255, 255)))

    def projection(self, vert, cam: Camera) -> list:
        # 3D à 2D
        cam.update()
        v = (vert[0] - cam.pos[0], vert[1] - cam.pos[1], vert[2] - cam.pos[2] + 0.0001)
        # Rotation in y-axis
        v = (v[0] * cos(cam.angle_y) + v[2] * sin(cam.angle_y),
             v[1],
             -v[0] * sin(cam.angle_y) + v[2] * cos(cam.angle_y))
        # Rotation in Z-axis
        v = (cos(cam.angle_z) * v[0] - sin(cam.angle_z) * v[1],
             sin(cam.angle_z) * v[0] + cos(cam.angle_z) * v[1],
             v[2])
        # Rotation in x-axis
        v = (v[0],
             v[1] * cos(cam.angle_x) + v[2] * sin(cam.angle_x),
             -v[1] * sin(cam.angle_x) + v[2] * cos(cam.angle_x),)
        v = [self.aspect_ratio * cam.factor * v[0] / v[2],
             cam.factor * v[1] / v[2],
             v[2] * cam.q - cam.q * cam.z_near]
        return v

    def collisions(self, cam):
        up_collision = False
        for obj in self.objects:
            mins, maxs = obj.get_xbox()
            x = max(mins[0] - cam.radius_x, min(cam.pos[0], maxs[0] + cam.radius_x))
            y = max(mins[1] - cam.radius_y, min(cam.pos[1], maxs[1] + cam.radius_y))
            z = max(mins[2] - cam.radius_x, min(cam.pos[2], maxs[2] + cam.radius_x))
            d = sqrt((x - cam.pos[0]) ** 2 + (y - cam.pos[1]) ** 2 + (z - cam.pos[2]) ** 2)
            if d < cam.radius_y:
                if y != cam.pos[1]:
                    vcx = x - cam.pos[0]
                    vcy = y - cam.pos[1] + int(y == maxs[1] + cam.radius_y) * \
                          cam.radius_y - int(y == mins[1] - cam.radius_y) * cam.radius_y
                    vcz = z - cam.pos[2]
                    if int(y == maxs[1] + cam.radius_y) * cam.radius_y - int(
                            y == mins[1] - cam.radius_y) * cam.radius_y < 0:
                        up_collision = True
                elif abs(x - cam.pos[0]) > abs(z - cam.pos[2]):
                    vcx = x - cam.pos[0] + int(x == maxs[0] + cam.radius_x) * cam.radius_x - int(
                        x == mins[0] - cam.radius_x) \
                          * cam.radius_x
                    vcy = y - cam.pos[1]
                    vcz = z - cam.pos[2]
                else:
                    vcx = x - cam.pos[0]
                    vcy = y - cam.pos[1]
                    vcz = z - cam.pos[2] + int(z == maxs[2] + cam.radius_x) * cam.radius_x - int(
                        z == mins[2] - cam.radius_x) \
                          * cam.radius_x
                cam.pos[0] += vcx
                cam.pos[1] += vcy
                cam.pos[2] += vcz
        return up_collision

    @staticmethod
    def splitting(points, plane_point, plane_normal):
        # static method est pour ne pas entrer le paramètre self, vous pouvez l'ignorer
        # permet de découper un triangle en 3d en plusieurs triangles en 3d ne dépassant pas le plan entré en paramètre
        plane_normal = normalise(plane_normal)
        distances = []
        inside_points = []
        outside_points = []
        nb_in = 0
        for i in range(len(points)):
            p = points[i]
            distances.append(plane_normal[0] * p[0] + plane_normal[1] * p[1] + plane_normal[2] * p[2] -
                             dot_product(plane_normal, plane_point))
            if distances[i] >= 0:
                inside_points.append(p)
                nb_in += 1
            else:
                outside_points.append(p)
        if nb_in == 0:
            return ()
        elif nb_in == 3:
            return points,
        elif nb_in == 1:
            return ((inside_points[0],
                     plane_line_inter(plane_point, plane_normal, inside_points[0], outside_points[0]),
                     plane_line_inter(plane_point, plane_normal, inside_points[0], outside_points[1])),)
        elif nb_in == 2:
            res1 = (inside_points[0],
                    inside_points[1],
                    plane_line_inter(plane_point, plane_normal, inside_points[0], outside_points[0]))
            res2 = (plane_line_inter(plane_point, plane_normal, inside_points[1], outside_points[0]),
                    res1[2],
                    inside_points[1])
            return res2, res1

    def objects_elements(self, cam):
        # met les éléments 4d sous forme de triangles 3d
        self.vertices = []
        self.faces = []
        self.objects_v_pos = []
        for obj in self.objects:
            v, f = obj.get_elements(len(self.vertices), cam)
            self.vertices += v
            self.faces += f
            self.objects_v_pos.append(len(self.faces))

    def set_debug(self):
        # active le mode debug 1
        self.debug += 1
        if self.debug > 2:
            self.debug = 0

    def set_debug2(self):
        # active le mode debug 2
        self.add_infos_ath = False if self.add_infos_ath else True

    def draw(self):
        # dessine tous les objets
        self.objects_elements(camera)
        self.screen.fill(black)
        faces = []
        sorted_faces = []
        distances = {}
        actual_obj = 0
        direction = normalise(camera.dir)
        near_point = (direction[0] * camera.z_near + camera.pos[0],
                      direction[1] * camera.z_near + camera.pos[1],
                      direction[2] * camera.z_near + camera.pos[2])
        far_point = (direction[0] * camera.z_far + camera.pos[0],
                     direction[1] * camera.z_far + camera.pos[1],
                     direction[2] * camera.z_far + camera.pos[2])
        fov = radians(camera.fov)
        left_fov = (direction[0] * cos(fov / 3) + direction[2] * sin(fov / 3),
                    direction[1],
                    -direction[0] * sin(fov / 3) + direction[2] * cos(fov / 3))
        right_fov = (direction[0] * cos(-fov / 3) + direction[2] * sin(-fov / 3),
                     direction[1],
                     -direction[0] * sin(-fov / 3) + direction[2] * cos(-fov / 3))
        up_fov = (0,
                  sin(-camera.angle_x + fov / 2),
                  cos(-camera.angle_x + fov / 2),)
        up_fov = (up_fov[0] * cos(-camera.angle_y) + up_fov[2] * sin(-camera.angle_y),
                  up_fov[1],
                  -up_fov[0] * sin(-camera.angle_y) + up_fov[2] * cos(-camera.angle_y))
        down_fov = (0,
                    sin(-camera.angle_x - fov / 2),
                    cos(-camera.angle_x - fov / 2),)
        down_fov = (down_fov[0] * cos(-camera.angle_y) + down_fov[2] * sin(-camera.angle_y),
                    down_fov[1],
                    -down_fov[0] * sin(-camera.angle_y) + down_fov[2] * cos(-camera.angle_y))
        for i in range(len(self.faces)):
            if i in self.objects_v_pos:
                actual_obj += 1
            face = self.faces[i]
            points_temp = [self.vertices[face[0]], self.vertices[face[1]], self.vertices[face[2]]]
            normal = normalise(normal_p_3d(points_temp[0], points_temp[1], points_temp[2]))
            center2face = (points_temp[0][0] - self.objects[actual_obj].center3d[0],
                           points_temp[0][1] - self.objects[actual_obj].center3d[1],
                           points_temp[0][2] - self.objects[actual_obj].center3d[2])
            if dot_product(normal, center2face) < 0:
                normal = (-normal[0], -normal[1], -normal[2])
                points_temp[0], points_temp[1] = points_temp[1], points_temp[0]
            #if dot_product(camera.dir, normal) > 0:
            #    continue
            projected = [self.projection(points_temp[0], camera),
                         self.projection(points_temp[1], camera),
                         self.projection(points_temp[2], camera)]
            normal = normalise(normal_p_3d(projected[0], projected[1], projected[2]))
            normal2 = normalise(normal_p_3d(points_temp[0], points_temp[1], points_temp[2]))
            luminosity = (-dot_product(normal2, self.light_dir) + 1) / 2
            #if not normal[2] < 0:
            #    continue
            triangles = self.splitting(points_temp, near_point, direction)
            tr_temp = ()
            for elt in triangles:
                tr_temp += self.splitting(elt, far_point, (-direction[0], -direction[1], -direction[2]))
            triangles = tr_temp
            tr_temp = ()
            for elt in triangles:
                tr_temp += self.splitting(elt, (camera.pos[0], camera.pos[1], camera.pos[2]), left_fov)
            triangles = tr_temp
            tr_temp = ()
            for elt in triangles:
                tr_temp += self.splitting(elt, (camera.pos[0], camera.pos[1], camera.pos[2]), right_fov)
            triangles = tr_temp
            tr_temp = ()
            for elt in triangles:
                tr_temp += self.splitting(elt, (camera.pos[0], camera.pos[1], camera.pos[2]), up_fov)
            triangles = tr_temp
            tr_temp = ()
            for elt in triangles:
                tr_temp += self.splitting(elt, (camera.pos[0], camera.pos[1], camera.pos[2]), down_fov)
            triangles = tr_temp
            for elt in triangles:
                projected = [self.projection(elt[0], camera),
                             self.projection(elt[1], camera),
                             self.projection(elt[2], camera)]
                points = ()
                for o in range(len(face)):
                    points += ((int((projected[o][0] + 1) * self.screen_size[0] / 2),
                                   int((projected[o][1] + 1) * self.screen_size[1] / 2)),)
                if points[0] == points[1] or points[1] == points[2] or points[2] == points[0]:
                    continue
                d1 = (camera.pos[0] - elt[0][0]) ** 2 + (camera.pos[1] - elt[0][1]) ** 2 + \
                     (camera.pos[2] - elt[0][2]) ** 2
                d2 = (camera.pos[0] - elt[1][0]) ** 2 + (camera.pos[1] - elt[1][1]) ** 2 + \
                     (camera.pos[2] - elt[1][2]) ** 2
                d3 = (camera.pos[0] - elt[2][0]) ** 2 + (camera.pos[1] - elt[2][1]) ** 2 + \
                     (camera.pos[2] - elt[2][2]) ** 2
                faces.append((points, luminosity, self.objects[actual_obj]))
                distances[faces[-1]] = moy((d1, d2, d3))
        # sorted_faces = faces
        """
        for _ in range(len(faces)):
            elt = max(distances)
            i = distances.index(elt)
            sorted_faces.append(faces[i])
            faces.pop(i)
            distances.pop(i)"""
        key_sort = key_sorted(distances)
        sorted_faces = sorted(faces, key=key_sort, reverse=True)
        for face in sorted_faces:
            points, luminosity, obj = face
            #luminosity = 0.7
            color = obj.get_color()
            if self.debug <= 1:
                pg.draw.polygon(self.screen, (color[0] * luminosity, color[1] * luminosity, color[2] * luminosity),
                                points)
            if self.debug >= 1:
                pg.draw.line(self.screen, white, points[0], points[1])
                pg.draw.line(self.screen, white, points[1], points[2])
                pg.draw.line(self.screen, white, points[0], points[2])
        half_screen = self.screen_size[0] / 2
        pg.draw.rect(self.screen, (0, 0, 0), pg.Rect(half_screen - 402, 28, 804, 9))
        pg.draw.rect(self.screen, (0, 0, 0), pg.Rect(half_screen - 403, 23, 6, 19))
        pg.draw.rect(self.screen, (0, 0, 0), pg.Rect(half_screen + 398, 23, 6, 19))
        pg.draw.rect(self.screen, (255, 255, 255), pg.Rect(half_screen - 400, 30, 800, 5))
        pg.draw.rect(self.screen, (255, 255, 255), pg.Rect(half_screen - 401, 25, 2, 15))
        pg.draw.rect(self.screen, (255, 255, 255), pg.Rect(half_screen + 400, 25, 2, 15))
        pg.draw.rect(self.screen, (255, 255, 0), pg.Rect(half_screen + camera.pos[3] * 40 - 2, 18, 4, 29))
        #  mini map
        pg.draw.rect(self.screen, (40, 40, 0), pg.Rect(20, 20, 100, 100))
        for elt in self.objects:
            x1 = 70 + (elt.mini_map_p1[0] - camera.pos[0]) * 5
            w1 = 70 + (elt.mini_map_p1[1] - camera.pos[3]) * 5
            x2 = (elt.mini_map_p2[0]-elt.mini_map_p1[0]) * 5
            w2 = (elt.mini_map_p2[1] - elt.mini_map_p1[1]) * 5
            if x1 <= 20:
                x2 = max(x2 - 20 + x1, 0)
                x1 = 20
            if w1 <= 20:
                w2 = max(w2 - 20 + w1, 0)
                w1 = 20
            if x1+x2 >= 120:
                x2 = -x1+120
            if w1+w2 >= 120:
                w2 = -w1+120
            pg.draw.rect(self.screen, (180, 180, 180), pg.Rect(x1, w1, x2, w2))
        alpha = camera.visible_space.alpha
        mini_map_vd = (cos(alpha), sin(alpha))
        pg.draw.line(self.screen, (255, 255, 255), (70, 70), (70 + mini_map_vd[0] * 50, 70 + mini_map_vd[1] * 50))
        clock.tick()
        fps = clock.get_fps()
        if self.add_infos_ath:
            fps_hud = self.basic_font.render("fps: "+str(fps), False, (255, 255, 255))
            self.screen.blit(fps_hud, (0, 0))
            player_pos_hud = self.basic_font.render("player position: "+str((camera.pos[0], camera.pos[1], camera.pos[2], camera.pos[3])), False,
                                                    (255, 255, 255))
            self.screen.blit(player_pos_hud, (0, 40))
            number_objhects = self.basic_font.render("number of objects: "+str(len(self.objects)), False, (255, 255, 255))
            self.screen.blit(number_objhects, (0, 80))
        pg.display.flip()
        return fps


controller = Controller()
# controller.import_element("E.obj")
camera = Camera()
end = True
fps_balancing = 1
exist_gravity = False
force_y = 0
w_angle = 0
w_rotation = False
right = left = up = down = pu = pd = fd_left = fd_right = False
f7 = False
c_up = False
clicks = (False, False, False)
clock = pg.time.Clock()
clock.tick()
controller.draw()
while end:
    # boucle principale
    for event in pg.event.get():  # controle des évènements (touches appuyées)
        if event.type == pg.QUIT:
            end = False
            pg.quit()
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_RIGHT or event.key == pg.K_d:
                right = True
            elif event.key == pg.K_LEFT or event.key == pg.K_q:
                left = True
            elif event.key == pg.K_UP or event.key == pg.K_z or event.key == pg.K_w:
                up = True
            elif event.key == pg.K_DOWN or event.key == pg.K_s:
                down = True
            elif event.key == pg.K_PAGEUP or event.key == pg.K_SPACE:
                pu = True
            elif event.key == pg.K_PAGEDOWN or event.key == pg.K_LSHIFT:
                pd = True
            elif event.key == pg.K_a:
                fd_left = True
            elif event.key == pg.K_e:
                fd_right = True
            elif event.key == pg.K_ESCAPE:
                pg.mouse.set_visible(True)
                pg.event.set_grab(False)
            elif event.key == pg.K_F4:
                exist_gravity = False if exist_gravity else True
            elif event.key == pg.K_F6:
                w_rotation = True
        elif event.type == pg.KEYUP:
            if event.key == pg.K_RIGHT or event.key == pg.K_d:
                right = False
            elif event.key == pg.K_LEFT or event.key == pg.K_q or event.key == pg.K_a:
                left = False
            elif event.key == pg.K_UP or event.key == pg.K_z or event.key == pg.K_w:
                up = False
            elif event.key == pg.K_DOWN or event.key == pg.K_s:
                down = False
            elif event.key == pg.K_PAGEUP or event.key == pg.K_SPACE:
                pu = False
            elif event.key == pg.K_PAGEDOWN or event.key == pg.K_LSHIFT:
                pd = False
            elif event.key == pg.K_a:
                fd_left = False
            elif event.key == pg.K_e:
                fd_right = False
            elif event.key == pg.K_F3:
                controller.set_debug()
            elif event.key == pg.K_F5:
                controller.set_debug2()
            elif event.key == pg.K_F6:
                w_rotation = False
            elif event.key == pg.K_F7:
                f7 = True
        elif event.type == pg.MOUSEMOTION and pg.event.get_grab():
            camera.angle_x -= event.rel[1] / 100
            camera.angle_y -= event.rel[0] / 100
        clicks = pg.mouse.get_pressed()
    if clicks[0]:
        if pg.mouse.get_visible(): pg.mouse.set_visible(False)
        if not pg.event.get_grab(): pg.event.set_grab(True)
        w_angle -= 1.5707963267948966 / 300 * fps_balancing
        camera.set_angle_wx(w_angle)
    if clicks[1]:
        pass
    if clicks[2]:
        w_angle += 1.5707963267948966 / 300 * fps_balancing
        camera.set_angle_wx(w_angle)
    if w_rotation:
        pass
    if right:
        vx, vy = rotate(1, 0, -degrees(camera.angle_y))
        vx, vw = (cos(camera.visible_space.alpha) * vx, sin(camera.visible_space.alpha) * vx)
        camera.pos[0] += vx / 100 * fps_balancing
        camera.pos[2] += vy / 100 * fps_balancing
        camera.pos[3] += vw / 100 * fps_balancing
    if left:
        vx, vy = rotate(-1, 0, -degrees(camera.angle_y))
        vx, vw = (cos(camera.visible_space.alpha) * vx, sin(camera.visible_space.alpha) * vx)
        camera.pos[0] += vx / 100 * fps_balancing
        camera.pos[2] += vy / 100 * fps_balancing
        camera.pos[3] += vw / 100 * fps_balancing
    if up:
        vx, vy = rotate(0, 1, -degrees(camera.angle_y))
        vx, vw = (cos(camera.visible_space.alpha)*vx, sin(camera.visible_space.alpha)*vx)
        camera.pos[0] += vx / 100 * fps_balancing
        camera.pos[2] += vy / 100 * fps_balancing
        camera.pos[3] += vw / 100 * fps_balancing
    if down:
        vx, vy = rotate(0, -1, -degrees(camera.angle_y))
        vx, vw = (cos(camera.visible_space.alpha) * vx, sin(camera.visible_space.alpha) * vx)
        camera.pos[0] += vx / 100 * fps_balancing
        camera.pos[2] += vy / 100 * fps_balancing
        camera.pos[3] += vw / 100 * fps_balancing
    if pu:
        if not exist_gravity:
            camera.pos[1] -= 0.01 * fps_balancing
        elif c_up:
            force_y -= 0.035 * fps_balancing
    if pd:
        camera.pos[1] += 0.01 * fps_balancing
    if exist_gravity:
        camera.pos[1] += force_y
        force_y += 0.0003 * fps_balancing ** 2
        if force_y >= 0.3: force_y = 0.3
    c_up = controller.collisions(camera)
    if c_up: force_y = 0

    if f7:
        controller.create_hypercube()
        f7 = False
    fps = controller.draw()
    if fps != 0:
        fps_balancing = 300 / fps
