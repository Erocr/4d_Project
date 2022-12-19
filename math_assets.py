from math import *


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


print(intersection_point(0, 0, 90, 10, 10, 10, -10))
