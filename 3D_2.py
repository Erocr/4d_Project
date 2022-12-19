from math_assets import *
import pygame as pg

white = (255, 255, 255)
black = (0, 0, 0)


class Model4d:
    def __init__(self, faces, w_pos: tuple, vertices1=None, vertices2=None, functions=None, color=white, rh=None,
                 rv=None, position=(0, 0, 0)):
        # assert functions is not None and vertices2 is not None, "Il faut soit des fonctions pour les sommets, soit " \
        #                                                        "les sommets de fin"
        self.faces = faces
        self.w_start = w_pos[0]
        self.w_end = w_pos[1]
        self.color = color
        self.vert_start = None
        self.vert_end = None
        self.vertices = []
        if functions is not None:
            self.functions = functions
        elif vertices1 is not None and vertices2 is not None:
            self.functions = []
            for i in range(len(vertices1)):
                vert1 = vertices1[i][0]+position[0], vertices1[i][1]+position[1], vertices1[i][2]+position[2]
                vert2 = vertices2[i][0] + position[0], vertices2[i][1] + position[1], vertices2[i][2] + position[2]
                self.functions.append(self.vertex_function(vert1, w_pos[0], vert2, w_pos[1]))
        elif vertices1 is not None and rh is not None and rv is not None:
            self.functions = []
            for i in range(len(vertices1)):
                vert1 = vertices1[i][0] + position[0], vertices1[i][1] + position[1], vertices1[i][2] + position[2]
                self.functions.append(self.rotating_function(vert1, rh, rv))
        else:
            self.functions = []
            for i in range(len(vertices1)):
                vert = vertices1[i][0] + position[0], vertices1[i][1] + position[1], vertices1[i][2] + position[2]
                self.functions.append(self.static_function(vert))

    @staticmethod
    def vertex_function(ver1: tuple, w1: float, ver2, w2):
        dw = (w1 - w2)
        mx = (ver1[0] - ver2[0]) / dw
        my = (-ver1[1] + ver2[1]) / dw
        mz = (ver1[2] - ver2[2]) / dw
        if mx == 0 or w1 == 0: px = ver1[0]
        else: px = ver1[0] - mx * w1
        if my == 0 or w1 == 0: py = -ver1[1]
        else: py = -ver1[1] - my * w1
        if mz == 0 or w1 == 0: pz = ver1[2]
        else: pz = ver1[2] - mz * w1
        return lambda w: mx * w + px, lambda w: my * w + py, lambda w: mz * w + pz

    @staticmethod
    def rotating_function(ver1: tuple, speed_h, speed_v):
        rh = speed_h*pi/10
        rv = 0
        d, theta, phi = cart_to_pol(*ver1)
        return (lambda w: d * sin(theta-w*rv) * cos(phi+w*rh),
                lambda w: d * cos(theta-w*rv),
                lambda w: d * sin(theta-w*rv) * sin(phi+w*rh))

    @staticmethod
    def static_function(ver):
        return (lambda w: ver[0],
                lambda w: ver[1],
                lambda w: ver[2])

    def get_elements(self, v_start, w):
        self.vert_start = v_start
        self.vert_end = v_start + len(self.functions)
        faces = []
        if not (self.w_start < w < self.w_end):
            return [], []
        for i in range(len(self.faces)):
            faces.append((self.faces[i][0] + v_start, self.faces[i][1] + v_start, self.faces[i][2] + v_start))
        return self.vertices, faces

    def get_xbox(self, w):
        self.vertices = []
        max_x = max_y = max_z = -100000
        min_x = min_y = min_z = 100000
        if self.w_start < w < self.w_end:
            for fs in self.functions:
                self.vertices.append((fs[0](w), fs[1](w), fs[2](w)))
                if self.vertices[-1][0] > max_x: max_x = self.vertices[-1][0]
                elif self.vertices[-1][0] < min_x: min_x = self.vertices[-1][0]
                if self.vertices[-1][1] > max_y: max_y = self.vertices[-1][1]
                elif self.vertices[-1][1] < min_y: min_y = self.vertices[-1][1]
                if self.vertices[-1][2] > max_z: max_z = self.vertices[-1][2]
                elif self.vertices[-1][2] < min_z: min_z = self.vertices[-1][2]
        return (min_x, min_y, min_z), (max_x, max_y, max_z)

    def get_color(self):
        return self.color


class Camera:
    def __init__(self):
        self.x = 0.5
        self.y = 0.5
        self.z = -5
        self.w = 0
        self.radius_x = 0.4
        self.radius_y = 0.7
        self.fov = 90  # fov ist for "field of controller"
        self.factor = 1 / tan(radians(self.fov) / 2)
        self.z_near = 0.3
        self.z_far = 50
        self.q = self.z_far / (self.z_far - self.z_near)
        self.angle_z = 0
        self.angle_y = 0
        self.angle_x = 0
        self.dir = 0, 0, 1

    def update(self):
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


class Controller:
    def __init__(self):
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
        points1 = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
        points2 = [(0, 0, 0), (1, 0, 0), (0.5, 1, 0.5), (0.5, 1, 0.5), (0, 0, 1), (1, 0, 1), (0.5, 1, 0.5),
                   (0.5, 1, 0.5)]
        points3 = [(0, 0, 0), (10, 0, 0), (0, 10, 0), (10, 10, 0), (0, 0, 10), (10, 0, 10), (0, 10, 10), (10, 10, 10)]
        rads = pi/1
        functions1 = [(lambda w: cos(w*rads+pi*0.25), lambda w: sin(w*rads+pi*0.25), lambda w: 0),
                      (lambda w: cos(w*rads+pi*0.75), lambda w: sin(w*rads+pi*0.75), lambda w: 0),
                      (lambda w: cos(w*rads+pi*1.75), lambda w: sin(w*rads+pi*1.75), lambda w: 0),
                      (lambda w: cos(w*rads+pi*1.25), lambda w: sin(w*rads+pi*1.25), lambda w: 0),
                      (lambda w: cos(w*rads+pi*0.25), lambda w: sin(w*rads+pi*0.25), lambda w: 1),
                      (lambda w: cos(w*rads+pi*0.75), lambda w: sin(w*rads+pi*0.75), lambda w: 1),
                      (lambda w: cos(w*rads+pi*1.75), lambda w: sin(w*rads+pi*1.75), lambda w: 1),
                      (lambda w: cos(w*rads+pi*1.25), lambda w: sin(w*rads+pi*1.25), lambda w: 1)]
        ws = (-5, 5)
        faces = [(1, 0, 2), (3, 1, 2), (5, 1, 3), (7, 5, 3), (5, 4, 0), (1, 5, 0), (4, 5, 7), (6, 4, 7),
                 (0, 4, 6), (2, 0, 6), (3, 2, 6), (7, 3, 6)]
        self.objects.append(Model4d(faces, ws, vertices1=points1, rh=3, rv=0, color=(100, 0, 185), position=(0, 0, 0)))
        self.objects.append(Model4d(faces, ws, vertices1=points1, rh=-2, rv=0, color=(100, 0, 185), position=(2, 0, 2)))
        self.objects.append(Model4d(faces, ws, vertices1=points1, rh=1, rv=0, color=(100, 0, 185), position=(2, 1, 5)))
        self.objects.append(Model4d(faces, ws, vertices1=points1, rh=1, rv=0, color=(100, 0, 185), position=(3, 0.5, 7)))
        self.objects.append(Model4d(faces, ws, vertices1=points1, color=(0, 0, 185), position=(-11, 0, 3.5)))
        self.objects.append(Model4d([(0, 1, 2), (2, 1, 0)], (-10, 10), vertices1=[points1[1], points1[0], points1[2]],))

    def projection(self, vert, cam: Camera) -> list:
        cam.update()
        v = (vert[0] - cam.x, vert[1] - cam.y, vert[2] - cam.z + 0.0001)
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
            mins, maxs = obj.get_xbox(cam.w)
            x = max(mins[0]-cam.radius_x, min(cam.x, maxs[0]+cam.radius_x))
            y = max(mins[1]-cam.radius_y, min(cam.y, maxs[1]+cam.radius_y))
            z = max(mins[2]-cam.radius_x, min(cam.z, maxs[2]+cam.radius_x))
            d = sqrt((x-cam.x)**2+(y-cam.y)**2+(z-cam.z)**2)
            if d < cam.radius_y:
                if y != cam.y:
                    vcx = x - cam.x
                    vcy = y - cam.y+int(y == maxs[1]+cam.radius_y)*\
                          cam.radius_y-int(y == mins[1]-cam.radius_y)*cam.radius_y
                    vcz = z - cam.z
                    if int(y == maxs[1]+cam.radius_y) * cam.radius_y-int(y == mins[1]-cam.radius_y) * cam.radius_y < 0:
                        up_collision = True
                elif abs(x - cam.x) > abs(z - cam.z):
                    vcx = x - cam.x + int(x == maxs[0]+cam.radius_x) * cam.radius_x - int(x == mins[0]-cam.radius_x)\
                          * cam.radius_x
                    vcy = y - cam.y
                    vcz = z - cam.z
                else:
                    vcx = x - cam.x
                    vcy = y - cam.y
                    vcz = z - cam.z + int(z == maxs[2]+cam.radius_x) * cam.radius_x - int(z == mins[2]-cam.radius_x)\
                          * cam.radius_x
                cam.x += vcx
                cam.y += vcy
                cam.z += vcz
        return up_collision

    @staticmethod
    def import_element(file, position=(0, 0, 0), color=(255, 255, 255)):
        f = open(file, "r")
        line = f.readline()
        vertices = []
        faces = []
        min_vert_x = 100000000000
        min_vert_y = 100000000000
        min_vert_z = 100000000000
        while line[0] != "v": line = f.readline()
        while line[0:2] == "v ":
            point = line[3:].split(" ")
            point = float(point[0]), -float(point[1]), -float(point[2])
            if point[0] < min_vert_x: min_vert_x = point[0]
            if point[1] < min_vert_y: min_vert_y = point[1]
            if point[2] < min_vert_z: min_vert_z = point[2]
            vertices.append(point)
            line = f.readline()
        add_x = -min_vert_x + position[0]
        add_y = -min_vert_y + position[1]
        add_z = -min_vert_z + position[2]
        for i in range(len(vertices)):
            vertices[i] = (vertices[i][0] + add_x, vertices[i][1] + add_y, vertices[i][2] + add_z)
        while line[0] != "f": line = f.readline()
        while len(line) > 0 and line[0] == "f":
            face = line[2:].split(" ")
            if face[-1] == "\n": face.pop(-1)
            if len(face) == 4:
                f1 = (int(face[2].split("/")[0]) - 1, int(face[1].split("/")[0]) - 1, int(face[0].split("/")[0]) - 1)
                faces.append(f1)
                f2 = (int(face[0].split("/")[0]) - 1, int(face[3].split("/")[0]) - 1,
                      int(face[2].split("/")[0]) - 1)
                faces.append(f2)
            elif len(face) == 5:
                f1 = (int(face[0].split("/")[0]) - 1, int(face[4].split("/")[0]) - 1,
                      int(face[1].split("/")[0]) - 1)
                faces.append(f1)
                f2 = (int(face[3].split("/")[0]) - 1, int(face[1].split("/")[0]) - 1,
                      int(face[4].split("/")[0]) - 1)
                faces.append(f2)
                f3 = (int(face[3].split("/")[0]) - 1, int(face[1].split("/")[2]) - 1,
                      int(face[1].split("/")[0]) - 1)
                faces.append(f3)
            elif len(face) == 6:
                f1 = (int(face[0].split("/")[0]) - 1, int(face[2].split("/")[0]) - 1,
                      int(face[1].split("/")[0]) - 1)
                faces.append(f1)
                f2 = (int(face[0].split("/")[0]) - 1, int(face[3].split("/")[0]) - 1,
                      int(face[2].split("/")[0]) - 1)
                faces.append(f2)
                f3 = (int(face[0].split("/")[0]) - 1, int(face[4].split("/")[0]) - 1,
                      int(face[3].split("/")[0]) - 1)
                faces.append(f3)
                f4 = (int(face[0].split("/")[0]) - 1, int(face[5].split("/")[0]) - 1,
                      int(face[4].split("/")[0]) - 1)
                faces.append(f4)
            line = f.readline()
        f.close()
        # self.objects.append(Model4d(vertices, faces, (0, 10), color=color))

    @staticmethod
    def splitting_polygon(face):
        faces = []
        for i in range(1, len(face) - 1):
            faces.append((face[0], face[i], face[i + 1]))
        return faces

    @staticmethod
    def splitting(points, plane_point, plane_normal):
        plane_normal = normalise(plane_normal)
        distancies = []
        inside_points = []
        outside_points = []
        nb_in = 0
        for i in range(len(points)):
            p = points[i]
            distancies.append(plane_normal[0]*p[0]+plane_normal[1]*p[1]+plane_normal[2]*p[2] -
                              dot_product(plane_normal, plane_point))
            if distancies[i] >= 0:
                inside_points.append(p)
                nb_in += 1
            else: outside_points.append(p)
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
        self.vertices = []
        self.faces = []
        self.objects_v_pos = []
        for obj in self.objects:
            v, f = obj.get_elements(len(self.vertices), cam.w)
            self.vertices += v
            self.faces += f
            self.objects_v_pos.append(len(self.faces))

    def set_debug(self):
        self.debug += 1
        if self.debug > 2:
            self.debug = 0

    def set_debug2(self):
        self.add_infos_ath = False if self.add_infos_ath else True

    def draw(self):
        self.objects_elements(camera)
        self.screen.fill(black)
        faces = []
        sorted_faces = []
        distances = []
        actual_obj = 0
        direction = normalise(camera.dir)
        near_point = (direction[0] * camera.z_near + camera.x,
                      direction[1] * camera.z_near + camera.y,
                      direction[2] * camera.z_near + camera.z)
        far_point = (direction[0] * camera.z_far + camera.x,
                     direction[1] * camera.z_far + camera.y,
                     direction[2] * camera.z_far + camera.z)
        fov = radians(camera.fov)
        left_fov = (direction[0] * cos(fov/3) + direction[2] * sin(fov/3),
                    direction[1],
                    -direction[0] * sin(fov/3) + direction[2] * cos(fov/3))
        right_fov = (direction[0] * cos(-fov/3) + direction[2] * sin(-fov/3),
                     direction[1],
                     -direction[0] * sin(-fov/3) + direction[2] * cos(-fov/3))
        #right_fov = (0 * cos(-camera.angle_y) + right_fov[2] * sin(-camera.angle_y),
        #             right_fov[1],
        #             -right_fov[0] * sin(-camera.angle_y) + right_fov[2] * cos(-camera.angle_y))
        #right_fov = (right_fov[0],
        #             right_fov[1] * cos(-camera.angle_x) + right_fov[2] * sin(-camera.angle_x),
        #             -right_fov[1] * sin(-camera.angle_x) + right_fov[2] * cos(-self.angle_x),)
        up_fov = (0,
                  sin(-camera.angle_x+fov/1.8),
                  cos(-camera.angle_x+fov/1.8),)
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
            projected = [self.projection(points_temp[0], camera),
                         self.projection(points_temp[1], camera),
                         self.projection(points_temp[2], camera)]
            normal = normalise(normal_p_3d(projected[1], projected[0], projected[2]))
            normal2 = normalise(normal_p_3d(points_temp[0], points_temp[1], points_temp[2]))
            luminosity = (dot_product(normal2, self.light_dir) + 1) / 2
            if normal[2] <= 0:
                triangles = self.splitting(points_temp, near_point, direction)
                tr_temp = ()
                for elt in triangles:
                    tr_temp += self.splitting(elt, far_point, (-direction[0], -direction[1], -direction[2]))
                triangles = tr_temp
                tr_temp = ()
                for elt in triangles:
                    tr_temp += self.splitting(elt, (camera.x, camera.y, camera.z), left_fov)
                triangles = tr_temp
                tr_temp = ()
                for elt in triangles:
                    tr_temp += self.splitting(elt, (camera.x, camera.y, camera.z), right_fov)
                triangles = tr_temp
                tr_temp = ()
                for elt in triangles:
                    tr_temp += self.splitting(elt, (camera.x, camera.y, camera.z), up_fov)
                triangles = tr_temp
                tr_temp = ()
                for elt in triangles:
                    tr_temp += self.splitting(elt, (camera.x, camera.y, camera.z), down_fov)
                triangles = tr_temp
                for elt in triangles:
                    projected = [self.projection(elt[0], camera),
                                 self.projection(elt[1], camera),
                                 self.projection(elt[2], camera)]
                    points = []
                    for o in range(len(face)):
                        points.append((int((projected[o][0] + 1) * self.screen_size[0] / 2),
                                       int((projected[o][1] + 1) * self.screen_size[1] / 2)))
                    if points[0] == points[1] or points[1] == points[2] or points[2] == points[0]:
                        continue
                    d1 = (camera.x - elt[0][0]) ** 2 + (camera.y - elt[0][1]) ** 2 + \
                         (camera.z - elt[0][2]) ** 2
                    d2 = (camera.x - elt[1][0]) ** 2 + (camera.y - elt[1][1]) ** 2 + \
                         (camera.z - elt[1][2]) ** 2
                    d3 = (camera.x - elt[2][0]) ** 2 + (camera.y - elt[2][1]) ** 2 + \
                         (camera.z - elt[2][2]) ** 2
                    faces.append((points, luminosity, self.objects[actual_obj]))
                    distances.append(moy((d1, d2, d3)))
        #sorted_faces = faces
        for _ in range(len(faces)):
            elt = max(distances)
            i = distances.index(elt)
            sorted_faces.append(faces[i])
            faces.pop(i)
            distances.pop(i)
        for face in sorted_faces:
            points, luminosity, obj = face
            color = obj.get_color()
            if self.debug <= 1:
                pg.draw.polygon(self.screen, (color[0]*luminosity, color[1]*luminosity, color[2]*luminosity), points)
            if self.debug >= 1:
                pg.draw.line(self.screen, white, points[0], points[1])
                pg.draw.line(self.screen, white, points[1], points[2])
                pg.draw.line(self.screen, white, points[0], points[2])
        half_screen = self.screen_size[0]/2
        pg.draw.rect(self.screen, (0, 0, 0), pg.Rect(half_screen-402, 28, 804, 9))
        pg.draw.rect(self.screen, (0, 0, 0), pg.Rect(half_screen-403, 23, 6, 19))
        pg.draw.rect(self.screen, (0, 0, 0), pg.Rect(half_screen+398, 23, 6, 19))
        pg.draw.rect(self.screen, (255, 255, 255), pg.Rect(half_screen-400, 30, 800, 5))
        pg.draw.rect(self.screen, (255, 255, 255), pg.Rect(half_screen-401, 25, 2, 15))
        pg.draw.rect(self.screen, (255, 255, 255), pg.Rect(half_screen+400, 25, 2, 15))
        pg.draw.rect(self.screen, (255, 255, 0), pg.Rect(half_screen+camera.w*40-2, 18, 4, 29))
        if self.add_infos_ath:
            player_pos_hud = self.basic_font.render(str((camera.x, camera.y, camera.z)), False, (255, 255, 255))
            self.screen.blit(player_pos_hud, (0, 0))
        pg.display.flip()


controller = Controller()
camera = Camera()
end = True
speed = 1
exist_gravity = False
force_y = 0
right = left = up = down = pu = pd = False
c_up = False
clicks = (False, False, False)
while end:
    for event in pg.event.get():
        clicks = pg.mouse.get_pressed()
        if event.type == pg.QUIT:
            end = False
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_RIGHT or event.key == pg.K_d:
                right = True
            elif event.key == pg.K_LEFT or event.key == pg.K_q or event.key == pg.K_a:
                left = True
            elif event.key == pg.K_UP or event.key == pg.K_z or event.key == pg.K_w:
                up = True
            elif event.key == pg.K_DOWN or event.key == pg.K_s:
                down = True
            elif event.key == pg.K_PAGEUP or event.key == pg.K_SPACE:
                pu = True
            elif event.key == pg.K_PAGEDOWN or event.key == pg.K_LSHIFT:
                pd = True
            elif event.key == pg.K_ESCAPE:
                pg.mouse.set_visible(True)
                pg.event.set_grab(False)
            elif event.key == pg.K_F4:
                exist_gravity = False if exist_gravity else True
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
            elif event.key == pg.K_F3:
                controller.set_debug()
            elif event.key == pg.K_F5:
                controller.set_debug2()
        elif event.type == pg.MOUSEMOTION and pg.event.get_grab():
            camera.angle_x -= event.rel[1] / 100
            camera.angle_y -= event.rel[0] / 100
    if clicks[0]:
        if pg.mouse.get_visible(): pg.mouse.set_visible(False)
        if not pg.event.get_grab(): pg.event.set_grab(True)
        camera.w -= 0.01*speed
        if camera.w < -10: camera.w = -10
    if clicks[1]:
        pass
    if clicks[2]:
        camera.w += 0.01*speed
        if camera.w > 10: camera.w = 10
    if right:
        vx, vy = rotate(1, 0, -degrees(camera.angle_y))
        camera.x += vx / 100 * speed
        camera.z += vy / 100 * speed
    if left:
        vx, vy = rotate(-1, 0, -degrees(camera.angle_y))
        camera.x += vx / 100 * speed
        camera.z += vy / 100 * speed
    if up:
        vx, vy = rotate(0, 1, -degrees(camera.angle_y))
        camera.x += vx / 100 * speed
        camera.z += vy / 100 * speed
    if down:
        vx, vy = rotate(0, -1, -degrees(camera.angle_y))
        camera.x += vx / 100 * speed
        camera.z += vy / 100 * speed
    if pu:
        if not exist_gravity:
            camera.y -= 0.01 * speed
        elif c_up:
            force_y -= 0.015
    if pd:
        camera.y += 0.01 * speed
    if exist_gravity:
        camera.y += force_y
        force_y += 0.0001
    c_up = controller.collisions(camera)
    if c_up: force_y = 0
    controller.draw()
