from math_assets import *
import pygame as pg

white = (255, 255, 255)
black = (0, 0, 0)


class Model4d:
    def __init__(self, faces, w_pos: tuple, vertices1=None, vertices2=None, functions=None, color=white):
        # assert functions is not None and vertices2 is not None, "Il faut soit des fonctions pour les sommets soit " \
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
                self.functions.append(self.vertex_function(vertices1[i], w_pos[0], vertices2[i], w_pos[1]))

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
        self.radius = 0.3
        self.fov = 90  # fov ist for "field of controller"
        self.factor = 1 / tan(radians(self.fov) / 2)
        self.znear = 0.1
        self.zfar = 10
        self.q = self.zfar / (self.zfar - self.znear)
        self.angle_z = 0
        self.angle_y = 0
        self.angle_x = 0
        self.dir_x = 0, 0, 1
        self.dir_y = 0, 0, 1

    def update(self):
        if self.angle_x > 0.5854:
            self.angle_x = 0.5854
        elif self.angle_x < -0.5854:
            self.angle_x = -0.5854
        v2 = (0, 0, 1)
        # Rotation in X-axis
        v2 = (v2[0],
              v2[1] * cos(-self.angle_x) + v2[2] * sin(-self.angle_x),
              -v2[1] * sin(-self.angle_x) + v2[2] * cos(-self.angle_x),)
        v = (0, 0, 1)
        v = (v[0] * cos(-self.angle_y) + v[2] * sin(-self.angle_y),
             v[1],
             -v[0] * sin(-self.angle_y) + v[2] * cos(-self.angle_y))
        v2 = (v2[0] * cos(-self.angle_y) + v2[2] * sin(-self.angle_y),
              v2[1],
              -v2[0] * sin(-self.angle_y) + v2[2] * cos(-self.angle_y))
        self.dir_x = v
        self.dir_y = v2


class Controller:
    def __init__(self):
        self.screen_size = (1280, 660)
        self.screen = pg.display.set_mode(self.screen_size) #, pg.FULLSCREEN)
        self.objects_v_pos = []
        self.objects = []
        pg.mouse.set_visible(False)
        pg.event.set_grab(True)
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
        functions1 = [(lambda w: cos(w*rads+pi*0.25), lambda w: sin(w*rads+pi*0.25), lambda w: 0), (lambda w: cos(w*rads+pi*0.75), lambda w: sin(w*rads+pi*0.75), lambda w: 0), \
        (lambda w: cos(w*rads+pi*1.75), lambda w: sin(w*rads+pi*1.75), lambda w: 0), (lambda w: cos(w*rads+pi*1.25), lambda w: sin(w*rads+pi*1.25), lambda w: 0), \
        (lambda w: cos(w*rads+pi*0.25), lambda w: sin(w*rads+pi*0.25), lambda w: 1), (lambda w: cos(w*rads+pi*0.75), lambda w: sin(w*rads+pi*0.75), lambda w: 1), \
        (lambda w: cos(w*rads+pi*1.75), lambda w: sin(w*rads+pi*1.75), lambda w: 1), (lambda w: cos(w*rads+pi*1.25), lambda w: sin(w*rads+pi*1.25), lambda w: 1)]
        ws = (-5, 5)
        faces = [(1, 0, 2), (3, 1, 2), (5, 1, 3), (7, 5, 3), (5, 4, 0), (1, 5, 0), (4, 5, 7), (6, 4, 7),
                 (0, 4, 6), (2, 0, 6), (3, 2, 6), (7, 3, 6)]
        self.objects.append(Model4d(faces, ws, functions=functions1))

    def projection(self, cam: Camera) -> list:
        cam.update()
        projected = []
        for vert in self.vertices:
            v = (vert[0] - cam.x, vert[1] - cam.y, vert[2] - cam.z + 0.0001)
            factor = dot_product(normalise(v), cam.dir_x)
            factor_y = dot_product(normalise(v), cam.dir_y)
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
                 v[2] * cam.q - cam.q * cam.znear]
            if factor < -0.4 and abs(v[0]) < 2:
                if v[0] == 0:
                    signe_x = 1
                else:
                    signe_x = v[0] / abs(v[0])
                v = [v[0] - 4 * signe_x, v[1], v[2]]
            if factor_y < -0.5 and abs(v[1]) < 2:
                if v[1] == 0:
                    signe_y = 1
                else:
                    signe_y = v[1] / abs(v[1])
                v = [v[0], v[1] - 4 * signe_y, v[2]]
            projected.append(v)
        return projected

    def collisions(self, cam):
        for obj in self.objects:
            mins, maxs = obj.get_xbox(cam.w)
            x = max(mins[0], min(cam.x, maxs[0]))
            y = max(mins[1], min(cam.y, maxs[1]))
            z = max(mins[2], min(cam.z, maxs[2]))
            d = sqrt((x-cam.x)**2+(y-cam.y)**2+(z-cam.z)**2)
            if d < cam.radius:
                if y != cam.y:
                    vcx = x - cam.x
                    vcy = y - cam.y + int(y == maxs[1]) * cam.radius - int(y == mins[1]) * cam.radius
                    vcz = z - cam.z
                elif abs(x - cam.x) > abs(z - cam.z):
                    vcx = x - cam.x + int(x == maxs[0]) * cam.radius - int(x == mins[0]) * cam.radius
                    vcy = y - cam.y
                    vcz = z - cam.z
                else:
                    vcx = x - cam.x
                    vcy = y - cam.y
                    vcz = z - cam.z + int(z == maxs[2]) * cam.radius - int(z == mins[2]) * cam.radius
                cam.x += vcx
                cam.y += vcy
                cam.z += vcz

    @staticmethod
    def import_element(file, position=(0, 0, 0), color=(255, 255, 255)):
        f = open(file, "r")
        line = f.readline()
        vertices = []
        faces = []
        min_vert_x = 10000000000
        min_vert_y = 10000000000
        min_vert_z = 10000000000
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

    def splitting_triangle(self, points):
        res_faces = []
        t = []
        f = []
        p = []
        #axes = [[], [], []]
        axes = ((0, 0, 0, self.screen_size[1]),
                (self.screen_size[0], 0, self.screen_size[0], self.screen_size[1]),
                (0, 0, self.screen_size[0], 0),
                (0, self.screen_size[1], self.screen_size[0], self.screen_size[1]))
        if not points: return []
        for i in range(3):
            """
            if points[i][0] < 0:
                axes[i] += [(0, 0, 0, self.screen_size[1])]
            if points[i][0] > self.screen_size[0]:
                axes[i] += [(self.screen_size[0], 0, self.screen_size[0], self.screen_size[1])]
            if points[i][1] < 0:
                axes[i] += [(0, 0, self.screen_size[0], 0)]
            if points[i][1] > self.screen_size[1]:
                axes[i] += [(0, self.screen_size[1], self.screen_size[0], self.screen_size[1])]
                """
            if 0 <= points[i][0] <= self.screen_size[0] and 0 <= points[i][1] <= self.screen_size[1]:
                t.append(i)
                p.append(points[i])
            else:
                f.append(i)
        nb_in_screen = len(t)
        if nb_in_screen == 3: return (p,)
        for i in range(0, 2):
            for j in range(1+i, 3):
                for k in axes:
                    temp = inter_segment(points[i][0], points[i][1], points[j][0], points[j][1],
                                 k[0], k[1], k[2], k[3])
                    if bool(temp):
                        p.append(temp)
        if in_triangle(points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1], 0, 0):
            p.append((0, 0))
        if in_triangle(points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1], self.screen_size[0], 0):
            p.append((self.screen_size[0], 0))
        if in_triangle(points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1], 0, self.screen_size[1]):
            p.append((0, self.screen_size[1]))
        if in_triangle(points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1], self.screen_size[0], self.screen_size[1]):
            p.append((self.screen_size[0], self.screen_size[1]))
        if len(p) >= 3:
          return self.splitting_polygon(p)
        return []
        if nb_in_screen == 1:
            p = []
            common_axis = None
            for axe in axes[f[0]]:
                if axe in axes[f[1]]:
                    common_axis = axe
                    break
            if common_axis is not None:
                for i in range(3):
                    if i in t: p.append(points[i])
                    else: p.append(intersection(points[i][0], points[i][1], points[t[0]][0], points[t[0]][1],
                                                common_axis[0], common_axis[1], common_axis[2], common_axis[3]))
                p = (p,)
            else:
                if t[0] == 2:
                    temp = f[0]
                    f[0] = f[1]
                    f[1] = temp
                p = [points[t[0]],
                     intersection(points[f[0]][0], points[f[0]][1], points[t[0]][0], points[t[0]][1],
                                  axes[f[0]][0][0], axes[f[0]][0][1], axes[f[0]][0][2], axes[f[0]][0][3]),
                     intersection(axes[f[0]][0][0], axes[f[0]][0][1], axes[f[0]][0][2], axes[f[0]][0][3],
                                  axes[f[1]][0][0], axes[f[1]][0][1], axes[f[1]][0][2], axes[f[1]][0][3]),
                     intersection(points[f[1]][0], points[f[1]][1], points[t[0]][0], points[t[0]][1],
                                  axes[f[1]][0][0], axes[f[1]][0][1], axes[f[1]][0][2], axes[f[1]][0][3])]
                p = self.splitting_polygon(p)
            for elt in p:
                res_faces += self.splitting_triangle(elt)
        elif nb_in_screen == 2:
            t1 = t[0]
            t2 = t[1]
            if t2 == 3 and t1 == 1:
                t1 = 3
                t2 = 1
            f = f[0]
            inter1 = intersection(points[f][0], points[f][1], points[t1][0], points[t1][1],
                                  axes[f][0][0], axes[f][0][1], axes[f][0][2], axes[f][0][3])
            inter2 = intersection(points[f][0], points[f][1], points[t2][0], points[t2][1],
                                  axes[f][0][0], axes[f][0][1], axes[f][0][2], axes[f][0][3])
            p1 = [inter1, points[t1], points[t2]]
            p2 = [inter2, inter1, points[t2]]
            res_faces += self.splitting_triangle(p1)
            res_faces += self.splitting_triangle(p2)
        elif nb_in_screen == 0:
            return []
        else:
            res_faces.append(points)
        return res_faces

    def objects_elements(self, cam):
        self.vertices = []
        self.faces = []
        self.objects_v_pos = []
        for obj in self.objects:
            v, f = obj.get_elements(len(self.vertices), cam.w)
            self.vertices += v
            self.faces += f
            self.objects_v_pos.append(len(self.faces))

    def draw(self):
        self.objects_elements(camera)
        projected = self.projection(camera)
        self.screen.fill(black)
        faces = []
        sorted_faces = []
        distances = []
        actual_obj = 0
        for i in range(len(self.faces)):
            face = self.faces[i]
            normal = normalise(normal_p_3d(projected[face[1]], projected[face[0]], projected[face[2]]))
            if normal[2] < 0:
                points = []
                for o in range(len(face)):
                    points.append((int((projected[face[o]][0] + 1) * self.screen_size[0] / 2),
                                   int((projected[face[o]][1] + 1) * self.screen_size[1] / 2)))
                if points[0] == points[1] or points[1] == points[2] or points[2] == points[0] or \
                        abs(points[0][0]-points[1][0]) > 2*self.screen_size[0] or \
                        abs(points[1][0]-points[2][0]) > 2*self.screen_size[0] or \
                        abs(points[0][0]-points[2][0]) > 2*self.screen_size[0] or \
                        abs(points[0][1]-points[1][1]) > 2*self.screen_size[1] or \
                        abs(points[1][1]-points[2][1]) > 2*self.screen_size[1] or \
                        abs(points[0][1]-points[2][1]) > 2*self.screen_size[1]:
                    if i + 1 in self.objects_v_pos:
                        actual_obj += 1
                    continue
                normal2 = normalise(normal_p_3d(self.vertices[face[1]], self.vertices[face[0]], self.vertices[face[2]]))
                luminosity = (dot_product(normal2, self.light_dir) + 1) / 2
                d1 = (camera.x - self.vertices[face[0]][0]) ** 2 + (camera.y - self.vertices[face[0]][1]) ** 2 + \
                     (camera.z - self.vertices[face[0]][2]) ** 2
                d2 = (camera.x - self.vertices[face[1]][0]) ** 2 + (camera.y - self.vertices[face[1]][1]) ** 2 + \
                     (camera.z - self.vertices[face[1]][2]) ** 2
                d3 = (camera.x - self.vertices[face[2]][0]) ** 2 + (camera.y - self.vertices[face[2]][1]) ** 2 + \
                     (camera.z - self.vertices[face[2]][2]) ** 2
                faces.append((points, luminosity, self.objects[actual_obj]))
                distances.append(moy((d1, d2, d3)))
            if i + 1 in self.objects_v_pos:
                actual_obj += 1
        # sorted_faces = faces
        for _ in range(len(faces)):
            elt = max(distances)
            i = distances.index(elt)
            sorted_faces.append(faces[i])
            faces.pop(i)
            distances.pop(i)
        for face in sorted_faces:
            points, luminosity, obj = face
            for elt in self.splitting_triangle(points):
                color = obj.get_color()
                pg.draw.polygon(self.screen, (color[0]*luminosity, color[1]*luminosity, color[2]*luminosity), elt)
                pg.draw.line(self.screen, white, elt[0], elt[1])
                pg.draw.line(self.screen, white, elt[1], elt[2])
                pg.draw.line(self.screen, white, elt[0], elt[2])
        half_screen = self.screen_size[0]/2
        pg.draw.rect(self.screen, (0, 0, 0), pg.Rect(half_screen-402, 28, 804, 9))
        pg.draw.rect(self.screen, (0, 0, 0), pg.Rect(half_screen-403, 23, 6, 19))
        pg.draw.rect(self.screen, (0, 0, 0), pg.Rect(half_screen+398, 23, 6, 19))
        pg.draw.rect(self.screen, (255, 255, 255), pg.Rect(half_screen-400, 30, 800, 5))
        pg.draw.rect(self.screen, (255, 255, 255), pg.Rect(half_screen-401, 25, 2, 15))
        pg.draw.rect(self.screen, (255, 255, 255), pg.Rect(half_screen+400, 25, 2, 15))
        pg.draw.rect(self.screen, (255, 255, 0), pg.Rect(half_screen+camera.w*40-2, 18, 4, 29))
        pg.display.flip()


controller = Controller()
camera = Camera()
end = True
speed = 1
right = left = up = down = pu = pd = False
clicks = (False, False, False)
while end:
    for event in pg.event.get():
        clicks = pg.mouse.get_pressed()
        if event.type == pg.QUIT:
            end = False
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_RIGHT:
                right = True
            elif event.key == pg.K_LEFT:
                left = True
            elif event.key == pg.K_UP:
                up = True
            elif event.key == pg.K_DOWN:
                down = True
            elif event.key == pg.K_PAGEUP:
                pu = True
            elif event.key == pg.K_PAGEDOWN:
                pd = True
            elif event.key == pg.K_ESCAPE:
                pg.mouse.set_visible(True)
                pg.event.set_grab(False)
        elif event.type == pg.KEYUP:
            if event.key == pg.K_RIGHT:
                right = False
            elif event.key == pg.K_LEFT:
                left = False
            elif event.key == pg.K_UP:
                up = False
            elif event.key == pg.K_DOWN:
                down = False
            elif event.key == pg.K_SPACE:
                space = False
            elif event.key == pg.K_PAGEUP:
                pu = False
            elif event.key == pg.K_PAGEDOWN:
                pd = False
        elif event.type == pg.MOUSEMOTION:
            camera.angle_x -= event.rel[1] / 100
            camera.angle_y -= event.rel[0] / 100
    if clicks[0]:
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
        camera.y -= 0.01 * speed
    if pd:
        camera.y += 0.01 * speed
    controller.collisions(camera)
    controller.draw()
