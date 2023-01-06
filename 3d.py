from PIL import Image

'''lines = ["def a():\n", "    return ["]

img = Image.open("85red.jpg")
for i in range(85):
    for j in range(85):
        r, g, b = img.getpixel((i,j))
        if r>=180 and g<=50 and b<=50:
            lines.append(f"        ({round(i*0.1, 2)}, {round(j*0.1-8.5, 2)}, 1),\n")
img.close()
lines[-1] = lines[-1][:-2]+"]"

txt = open("cubes.py", "w")
txt.writelines(lines)
txt.close()'''

txt = open("cubes.py", "r")
lines = txt.readlines()
txt.close()
lines.pop(0)
lines[0] = '        ' + lines[0][12:]

img = Image.open("85px.jpg")
for index in range(len(lines)):
    elt = lines[index]
    if elt == "": continue
    elt = elt[9:21].split(", ")
    i, j = float(elt[0]), float(elt[1])
    x, y = (i*10,round(j+8.5, 1)*10)
    print(x, y)
    r, g, b = img.getpixel((x, y))
    '''if r <= 50 and g < 10 and b < 10:
        lines[index] = f"        ({i}, {j}, 1.1),\n"
        print('oui')
    else: '''
    lines[index] = f"        ({i}, {j}, 1.0, {r}, {g}, {b}),\n"
img.close()
lines[0] = '    return [' + lines[0][8:]
lines[-1] = lines[-1][:-1] + "]"
lines.insert(0, "def a():\n")
file = open("cubes2.py", "w", encoding="utf-8")
file.writelines(lines)
file.close()