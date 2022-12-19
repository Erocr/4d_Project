file = open("Rat.obj", "r")
rat = file.readlines()
file.close()

for i in range(len(rat)):
    elt = rat[i]
    if elt != "" and elt[0] == "v":
        eltl = elt.split(" ")
        for j in range(1, 4):
            eltl[j] = float(eltl[j])#/30
        rat[i] = f"v {eltl[1]+0.3} {eltl[2]} {eltl[3]}\n"

newfile = open("Rat.obj", "w", encoding="utf-8")
newfile.writelines(rat)
newfile.close()