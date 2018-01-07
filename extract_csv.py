import csv


f = open("summary.txt")
text = f.read()
lines = text.split("\n")
pairs = list(filter( lambda x: len(x) == 2, map( lambda x : x.split("="), lines)))
pairs.sort(key=lambda x: x[1])

out = open("summary.csv", "w")
w = csv.writer(out)
for row in pairs:
    cols = row[0].split("_")
    fields = cols[1:6] + [row[1]]
    w.writerow(fields)
out.close()