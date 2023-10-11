text  = input("Text: ")
helper = 0
ergebnis = ""
ergebnis2 = ""

for char in text:
    ergebnis = char + ergebnis
    
while helper < len(text):
    ergebnis2 += text[-1*helper]