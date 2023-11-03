import random
rnumber = random.randint(1,128)
result = 1
guess = 0

while True:
    guess = int(input("Die geschätze Zahl: "))
    if guess == rnumber:
        print("Richtig mit ", result, "Versuchen")
        break
    elif guess < rnumber:
        print("Zahl ist größer")
        result += 1
    elif guees > rnumber:
        print("Zahl isr kleiner")
        result += 1
