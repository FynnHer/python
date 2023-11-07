#Implementiere den Selectionsort-Algorithmus und recherchiere bei Bedarf
#eigenständig nach den benötigten Befehlen
liste = [45,3,24,234,35,234,2412,31,42]
lneu = []

def searchLow(list, lneu):
    helper = 
    counter = 0
    while counter < len(list):
        if list[counter] < helper:
            lneu.append(list[counter])
            list.pop(counter)
            break
        if not counter == len(list) - 1:
        	counter += 1
        else:
            
	