#Implementiere den Selectionsort-Algorithmus und recherchiere bei Bedarf
#eigenständig nach den benötigten Befehlen
liste = [45,3,24,234,35,234,2412,31,42]
lneu = []

def searchLow(list, lneu):
 while len(list) > 0:
    lneu.append(min(list))
    list.pop(min(list))
            
	