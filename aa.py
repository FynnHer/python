import random

class Wuerfel():
  def __init__(self):
    self.augen = 0
  def werfen(self):
    self.augen = random.randint(1,6)
  def getAugen(self):
    self.werfen()
    return self.augen
  
class Konto():
  def __init__(self, startbetrag):
    self.stand = startbetrag
  def einzahlen(self, betrag):
    self.stand += betrag
  def auszahlen(self, betrag):
    self.stand -= betrag
  def getStand(self):
    return self.stand
  
class Spielfeld():
  def __init__(self):
    self.gesetzteZahl = 0
  def setzen(self, zahl):
    self.gesetzteZahl = zahl
  def getGesetzeZahl(self):
    return self.gesetzteZahl
  
class Spieler():
  def __init__(self):
    pass

  def spielen(self, zahl):
    spielfeld.gesetzteZahl = zahl
  
class Spielanbieter():
  def __init__(self):
    pass
  def gewinnAuszahlen(self):
    richtig = False
    for spielfeld.getGesetzeZahl in [wuerfelA.getAugen(),wuerfelB.getAugen(), wuerfelC.getAugen()]:
      richtig = True
      konto.einzahlen(1)
    if richtig:
      konto.einzahlen(1)

wuerfelA = Wuerfel()
wuerfelB = Wuerfel()
wuerfelC = Wuerfel()
konto = Konto(10)
spielfeld = Spielfeld()
spieler = Spieler()
spielanbierter = Spielanbierter()
print('Wuerfel: ', wuerfelA.getAugen(), wuerfelB.getAugen(), wuerfelC.getAugen())
print('Konto  : ', konto.getStand())
print()

# Durchführung des Spiels
for i in range(10):
    spieler.spielen(2)
    spielanbieter.gewinnAuszahlen()
    # Ausgabe
    print('Wuerfel: ', wuerfelA.getAugen(), wuerfelB.getAugen(), wuerfelC.getAugen())
    print('Konto  : ', konto.getStand())
    print()
