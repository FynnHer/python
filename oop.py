import random
import os
import time
import asyncio

class blackjack():
    def __init__(self, players = 2, sscore = 100, possiblecards = [2,3,4,5,6,7,8,9,10,10,10,10,11], bids = True):
        self.bids = bids
        self.startmoney = sscore
        self.possiblecards = possiblecards
        self.players = []
        self.players.append(["bank", 0, 0, True, False])
        for i in range(players):
            # Format: [name, money, currentscore, over21?, finished?, bid]
            self.players.append([input("YOUR NAME: "), self.startmoney, 0, True, False, 10])
        self.round()

    def roundstart(self):
        '''
        resetting all the necessary variables for the beginning of a new round
        '''
        self.bankcount = 0
        for i in range(1, len(self.players)):
            self.players[i][2] = 0
            self.players[i][3] = True
            self.players[i][4] = False
            if self.bids == True:
                self.players[i][5] = self.bidding(i)
        
    def drawcard(self, player):
        '''
        function for players to draw cards
        '''
        print(self.players[player][0] + ": DO YOU WANNA DRAW A CARD? y/n")
        print("CURRENT SCORE: " + str(self.players[player][2]))
        if input() == "y":
            card = self.possiblecards[random.randint(0,len(self.possiblecards)-1)] 
            print("YOU PULLED: " + str(card) + "     WOOOOOOOOW")
            self.players[player][2] += card
            if self.players[player][2] > 21:
                self.players[player][3] = False
                self.players[player][2] = 0
                return False
            print("NEW SCORE: " + str(self.players[player][2]))
            print("------------------------------------------")
            print("")
            return True
        else:
            self.players[player][4] = True
            return False

    def bidding(self, player):
        print("------------------------")
        print(self.players[player][0] + " YOUR BID???")
        return int(input())


    def newroundrequest(self):
        print("")
        print("DO YOU WANT TO PLAY ONE MORE ROUND? y/n")
        if input() == "y":
            print("------------- HAVE FUN WITH ONE MORE ROUND!!! --------------------------")
            print("")
            self.round()
        else:
            self.printresults(1)

    def printresults(self, printer):
        #player results
        print("-------------RESULTS: ------------")
        results = []
        for e in range(1, len(self.players)):
            print(self.players[e][0].upper() + " SCORE: " + str(self.players[e][printer]))
        print("-----------------------------------")
        print("")

    def payout(self):
        #checking if players won
        print("------------- $$$ MONEY $$$ ----------")
        for i in range(1, len(self.players)):
            if self.players[i][2] > self.players[0][2]:
                self.players[i][1] += self.players[i][5]
            elif self.players[i][2] == self.players[0][2]:
                pass
            else:
                self.players[i][1] -= self.players[i][5]
            print("YOUR MONEY $$$:   " + self.players[i][0].upper() + " " + str(self.players[i][1]))

    def round(self):
        '''
        one complete round of blackjack including drawing cards and receiving money
        '''
        self.roundstart()


        #drawing cards
        roundgoing = True
        while roundgoing == True:
            for e in range(1, len(self.players)):
                if self.drawcard(e) == False:
                    roundgoing = False

        self.printresults(2)

        #calculating bank result and printing it
        print("-------------- === BANK === -------------")
        while self.players[0][2] <= 16:
            self.players[0][2] += self.possiblecards[random.randint(0,len(self.possiblecards)-1)]
        if self.players[0][2] > 21:
            self.players[0][2] = 0
        print("SCORE: " + str(self.players[0][2]))

        self.payout()
        
        self.newroundrequest()


class drawer():
    def __init__(self):
        self.clear = lambda: os.system('cls')
        self.clear()
    def rectangle(self):
        counter = 0
        counter2 = 0
        for i in range(1000):
            counter += 1
            if counter == 20:
                counter2 += 1
                counter = 0
            for e in range(10):
                print(counter2 * 'x')
            time.sleep(0.01)
            self.clear()

    def create_matrix(self, h, w):
        self.matrix = []
        for i in range(h):
            self.matrix.append([])
            for e in range(w):
                self.matrix[i].append(0)
        print(self.matrix)
    
    def set_coords(self, coords, item):
        self.matrix[coords[1]] = self.matrix[coords[1]][:coords[0]] + [item] + self.matrix[coords[1]][coords[0]+1:]

    async def renderer(self, coords):
        print(self.matrix)
        for e in range(100):
            for i in self.matrix:
                printing = ''
                for e in i:
                    if e == 1:
                        printing += 'x'
                    elif e == 0:
                        printing += 'x'
                    elif e == 2:
                        printing += 'o'
                print(printing)
                time.wait(0.1)
                self.clear()

drawing = drawer()
drawing.create_matrix(5,5)
for i in range(3):
    drawing.set_coords([i,2], 2)
drawing.renderer([2,2])

#blackjackgame = blackjack()
