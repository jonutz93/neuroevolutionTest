import Brasin as BirdBrain
import pygame
import random
id = 0
class Bird(object):
    def __init__(self):
        global id
        self.id = id
        self.brain = BirdBrain.Brain(self.id)
        id+=1
        self.brain.randomWeights()
        self.resetBird
        self.velocityY =-9
        self.playerFlapped = False
        self.PosY = 0
        self.PosX = 0
        self.height = 0
        self.width = 0
        #they all start above 0
        self.fitness = 100
        

    def resetBird(self):
        self.fitness = 100
        self.brain.randomWeights()
    def reesetFitness(self):
         self.fitness = 100
    def updateFitness(fitness):
        self.fitness = fitness
    def setSprite(self,sprite):
         self.sprite = sprite
         self.width = sprite[0].get_width()
         self.height = sprite[0].get_height()
    def getBrain(self):
        return self.brain


