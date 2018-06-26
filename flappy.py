from itertools import cycle
import random
import sys
import Brasin
import pygame
from pygame.locals import *
import win32com.client as comclt
import Bird
import time
import datetime
import Logger
import copy
#Constants
FPS = 30
SCREENWIDTH  = 288
SCREENHEIGHT = 512

AIScore = 1

PopulationSize = 20
howManyWePick = 4 # how many we pick based on their fitness
currentPopulation=1
BirdsIteration = 1
maxScore = 0

birds = []
savedBirds = []
mutateRate = 1
# amount by which base can maximum shift to left
PIPEGAPSIZE  = 100 # gap between upper and lower part of pipe
BASEY        = SCREENHEIGHT * 0.79
# image, sound and hitmask  dicts
IMAGES, SOUNDS, HITMASKS = {}, {}, {}

# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # red bird
    (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    ),
    # blue bird
    (
        # amount by which base can maximum shift to left
        'assets/sprites/bluebird-upflap.png',
        'assets/sprites/bluebird-midflap.png',
        'assets/sprites/bluebird-downflap.png',
    ),
    # yellow bird
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
    'assets/sprites/background-day.png',
    'assets/sprites/background-night.png',
)

# list of pipes
PIPES_LIST = (
    'assets/sprites/pipe-green.png',
    'assets/sprites/pipe-red.png',
)


try:
    xrange
except NameError:
    xrange = range


def main():
    global SCREEN, FPSCLOCK,birds,PopulationSize,savedBirds
    birds = [Bird.Bird() for x in range(PopulationSize)]
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    pygame.display.set_caption('Flappy Bird')

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )
    # game over sprite
    IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
    # message sprite for welcome screen
    IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    # sounds
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'

    SOUNDS['die']    = pygame.mixer.Sound('assets/audio/die' + soundExt)
    SOUNDS['hit']    = pygame.mixer.Sound('assets/audio/hit' + soundExt)
    SOUNDS['point']  = pygame.mixer.Sound('assets/audio/point' + soundExt)
    SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
    SOUNDS['wing']   = pygame.mixer.Sound('assets/audio/wing' + soundExt)

    while True:
        
        # select random background sprites
        randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
        IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

        # select random player sprites
        for bird in birds:
            randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
            sprite=(
            pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
            )
            bird.setSprite(sprite)
        # select random pipe sprites
        pipeindex = random.randint(0, len(PIPES_LIST) - 1)
        IMAGES['pipe'] = (
            pygame.transform.rotate(
                pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), 180),
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
        )

        # hismask for pipes
        HITMASKS['pipe'] = (
            getHitmask(IMAGES['pipe'][0]),
            getHitmask(IMAGES['pipe'][1]),
        )

        HITMASKS['player'] = (
            getHitmask(birds[0].sprite[0]),
            getHitmask(birds[0].sprite[1]),
            getHitmask(birds[0].sprite[2]),
        )
        mainGame()
def crossOver(birdA,birdB):
    #get a cross over cutting point
    numberOfBiases = birdA.brain.n_hidden-1
    limit = random.randint(0,numberOfBiases)
 
	#swap 'bias' information between both parents from the hidden layer:
	# 1. left side to the crossover point is copied from one parent
	# 2. right side after the crossover point is copied from the second parent
    networkA = birdA.brain.getWeights()
    networkB = birdB.brain.getWeights()
    for i in range(limit,numberOfBiases):
        biasFrombirdA = networkA["bias1"][i]
        networkA["bias1"][i] = networkB["bias1"][i]
        networkB["bias1"][i] = biasFrombirdA;

    whichBirdShouldIchoose = random.randint(0,1)
    if whichBirdShouldIchoose == 1:
        return networkA
    else:
        return networkB
def resetGame():                       
    global BirdsIteration,PopulationSize,currentPopulation,maxScore,bestBrain,birdBrain,AIScore,howManyWePick,mutateRate,birds
    #pick the best birds
    currentPopulation=currentPopulation+1
    if(mutateRate == 1 and savedBirds[PopulationSize-1].fitness<110):
        #this is bad. None reached the first pipe. Instead of mutating and crossover we will recreate the population
        #We set again random weights
        for bird in savedBirds:
            print("reset")
            Logger.Logger.Log("reset")
            bird.resetBird();
        birds = savedBirds.copy()
       
    else:
        #the real mutate rate
        mutateRate = 0.1
        # the top 4 birds 
        Winners = []
        newlist = sorted(savedBirds, key=lambda x: x.fitness, reverse=True)
        for i in range(0,howManyWePick):
            #the birds are sorted based on their fitness. 
            #This means that savedBirds[PopulationSize-1] has the best fitness
            birds.insert(len(birds),newlist[i])
            Winners.insert(len(birds),newlist[i])
        for i in range(howManyWePick,howManyWePick+1):
            parentA = Winners[0]
            parentB = Winners[1]
            newWeights = crossOver(parentA,parentB)
            newlist[i].brain.updateWeightsJson(newWeights)
            birds.insert(len(birds),newlist[i])
        for i in range(howManyWePick+1,howManyWePick+4):
            #get 2 random parrents of the top 4
            parentA = random.choice(Winners)
            parentB = random.choice(Winners)
            newWeights = crossOver(parentA,parentB)
            newlist[i].brain.updateWeightsJson(newWeights)
            birds.insert(len(birds),newlist[i])
        for i in range(howManyWePick+4,PopulationSize):
            randomWinner = random.choice(Winners)
            newWeights = randomWinner.brain.getWeights()
            newlist[i].brain.updateWeightsJson(newWeights)
            birds.insert(len(birds),newlist[i])
        #save the best score of all time
        if maxScore < Winners[0].fitness:
            #save the best score
            maxScore = Winners[0].fitness
            print(maxScore)
        for i in range(0,PopulationSize):
            if i>=howManyWePick:
                newlist[i].brain.mutate(mutateRate)
            newlist[i].reesetFitness()
        Winners.clear()

    #for i in range (PopulationSize,PopulationSize-howManyWePick):
    BirdsIteration=0
    AIScore = 0
    BirdsIteration += 1
    savedBirds.clear()
    print("restart")
    Logger.Logger.Log("restart")
    mainGame()

def mainGame():
    global currentPopulation,AIScore,birds
    score = playerIndex = loopIter = 0
    #playerIndexGen = movementInfo['playerIndexGen']
    #playery = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)
    #playerx, playery = int(SCREENWIDTH * 0.2), movementInfo['playery']
    for bird in birds:
        #uncomment this to test that it works
        #random.uniform(0, 1)
        bird.PosX = int(SCREENWIDTH * 0.2)
        bird.PosY = int((SCREENHEIGHT - bird.sprite[0].get_height()) / 2)
    basex = 0
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # get 2 new pipes to add to upperPipes lowerPipes list
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    # list of upper pipes
    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
    ]

    # list of lowerpipe
    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
    ]

    pipeVelX = -4

    # player velocity, max velocity, downward accleration, accleration on flap

    playerMaxVelY =  10   # max vel along Y, max descend speed
    playerMinVelY =  -8   # min vel along Y, max ascend speed
    playerAccY    =   1   # players downward accleration
    playerRot     =  45   # player's rotation
    playerVelRot  =   3   # angular speed
    playerRotThr  =  20   # rotation threshold
    playerFlapAcc =  -9   # players speed on flapping


    oldtime =datetime.datetime.now()
    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                 for bird in birds:
                    jump(bird)
                    #SOUNDS['wing'].play()

        # check for crash here
        for bird in birds:
            crashTest = checkCrash(bird,
                               upperPipes, lowerPipes)
            if crashTest[0]:
                #here we actuualy substract from the fitness the distance to the next pipe.
                #This is in order to punish them.
                #Also if all birds fitness is below 0 than we will recreate the population
                #bird.fitness =bird.fitness + (bird.PosX - upperPipes[0]["x"])
                birds.remove(bird)
                savedBirds.insert(len(savedBirds),bird)
                print("death")
                Logger.Logger.Log("death")
                if len(birds) == 0:
                    resetGame()

        # check for score
        #playerMidPos = playerx + IMAGES['player'][0].get_width() / 2
        onePassed = False
        for pipe in upperPipes:
            pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
            for bird in birds:
                playerMidPos = bird.PosX + bird.sprite[0].get_width() / 2
                if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                    onePassed =True
                #SOUNDS['point'].play()
        if onePassed ==True :
            score+=1

        # playerIndex basex change
        if (loopIter + 1) % 3 == 0:
            playerIndex =(playerIndex+1)%3
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 100) % baseShift)

        # rotate the player
        if playerRot > -90:
            playerRot -= playerVelRot

        # player's movement
        for bird in birds:
            if bird.velocityY < playerMaxVelY and not bird.playerFlapped:
                bird.velocityY += playerAccY
            if bird.playerFlapped:
                bird.playerFlapped = False

            # more rotation to cover the threshold (calculated in visible rotation)
            playerRot = 45
            bird.PosY += min(bird.velocityY, BASEY - bird.PosY - bird.height)

        # move pipes to left
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        # print score so player overlaps the score
        showScore(score)
        showPopulation(currentPopulation)
        if len(birds) > 0 and birds[0].fitness>0:
            showNumber(birds[0].fitness,100, SCREENHEIGHT * 0.9)
        showNumber(maxScore,0, SCREENHEIGHT * 0.9)
        # Player rotation has a threshold
        visibleRot = playerRotThr
        if playerRot <= playerRotThr:
            visibleRot = playerRot
        for bird in birds:
            playerSurface = pygame.transform.rotate(bird.sprite[playerIndex], visibleRot)
            SCREEN.blit(playerSurface, (bird.PosX, bird.PosY))

        pygame.display.update()
        FPSCLOCK.tick(FPS)

        pipesX = upperPipes[0]["x"]
        upperPipeY = upperPipes[0]["y"] + IMAGES['pipe'][0].get_height()
        lowerPipeY =  upperPipeY + PIPEGAPSIZE
        #call the brain with location of bird and pipes 
        AIScore = AIScore+1
        start = time.time()
        #we should make this check only once per second
        if (datetime.datetime.now() - oldtime).total_seconds() >= 0:
            oldtime = datetime.datetime.now()
            for bird in birds:
                bird.fitness+=1
                Logger.Logger.Log("bird id " + str(bird.brain.id) + "fitness " + str(bird.fitness))
            for bird in birds:
                if(bird.PosX>pipesX+IMAGES['pipe'][0].get_width()):
                     upperPipeY = upperPipes[1]["y"] + IMAGES['pipe'][0].get_height()
                     lowerPipeY =  upperPipeY + PIPEGAPSIZE
                response = bird.brain.Think(bird.PosY,pipesX,upperPipeY,lowerPipeY)
                if(response > 0.5):
                    jump(bird)      



def showGameOverScreen(crashInfo):
    """crashes the player down ans shows gameover image"""
    score = crashInfo['score']
    playerx = SCREENWIDTH * 0.2
    playery = crashInfo['y']
    playerHeight = IMAGES['player'][0].get_height()
    playerVelY = crashInfo['playerVelY']
    playerAccY = 2
    playerRot = crashInfo['playerRot']
    playerVelRot = 7

    basex = crashInfo['basex']

    upperPipes, lowerPipes = crashInfo['upperPipes'], crashInfo['lowerPipes']

    # play hit and die sounds
    #SOUNDS['hit'].play()
    #if not crashInfo['groundCrash']:

        #SOUNDS['die'].play()

    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                if playery + playerHeight >= BASEY - 1:
                    return

        # player y shift
        if playery + playerHeight < BASEY - 1:
            playery += min(playerVelY, BASEY - playery - playerHeight)

        # player velocity change
        if playerVelY < 15:
            playerVelY += playerAccY

        # rotate only when it's a pipe crash
        if not crashInfo['groundCrash']:
            if playerRot > -90:
                playerRot -= playerVelRot

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        showScore(score)

        playerSurface = pygame.transform.rotate(IMAGES['player'][1], playerRot)
        SCREEN.blit(playerSurface, (playerx,playery))

        FPSCLOCK.tick(FPS)
        pygame.display.update()


def playerShm(playerShm):
    """oscillates the value of playerShm['val'] between 8 and -8"""
    if abs(playerShm['val']) == 8:
        playerShm['dir'] *= -1

    if playerShm['dir'] == 1:
         playerShm['val'] += 1
    else:
        playerShm['val'] -= 1


def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapY = 110 #random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE}, # lower pipe
    ]

def showNumber(text,xPos,yPos):
    scoreDigits = [int(x) for x in list(str(text))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width() 
    Xoffset = xPos

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, yPos))
        Xoffset += IMAGES['numbers'][digit].get_width()
def showScore(score):
    showNumber(score,(SCREENWIDTH) / 1.5, SCREENHEIGHT * 0.1)
def showPopulation(population):
    showNumber(population,0, SCREENHEIGHT * 0.1)

def checkCrash(bird, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = 0
    birdX = bird.PosX
    birdY = bird.PosY
    birdW = bird.sprite[0].get_width()
    birdH = bird.sprite[0].get_height()
    # if player hits the sky
    if birdY + birdH >= BASEY - 1:
        return [True, True]
    # if player crashes into ground
    if birdY + birdH <0:
        return [True, True]
    else:
        playerRect = pygame.Rect(birdX, birdY,
                      birdW, birdH)
        pipeW = IMAGES['pipe'][0].get_width()
        pipeH = IMAGES['pipe'][0].get_height()
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return [True, False]

    return [False, False]

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in xrange(rect.width):
        for y in xrange(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in xrange(image.get_width()):
        mask.append([])
        for y in xrange(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask
def jump(bird):
    bird.velocityY = -9
    bird.Flapped = True
if __name__ == '__main__':
    main()
