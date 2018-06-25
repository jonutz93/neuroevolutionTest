import tensorflow as tf
import numpy
class Brain(object):
    """description of class"""
    def __init__(self):
        #parameters
        self.n_input = 4
        self.n_hidden = 4
        self.n_output = 1
        self.learning_rate = 0.1
        self.epochs = 0
        self.setup()
    def setup(self):
        #placeholders
        self.X = tf.placeholder(tf.float32)
        self.Y = tf.placeholder(tf.float32)
        #weights
        w1 = tf.Variable(tf.random_uniform([self.n_input, self.n_hidden], -1.0, 1.0))
        w2 = tf.Variable(tf.random_uniform([self.n_hidden, self.n_output], -1.0, 1.0))
        b1 = tf.Variable(tf.zeros([self.n_hidden]))
        b2 = tf.Variable(tf.zeros([self.n_output]))
             
        self.weight1 =numpy.random.random((self.n_input, self.n_hidden)).astype(numpy.float32)
        self.weight2 = numpy.random.random((self.n_hidden, self.n_output)).astype(numpy.float32)
        self.bias1 = numpy.random.random((self.n_hidden))
        self.bias2=numpy.random.random(self.n_output)
        print( self.weight1)
        print("lol")
        #Weights
        self.W1 = tf.Variable(self.weight1)
        self.W2 = tf.Variable(self.weight2)
        
        # Bias
        self.b1 = tf.Variable(tf.zeros([self.n_hidden]) , name="B1")
        self.b2 = tf.Variable(tf.zeros([self.n_output]) , name="B2")

        #Layers
        self.L2 = tf.sigmoid(tf.matmul(self.X, self.W1) + self.b1)
        self.hy = tf.sigmoid(tf.matmul(self.L2, self.W2) + self.b2)
        
        #not sure what this does yet
        self.cost = tf.reduce_mean(-self.Y*tf.log(self.hy) - (1-self.Y)*tf.log(1-self.hy))
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
        #weights
        self.W1_placeholder = tf.placeholder(tf.float32, shape=[self.n_input, self.n_hidden])
        self.W2_placeholder = tf.placeholder(tf.float32, shape=[self.n_hidden, self.n_output])
        self.W1_assign = tf.assign(self.W1, self.W1_placeholder)
        self.W2_assign = tf.assign(self.W2, self.W2_placeholder)
        #bias
        self.B1_placeholder = tf.placeholder(tf.float32, shape=[self.n_hidden])
        self.B2_placeholder = tf.placeholder(tf.float32, shape=[self.n_output])
        self.b1_assign = tf.assign(self.b1, self.B1_placeholder)
        self.b2_assign = tf.assign(self.b2, self.B2_placeholder)
        self.init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(self.init)
    def updateWeights(self,weights1,weights2,bias1,bias2):
        #internal weights. Are generated random for now. I hjave to generated them here otherwise are generated at each iteration we have new numbers.
        #last colomn represents the bias
        print("update weights")
        print(weights1,weights2,bias1,bias2)
        self.session.run([self.W1_assign, self.W2_assign],feed_dict={self.W1_placeholder:weights1, self.W2_placeholder:weights2})
        self.session.run([self.b1_assign, self.b2_assign],feed_dict={self.B1_placeholder:bias1, self.B2_placeholder:bias2})

    def randomWeights(self):
        #internal weights. Are generated random for now. I hjave to generated them here otherwise are generated at each iteration we have new numbers.
        #last colomn represents the bias
        self.weight1 =numpy.random.uniform(low=-1.,high=1.,size=(self.n_input, self.n_hidden)).astype(numpy.float32)
        self.weight2 = numpy.random.uniform(low=-1.,high=1.,size=(self.n_hidden, self.n_output)).astype(numpy.float32)
        self.bias1 = numpy.random.uniform(low=-1.,high=1.,size=(self.n_hidden))
        self.bias2 = numpy.random.uniform(low=-1.,high=1.,size=(self.n_output))
        self.updateWeights(self.weight1,self.weight2,self.bias1,self.bias2)
    def copyBrain(self):
        newBrain = Brain()
        newBrain.Weights1 = self.Weights1
        newBrain.Weights1 = self.Weights1
        newbrain.Weights2 = self.Weights2
    def mutate(self,x):
        if (random(1) < 0.1):
            offset = randomGaussian() * 0.5;
            newx = x + offset;
            return newx;
        else:
            return x
    def Think(self, yBirdPosition, pipesXPosition, upperPipeY, lowerPipeY):
        x_data = numpy.array([
        [yBirdPosition,pipesXPosition,upperPipeY,lowerPipeY]])
        y_data = numpy.array([
        [1]])
        
            
        #self.session.run(self.init)
        #answer = tf.equal(tf.floor(self.hy + 0.5), self.Y)
        legitAnswer=self.session.run([self.hy], feed_dict={self.X: x_data, self.Y: y_data})
        #because reasons?
        answer = legitAnswer[0][0][0]
        print(answer)
        #accuracy = tf.reduce_mean(tf.cast(answer, "float"))
        #print(accuracy)
        #accuracyPercent = accuracy.eval({self.X: x_data, self.Y: y_data}) * 100
        #print("Accuracy: ",accuracyPercent, "%")
        # print("W1 EVAL ")
        # print(self.W1.eval(),set_iterator=)
        return answer

    def mutate(self,chance):
        if numpy.random() > chance:
            print("plm")

        #var index = Math.floor(Math.random()*this.code.length);
        #var upOrDown = Math.random()
 