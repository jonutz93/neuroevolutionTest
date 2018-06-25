import tensorflow as tf
import numpy
class Brain(object):
    """description of class"""
    def __init__(self):
        #do nothing here
        #hyperparameters
        self.n_input = 4
        self.n_hidden = 4
        self.n_output = 1
        self.learning_rate = 0.1
        self.epochs = 0
        
        #placeholders
        self.X = tf.placeholder(tf.float32)
        self.Y = tf.placeholder(tf.float32)
        #weights
        '''
        self.W1 = tf.Variable(tf.zeros([self.n_input,self.n_hidden]))
        self.W2 = tf.Variable(tf.zeros([self.n_hidden,self.n_output]))
        #bias
        self.b1 = tf.Variable(tf.zeros([1,self.n_hidden]), name="Bias1")
        self.b2 = tf.Variable(tf.zeros([1,self.n_output]), name="Bias2")
        '''
        self.Weights1 = numpy.random.random((self.n_input, self.n_hidden+1))
        self.Weights1 = self.Weights1.astype(numpy.float32)
        self.Weights2 = numpy.random.random((self.n_hidden, self.n_output+1))
        self.Weights2 = self.Weights2.astype(numpy.float32)

        self.W1 = tf.Variable(self.Weights1[:,0:self.n_input])
        self.W2 = tf.Variable(self.Weights2[:,0:self.n_output])
         #bias
        self.b1 = tf.Variable(self.Weights1[:,self.n_hidden], name="Bias1")
        self.b2 = tf.Variable(self.Weights2[:,self.n_output], name="Bias2")

        self.L2 = tf.sigmoid(tf.matmul(self.X, self.W1) + self.b1)
        self.hy = tf.sigmoid(tf.matmul(self.L2, self.W2) + self.b2)
        
        #not sure what this does yet
        self.cost = tf.reduce_mean(-self.Y*tf.log(self.hy) - (1-self.Y)*tf.log(1-self.hy))
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
        W1_assign = tf.assign(W1, self.W1)
        W2_assign = tf.assign(W2, self.W2)
        b1_assign = tf.assign(B1, self.b1)
        b2_assign = tf.assign(B2, self.b2)
        self.init = tf.global_variables_initializer()
    def initialize(self):
        #internal weights. Are generated random for now. I hjave to generated them here otherwise are generated at each iteration we have new numbers.
        #last colomn represents the bias
        self.Weights1 = numpy.random.random((self.n_input, self.n_hidden+1))
        self.Weights1 = self.Weights1.astype(numpy.float32)
        self.Weights2 = numpy.random.random((self.n_hidden, self.n_output+1))
        self.Weights2 = self.Weights2.astype(numpy.float32)
        with tf.Session() as session:
            session.run(self.init)
            session.run([W1_assign, W2_assign],feed_dict={W1_placeholder:w1_, W2_placeholder:w2_})

            print("plm")
    def updateWeights(matrix):
          with tf.Session() as session:
            session.run(self.init)
            npvar = session.run(self.W1)
            npvar = matrix
            sess.run(self.W1.assign(npvar))

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
        [0,0,-1,-1]])
        y_data = numpy.array([
        [1]])
        
   
        with tf.Session() as session:                  
            session.run(self.init)
            variables_names =[v.name for v in tf.trainable_variables()]
            #values = session.run(variables_names)
            #for k,v in zip(variables_names, values):
                #print(k, v)
            for step in range(self.epochs):
                session.run(optimizer, feed_dict={X: x_data, Y: y_data})
        
                if step % 1000 == 0:
                    print(session.run(cost, feed_dict={X: x_data, Y: y_data}))

            
            answer = tf.equal(tf.floor(self.hy + 0.5), self.Y)
            legitAnswer=session.run([self.hy], feed_dict={self.X: x_data, self.Y: y_data})
            #because reasons?
            answer = legitAnswer[0][0][0]
            #print(answer)
            accuracy = tf.reduce_mean(tf.cast(answer, "float"))
            #print(accuracy)
            accuracyPercent = accuracy.eval({self.X: x_data, self.Y: y_data}) * 100
            #print("Accuracy: ",accuracyPercent, "%")

            return answer
