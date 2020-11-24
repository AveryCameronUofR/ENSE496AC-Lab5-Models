import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if (nn.as_scalar(self.run(x)) >= 0.0):
            return 1
        return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 1
        while True:
            error = False
            for x, y in dataset.iterate_once(batch_size):
                y_pred = self.get_prediction(x)
                y = nn.as_scalar(y)
                if y != y_pred:
                    error = True
                    nn.Parameter.update(self.get_weights(),x,y)
            if error == False:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # vector of weights for layer 1 (1 of the dimensions will change depending on how many nodes)
        # biases layer 1
        # vector of weights for layer 2 (one will be 1, the other will be # of nodes)
        # bias layer 2
        #learning rate 
        """
        with 10 nodes, batch_size 5 and learning rate 0.005 infinity error
        with  25, 25, 25 -0.001 slowly created a couple straight lines and stagnated
        with 25,25,25 0.009 batchsize 5 started to create curved lines (passed but almost failed)
        with 100,100,100 0.009,5 0.019984 loss (worse than before)
        with 250 250 250 0.015 2 0.019662 Loss (best yet)
        """
        self.w0 = nn.Parameter(1, 250)
        self.b0 = nn.Parameter(1, 250)
        self.w1 = nn.Parameter(250,1)
        self.b1 = nn.Parameter(1,1)
        self.batch_size = 2
        self.learning_rate = -0.015

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        #l1 = nn.linear(x, self.w0)
        #l1b = nn.addbias(l1, ...)
        #l1r = nn.relu()
        "*** YOUR CODE HERE ***"
        l1 = nn.Linear(x, self.w0)
        l1b = nn.AddBias(l1, self.b0)
        r1 = nn.ReLU(l1b)
        l2 = nn.Linear(r1, self.w1)
        l2b = nn.AddBias(l2, self.b1)
        return l2b

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        #make your predictions using run
        #compute loss nn.squareloss
        y_pred = self.run(x)
        return nn.SquareLoss(y_pred,y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        #loop over dataset
            #compute loss
            #use loss to compute gradients/derivatives (nn.gradients)
            #update weights and biases based on gradients (self.w0.update)
            #check stopping condition
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                grad_wrt_w0, grad_wrt_b0, grad_wrt_w1, grad_wrt_b1 = nn.gradients(loss, [self.w0, self.b0, self.w1, self.b1])
                self.w0.update(grad_wrt_w0, self.learning_rate)
                self.b0.update(grad_wrt_b0, self.learning_rate)
                self.w1.update(grad_wrt_w1, self.learning_rate)
                self.b1.update(grad_wrt_b1, self.learning_rate)
            
            loss = self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))
            if (nn.as_scalar(loss) < 0.02):
                break

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        """
        10 batchsize 2, learning rate 0.015 slowly trained to 92% in 4 epochs
        50 batchsize 2, learning rate 0.025 trained to ~95-96% in 2 epochs
        250 batchsize 5, learning rate 0.045 trained to ~97.7% in 5 epochs 
        100 batchsize 3, learning rate 0.05 trained to 97%
        150 4 0.04 passed (5 minutes)
        10
        """
        self.w0 = nn.Parameter(784, 150)
        self.b0 = nn.Parameter(1, 150)
        self.w1 = nn.Parameter(150,10)
        self.b1 = nn.Parameter(1,10)
        self.batch_size = 4
        self.learning_rate = -0.03

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        l1 = nn.Linear(x, self.w0)
        l1b = nn.AddBias(l1, self.b0)
        r1 = nn.ReLU(l1b)
        l2 = nn.Linear(r1, self.w1)
        l2b = nn.AddBias(l2, self.b1)
        return l2b

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_pred = self.run(x)
        return nn.SoftmaxLoss(y_pred,y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        loop = True
        while loop:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                grad_wrt_w0, grad_wrt_b0, grad_wrt_w1, grad_wrt_b1 = nn.gradients(loss, [self.w0, self.b0, self.w1, self.b1])
                self.w0.update(grad_wrt_w0, self.learning_rate)
                self.b0.update(grad_wrt_b0, self.learning_rate)
                self.w1.update(grad_wrt_w1, self.learning_rate)
                self.b1.update(grad_wrt_b1, self.learning_rate)
            
            #loss = self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))
            if (dataset.get_validation_accuracy() >= 0.97):
                loop = False

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        """
        tried 50 50 50, batch 3, lr 0.05 obtained: 70% fail
        tried 200 200 200 batch 3 lr 0.05 obtained: inf fail
        tried 200 200 200 batch 3 lr 0.001 obtained ~80%
        tried 200 200 200 batch 3 lr 0.005 obtained ~82%
        tried 200 200 200 batch 5 lr 0.005 obtained 81
        tried 300 300 300 batch 5 lr 0.01 obtained 82
        tried 400 400 400 batch 5 lr 0.05 jumped around
        tried 400 400 400 batch 5 lr 0.03 obtained ~ 77
        tried 300 300 300 batch 2 lr 0.01 obtained 77 
        tried 200 200 200 batch 2 lr 0.0025 obtained ~81
        tried 200 200 200 batch 3 lr 0.0025 obtained ~81
        tried 50 50 50 batch 2 lr 0.001 obtained 82
        tried 75 75 75 batch 1 0.0015 obtained 80

        Tried 100, 100, 100 batch 1 lr 0.01 Passed 83 4 minutes
        Tried 64 64 64 batch 4 0.01 slow
        Tried 64 64 64 batch 2 lr 0.015 passed 84, 2 minutes
        """
        self.w_initial = nn.Parameter(self.num_chars, 64)
        self.w_hidden = nn.Parameter(64,64)
        self.w_final = nn.Parameter(64, len(self.languages))
        self.batch_size = 2
        self.learning_rate = -0.015

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        h = nn.Linear(xs[0],self.w_initial)
        z = nn.ReLU(h)
        for i in range(1,len(xs)):
            z = nn.Add(nn.Linear(xs[i], self.w_initial), nn.Linear(z, self.w_hidden))
            z = nn.ReLU(z)
        return nn.Linear(z, self.w_final)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_pred = self.run(xs)
        return nn.SoftmaxLoss(y_pred,y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        import datetime
        print(datetime.datetime.now())
        loop = True
        while loop:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                grad_wrt_wi, grad_wrt_wh, grad_wrt_wf = nn.gradients(loss, [self.w_initial, self.w_hidden, self.w_final])
                self.w_initial.update(grad_wrt_wi, self.learning_rate)
                self.w_hidden.update(grad_wrt_wh, self.learning_rate)
                self.w_final.update(grad_wrt_wf, self.learning_rate)
            if (dataset.get_validation_accuracy() > 0.85):
                loop = False
        print(datetime.datetime.now())
