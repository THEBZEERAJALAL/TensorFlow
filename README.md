# TensorFlow
##R&amp;D on Activation function,Optimizer,Cost function :
### Activation Function:

  A function used to transform the activation level of a unit (neuron) into an output signal, that is known as activation function. There are different type of activation function using depending upon the requirement,
  
####1.SIGMOID = tf.nn.sigmoid(x, name=None)<br\>


   A sigmoid function is a mathematical function having an "S" shaped curve (sigmoid curve). Often, sigmoid function refers to the special case of the logistic function  defined by the formula,
              `y = 1 / (1 + exp(-x))`. Output ranges from[0,1]

#####Args: 
 
 *  <b>`x`</b>: A Tensor with type float32, float64, int32, complex64, int64, or qint32.`
 *  <b>`name`</b>: A name for the operation (optional).

####2.TANH = tf.nn.tanh(x, name=None)<br\>

The tanh(z) function is a rescaled version of the sigmoid, and its output range is [ − 1,1] instead of [0,1]. 
 `f(z) = exp(z) - exp(-z))/(exp(z) + exp(-z))`
 
####3.RELU = tf.nn.relu(features, name=None)<br\>
 
 A unit employing the rectifier is  called a rectified linear unit (ReLU) ` f(x)=\max(0,x)`.Computes rectified linear: `max(features, 0).
 
#####Args:

*  <b>`features`</b>: A Tensor. Must be one of the following types: float32, float64, int32, int64, uint8, int16, int8, uint16, half.
*  <b>`name`</b>: A name for the operation (optional)

####4.RELU6 = min(max(features, 0), 6)<br\>
#####Args:

*  <b>`features`</b>: A Tensor with type float, double, int32, int64, uint8, int16, or int8.
*  <b>`name`</b>: A name for the operation (optional).`

####5.SOFT_PLUS = tf.nn.softplus(features, name=None)

A smooth approximation to the rectifier is the analytic function,

`f(x)=ln(1+exp(x),which is called the softplus function. Computes softplus:- log(exp(features) + 1).

#####Args:
<b>`features`</b>: A Tensor. Must be one of the following types: float32, float64, int32, int64, uint8, int16, int8, uint16, half.
<b>`name`</b>: A name for the operation (optional).`

####6.SOFT_SIGN = tf.nn.softsign(features, name=None)

Computes softsign: features / (abs(features) + 1).

#####Args:

*  <b>`features`</b>: A Tensor. Must be one of the following types: float32, float64, int32, int64, uint8, int16, int8, uint16, half.  
*  <b>`name'</b>:A name for the operation (optional).`

### Cost Function:
the function to return a number representing how well the neural network performed to map training examples to correct output. There are different type of cost function using depending upon the requirement,

####1.Mean_Square_Error = tf.losses.mean_squared_error(labels, predictions, weights=1.0, scope=None, loss_collection=tf.GraphKeys.LOSSES)

Adds a Sum-of-Squares loss to the training procedure.weights acts as a coefficient for the loss. If a scalar is provided, then the loss is simply scaled by the given value. If weights is a tensor of size [batch_size], then the total loss for each sample of the batch is rescaled by the corresponding element in the weights vector. If the shape of weights matches the shape of predictions, then the loss of each measurable element of predictions is scaled by the corresponding value of weights.

##### Args:

*  <b>`labels`</b>: The ground truth output tensor, same dimensions as 'predictions'.
*  <b>`predictions`</b>: The predicted outputs.
*  <b>`weights`</b>: Optional Tensor whose rank is either 0, or the same rank as labels, and must be broadcastable to labels (i.e., all dimensions must be either 1, or the same as the corresponding losses dimension)
*  <b>`scope`</b>: The scope for the operations performed in computing the loss. loss_collection: collection to which the loss will be added.


####2.SOFTMAX_CROSS_ENTROPY = tf.nn.softmax_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, dim=-1, name=None)

Logits simply means that the function operates on the unscaled output of earlier layers and that the relative scale to understand the units is linear. It means, in particular, the sum of the inputs may not equal 1.`tf.nn.softmax` produces just the result of applying the softmax function to an input tensor. The softmax "squishes" the inputs so that `sum(input) = 1`.   It's a way of normalizing. The shape of output of a softmax is the same as the input. It just normalizes the values.
 `tf.nn.softmax_cross_entropy_with_logits` computes the cross entropy of the result after applying the softmax function (but it does it all together in a more mathematically careful way). It's similar to the result of:

`sm = tf.nn.softmax(x)
ce = cross_entropy(sm)`

Computes softmax cross entropy between logits and labels.

Measures the probability error in discrete classification tasks in which the classes are mutually exclusive (each entry is in exactly one class). For example, each CIFAR-10 image is labeled with one and only one label: an image can be a dog or a truck, but not both.

If using exclusive labels (wherein one and only one class is true at a time), see sparse_softmax_cross_entropy_with_logits.

logits and labels must have the same shape [batch_size, num_classes] and the same dtype (either float16, float32, or float64).

#####Args:

*  <b>`_sentinel`</b>: Used to prevent positional parameters. Internal, do not use.
*  <b>`labels`</b>: Each row labels[i] must be a valid probability distribution.
*  <b>`logits`</b>: Unscaled log probabilities.
*  <b>`dim`</b>: The class dimension. Defaulted to -1 which is the last dimension.
*  <b>`name`</b>: A name for the operation (optional).

### optimizer:

####1.ADAM(Adaptive Moment Estimation) = tf.train.AdamOptimizer 
It is a method that computes adaptive learning rates for each parameter.

`class tf.train.AdamOptimizer`

Methods

`__init__(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')`


#####Args:

*  <b>`learning_rate`</b>: A Tensor or a floating point value. The learning rate.
*  <b>`beta1`</b>: A float value or a constant float tensor. The exponential decay rate for the 1st moment estimates.
*  <b>`beta2`</b>: A float value or a constant float tensor. The exponential decay rate for the 2nd moment estimates.
*  <b>`ilon`</b>: A small constant for numerical stability.
*  <b>`use_locking`</b>: If True use locks for update operations.
*  <b>`name`</b>: Optional name for the operations created when applying gradients. Defaults to "Adam".

####2.GRADIENT_DESCENTOPTIMIZER = tf.train.GradientDescentOptimizer

`class tf.train.GradientDescentOptimizer`

Methods

`__init__(learning_rate, use_locking=False, name='GradientDescent')`

#####Args:

*  <b>`learning_rate`</b>: A Tensor or a floating point value. The learning rate to use.
*  <b>`use_locking`</b>: If True use locks for update operations.
*  <b>`name`</b>: Optional name prefix for the operations created when applying gradients. Defaults to "GradientDescent".
















`


               

    
 
         
        

