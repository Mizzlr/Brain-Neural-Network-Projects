import cPickle, os, sys, timeit, numpy, theano, gzip
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

foods = ["apple","banana","biryani","bread","burger","cereal",
    "chiken","dosa","fries","idli","lemonrice","mango","milk",
    "omelette","orange","papaya","pineapple","pizza","pulao",
    "puri","rice","roti","samosa","water",]

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.input = input

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        self.input = input
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]


class ConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]
        self.input = input


class HumanVisualCortex(object):
    def __init__(self,nkerns,batch_size):
        rng = numpy.random.RandomState(23455)
        self.lzi = T.matrix('lzi')
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        self.nkerns = nkerns
        self.batch_size = batch_size

        self.layer0_input = self.x.reshape((self.batch_size, 3, 28, 28))
        self.layer0 = ConvPoolLayer(
            rng,
            input= self.layer0_input,
            image_shape=(self.batch_size, 3, 28, 28),
            filter_shape=(self.nkerns[0], 3, 5, 5),
            poolsize=(2, 2)
        )
        self.layer1 = ConvPoolLayer(
            rng,
            input= self.layer0.output,
            image_shape=(self.batch_size, self.nkerns[0], 12, 12),
            filter_shape=(self.nkerns[1], self.nkerns[0], 5, 5),
            poolsize=(2, 2)
        )
        self.layer2_input = self.layer1.output.flatten(2)
        self.layer2 = HiddenLayer(
            rng,
            input=self.layer2_input,
            n_in=self.nkerns[1] * 4 * 4,
            n_out=500,
            activation=T.tanh
        )
        self.layer3 = LogisticRegression(input= self.layer2.output, n_in=500, n_out=24)
        self.cost = self.layer3.negative_log_likelihood(self.y)

def load_data(dataset):
    print 'loading data...'
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        #print 'type of data_x: ', type(data_x), type(data_y)
        shared_x = theano.shared(numpy.asarray(data_x.tolist(),
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y.tolist(),
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def train(learning_rate=0.1, n_epochs=200,
                    dataset='food.pkl.gz', model_file = 'food_model.pkl',
                    nkerns=[20, 50], batch_size= 500):
    index = T.lscalar()
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    valid_set_x, valid_set_y = datasets[1]
    train_set_x, train_set_y = datasets[0]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size


    print 'building the model...'
    human_visual_cortex = HumanVisualCortex(nkerns, batch_size)
    test_model = theano.function(
        [index],
        human_visual_cortex.layer3.errors(human_visual_cortex.y),
        givens={
            human_visual_cortex.x: test_set_x[index * batch_size: (index + 1) * batch_size],
            human_visual_cortex.y: test_set_y[index * batch_size: (index + 1) * batch_size],
        }
    )
    validate_model = theano.function(
        [index],
        human_visual_cortex.layer3.errors(human_visual_cortex.y),
        givens={
            human_visual_cortex.x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            human_visual_cortex.y: valid_set_y[index * batch_size: (index + 1) * batch_size],
        }
    )
    params = human_visual_cortex.layer3.params + human_visual_cortex.layer2.params + \
        human_visual_cortex.layer1.params + human_visual_cortex.layer0.params
    grads = T.grad(human_visual_cortex.cost, params)
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]
    train_model = theano.function(
        [index],
        human_visual_cortex.cost,
        updates=updates,
        givens={
            human_visual_cortex.x: train_set_x[index * batch_size: (index + 1) * batch_size],
            human_visual_cortex.y: train_set_y[index * batch_size: (index + 1) * batch_size],
        }
    )
    #theano.printing.pydotprint(human_visual_cortex.cost,outfile = "hvc_cost.png", var_with_name_simple = True)
    #return

    print 'training ...'
    patience = 10050
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = 100 #min(n_train_batches, patience / 2)
    save_frequency = 10
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()
    epoch = 0
    done_looping = False
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index
            #if iter % 100 == 0:
            print 'training iteration: ', iter
            cost_ij = train_model(minibatch_index)
            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))
                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
            if (iter + 1) % save_frequency == 0:
                with open(model_file,'w') as f:
                    cPickle.dump(human_visual_cortex,f)
                    print "model saved"
                # human_visual_cortex = cPickle.load(open('model.pkl'))
                # print "model loaded"
                #predict()
            if patience <= iter:
                done_looping = True
                break
                
    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

def predict(dataset_file="food.pkl.gz",model_file="food_model.pkl"):
    global foods
    datasets = load_data(dataset_file)
    print "loading model ..."
    human_visual_cortex = cPickle.load(open(model_file))
    layer0_input = datasets[0][0][0:500]
    predict_model = theano.function(
        [],
        human_visual_cortex.layer3.y_pred,
        givens={
            human_visual_cortex.x: layer0_input
        }
    )
    accuracy = theano.function(
        [],
        human_visual_cortex.layer3.y,
    )
    print "predicting ..."
    predictions = predict_model()
    print "Test predictions: ", [foods[pred] for pred in predictions]
    print "percentage accuracy: ", numpy.mean( predictions == actual_values)



if __name__ == '__main__':

    helpmsg = """Usage: 
        $ python human_visual_cortex.py --train --dataset <<dataset.pkl.gz>>
                   --OR--
        $ python human_visual_cortex.py --predict --dataset <<dataset.pkl.gz>> --model <<model.pkl>>

        Use food.pkl.gz generated by running as dataset.pkl.gz 
        $ python create_pkl_gz_dataset.py --directory dataset

        and food_mini.pkl.gz generated by running as dataset.pkl.gz 
        $ python create_pkl_gz_dataset.py --directory dataset_mini

        the model.pkl file corresponding to food.pkl.gz is food_model.pkl
        and the model.pkl file corresponding to food_mini.pkl.gz is food_model_mini.pkl

        """

    if (len(sys.argv) == 1):
        print helpmsg
        exit()
    if (sys.argv[1] == "--train"):
        if sys.argv[3] == "food.pkl.gz":
            model_file = "food_model.pkl"
        elif sys.argv[3] == "food_mini.pkl.gz":
            model_file = "food_model_mini.pkl"
        else:
            print helpmsg
            exit()

        train(dataset=sys.argv[3], model_file=model_file)
    elif (sys.argv[1] == "--predict"):
        predict(dataset_file=sys.argv[3],model_file=sys.argv[5])
    else :
        print helpmsg
