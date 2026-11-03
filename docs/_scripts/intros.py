"""One-paragraph introductions for every demo: what the demo is for and what
it is meant to show, as opposed to how to operate it (that text comes from the
demo window itself).

Keys are ``book{N}/ch{NN}/{slug}`` as printed by ``python catalog.py``.
``gen_tutorials.py --check`` fails when a demo has no entry here.
"""

INTROS = {
    # ------------------------------------------------------------------ Book 1
    # Chapter 2: Neuron Model & Network Architecture
    "book1/ch02/one_input_neuron": """
        The simplest building block of every network in the book: one neuron
        with one scalar input. It shows how the weight scales the input, the
        bias shifts it, and the transfer function shapes the result into the
        output a = f(wp + b).""",
    "book1/ch02/two_input_neuron": """
        Extends the single neuron to two inputs, so you can see how the net
        input n = w1 p1 + w2 p2 + b combines several inputs before the transfer
        function. This is the step toward layers described by weight matrices.""",

    # Chapter 3: An Illustrative Example
    "book1/ch03/perceptron_classification": """
        The apple-versus-orange sorting problem of Chapter 3 solved with a
        single-neuron perceptron. The hard-limit neuron separates the two fruit
        classes with a plane in the three-dimensional space of shape, texture
        and weight measurements.""",
    "book1/ch03/hamming_classification": """
        The same fruit problem solved by a Hamming network: a feedforward layer
        measures how close the input is to each stored prototype, then a
        recurrent competitive layer lets the closest prototype win.""",
    "book1/ch03/hopfield_classification": """
        The same problem solved with a Hopfield network, a recurrent network
        whose state is iterated until it settles on the stored prototype that
        is nearest to the input.""",

    # Chapter 4: Perceptron Learning Rule
    "book1/ch04/decision_boundaries": """
        Shows the geometry behind a perceptron: the decision boundary is a line
        perpendicular to the weight vector, offset by the bias. Moving the
        boundary by hand shows which weight and bias values produce it.""",
    "book1/ch04/perceptron_rule": """
        Demonstrates the perceptron learning rule, which adds or subtracts
        input vectors to the weights until every training point falls on the
        correct side of the boundary. It converges whenever the classes are
        linearly separable.""",

    # Chapter 5: Signal & Weight Vector Spaces
    "book1/ch05/gram_schmidt": """
        Illustrates Gram-Schmidt orthogonalization, which turns a set of
        independent vectors into an orthogonal set. Orthogonal bases make the
        vector-space calculations used later in the book much simpler.""",
    "book1/ch05/reciprocal_basis": """
        Shows how to expand a vector in terms of a basis that is not
        orthogonal: the reciprocal basis vectors give the expansion
        coefficients directly.""",

    # Chapter 6: Linear Transformations for Neural Networks
    "book1/ch06/linear_transformations": """
        Lets you define a linear transformation by choosing where two vectors
        are sent, then displays its eigenvectors: the directions that are only
        stretched or flipped, never rotated.""",
    "book1/ch06/eigenvector_game": """
        Builds intuition for eigenvectors by turning them into a search: a
        vector is an eigenvector of the hidden transformation when its image
        points in the same or the opposite direction.""",

    # Chapter 7: Supervised Hebbian Learning
    "book1/ch07/supervised_hebb": """
        Demonstrates supervised Hebbian learning on a pattern-recognition task.
        The Hebb rule or the pseudoinverse rule builds the weight matrix from
        the target patterns, and the resulting network reconstructs a corrupted
        test pattern.""",

    # Chapter 8: Performance Surfaces & Optimum Points
    "book1/ch08/taylor_series_1": """
        Shows how a Taylor series approximates a function near a point, using
        the zeroth-, first- and second-order approximations of the cosine. These
        local approximations are the tool for analysing performance surfaces.""",
    "book1/ch08/taylor_series_2": """
        Extends the Taylor series to a function of two variables, showing the
        first- and second-order approximations as 3D surfaces and contour
        plots around the chosen point.""",
    "book1/ch08/directional_derivatives": """
        Illustrates the directional derivative, the slope of a performance
        surface in a chosen direction, which is computed from the gradient and
        the direction vector.""",
    "book1/ch08/quadratic_function": """
        Shows the quadratic function F(x) = ½ xᵀAx + dᵀx + c. Its shape, a bowl,
        a saddle or a valley, is decided by the eigenvalues and eigenvectors of
        the Hessian matrix A.""",

    # Chapter 9: Performance Optimization
    "book1/ch09/steepest_descent_for_quadratic": """
        Steepest descent on a quadratic function. Shows how the learning rate
        decides whether the trajectory converges smoothly, oscillates or
        diverges, with the stability limit set by the largest eigenvalue of the
        Hessian.""",
    "book1/ch09/method_comparison": """
        Compares steepest descent with the conjugate gradient method on the
        same quadratic function, both using a line search, to show how
        conjugate gradient reaches the minimum in far fewer steps.""",
    "book1/ch09/newton_s_method": """
        Illustrates Newton's method, which minimizes the second-order Taylor
        approximation at each step. It finds the minimum of a quadratic in one
        step but can be misled on functions that are not quadratic.""",
    "book1/ch09/steepest_descent": """
        Steepest descent on a performance surface that is not quadratic. The
        trajectory depends on the starting point and the learning rate and can
        be trapped in a local minimum.""",

    # Chapter 10: Widrow-Hoff Learning
    "book1/ch10/adaptive_noise_cancellation": """
        The classic ADALINE application: an adaptive filter trained with the
        LMS algorithm learns to predict the noise contaminating a signal, so the
        noise can be subtracted and the original signal recovered.""",
    "book1/ch10/eeg_noise_cancellation": """
        Applies LMS noise cancellation to an electroencephalogram (EEG)
        contaminated by power-line interference, a realistic use of the
        adaptive linear network.""",
    "book1/ch10/linear_classification": """
        Shows an ADALINE trained with the LMS algorithm as a pattern
        classifier, and how its output meter responds as the input pattern is
        edited one pixel at a time.""",

    # Chapter 11: Backpropagation
    "book1/ch11/network_function": """
        Shows the function computed by a 1-2-1 network, two log-sigmoid hidden
        neurons feeding a linear output, and how each weight and bias reshapes
        the response. This is the foundation of function approximation with
        multilayer networks.""",
    "book1/ch11/function_approximation": """
        Trains a two-layer network with backpropagation to fit a target
        function. Shows how the number of hidden neurons limits how complex a
        function the network can approximate.""",
    "book1/ch11/generalization": """
        Trains the same network on noisy data points to show how network size
        affects generalization: too few neurons underfit, too many fit the
        noise.""",

    # Chapter 12: Variations on Backpropagation
    "book1/ch12/steepest_descent_1": """
        Basic backpropagation seen as steepest descent on the error surface of
        a 1-2-1 network, with the surface shown in 3D. Flat regions and local
        minima explain why plain backpropagation can be slow.""",
    "book1/ch12/steepest_descent_2": """
        The same algorithm on the contour plot with an adjustable learning
        rate, showing oscillation and divergence when the rate is too large.""",
    "book1/ch12/momentum": """
        Adds momentum to backpropagation. The low-pass filtered update smooths
        the trajectory and allows a larger learning rate without
        oscillation.""",
    "book1/ch12/variable_learning_rate": """
        Demonstrates variable learning rate backpropagation, which raises the
        learning rate while the error keeps falling and lowers it when the error
        jumps.""",
    "book1/ch12/cg_line_search": """
        Shows the line search used by the conjugate gradient algorithm:
        locating the minimum of the error along a single search direction with
        interval location and interval reduction.""",
    "book1/ch12/conjugate_gradient": """
        Conjugate gradient backpropagation chooses successive search directions
        that are conjugate with respect to the Hessian, so it converges in far
        fewer iterations than steepest descent.""",
    "book1/ch12/marquardt_step": """
        Illustrates a single Levenberg-Marquardt step: as the parameter mu
        changes, the step interpolates between the Gauss-Newton step and a small
        step in the gradient direction.""",
    "book1/ch12/marquardt": """
        Trains with the Levenberg-Marquardt algorithm, the fastest method in the
        chapter for small networks, and shows its trajectory on the contour
        plot.""",

    # Chapter 13: Generalization
    "book1/ch13/early_stopping": """
        Demonstrates early stopping: the training error keeps falling while the
        validation error eventually rises, so training should stop at the
        minimum of the validation error to avoid overfitting.""",
    "book1/ch13/regularization": """
        Shows regularization, which adds a penalty on the size of the weights
        to the performance index. The regularization ratio sets the trade-off
        between fitting the data and keeping the network smooth.""",
    "book1/ch13/bayesian_regularization": """
        Demonstrates Bayesian regularization, which chooses the regularization
        parameters automatically from the data and reports the effective number
        of parameters the network is using.""",
    "book1/ch13/early_stopping_regularization": """
        Shows the link between early stopping and regularization: on a
        quadratic error surface the early-stopping trajectory passes through the
        same points as the minima of the regularized performance index for
        increasing regularization ratios.""",

    # Chapter 14: Dynamic Networks
    "book1/ch14/fir_network": """
        A finite impulse response (FIR) network: a layer fed by a tapped delay
        line of the input. Shows how the weights shape its response to
        sinusoidal, square and other input sequences.""",
    "book1/ch14/iir_network": """
        An infinite impulse response (IIR) network, which feeds its own delayed
        output back to the input, so the response depends on previous outputs
        as well as previous inputs.""",
    "book1/ch14/dynamic_derivatives": """
        Compares static and dynamic derivatives in a recurrent network. The
        dynamic derivative includes the effect of a weight change propagating
        around the feedback loop over time, which is what dynamic
        backpropagation must compute.""",
    "book1/ch14/recurrent_network_training": """
        Shows the training of a recurrent network with dynamic
        backpropagation, and the spurious valleys in the error surface that
        make recurrent networks hard to train.""",

    # Chapter 15: Associative Learning
    "book1/ch15/unsupervised_hebb": """
        Unsupervised Hebbian learning on the banana-recognition network: a
        weight grows when its input and the neuron's output are active
        together, so the network learns to respond to a stimulus that keeps
        accompanying one it already recognizes.""",
    "book1/ch15/effects_of_decay_rate": """
        Shows how a decay term in the Hebb rule keeps the weights bounded. The
        learning rate and the decay rate together set the maximum weight and
        how quickly the network forgets.""",
    "book1/ch15/hebb_with_decay": """
        The banana network again, now with weight decay: the weight stops
        growing at a limit and shrinks when the association is not
        reinforced.""",
    "book1/ch15/graphical_instar": """
        A geometric view of the instar rule: when the neuron is active, the
        weight vector moves toward the input vector by a fraction set by the
        learning rate.""",
    "book1/ch15/outstar": """
        Demonstrates the outstar rule, which learns to recall a pattern. After
        the network has seen several pineapples with both scanners on, it
        reproduces their measurements from the identifying signal alone.""",

    # Chapter 16: Competitive Networks
    "book1/ch16/competitive_classification": """
        The fruit-sorting problem solved with a competitive layer: the neurons
        compete and the one whose weight vector is closest to the input
        wins.""",
    "book1/ch16/competitive_learning": """
        Competitive learning with the Kohonen rule: the winning neuron's weight
        vector moves toward each input, so the prototypes end up clustering the
        data.""",
    "book1/ch16/1_d_feature_map": """
        A one-dimensional self-organizing feature map. Neighbouring neurons
        learn together, so their weight vectors arrange themselves along a
        chain that follows the distribution of the inputs.""",
    "book1/ch16/2_d_feature_map": """
        A two-dimensional self-organizing map that spreads a grid of neurons
        over the input distribution, preserving the topology of the input
        space.""",
    "book1/ch16/lvq_1": """
        Learning vector quantization (LVQ1), a competitive network trained with
        supervision: the winning prototype moves toward the input if it has the
        right class and away from it if it does not.""",
    "book1/ch16/lvq_2": """
        LVQ2 refines LVQ1 by also adjusting the runner-up prototype when the
        winner is wrong, which corrects misclassifications faster.""",

    # Chapter 17: Radial Basis Networks
    "book1/ch17/network_function_radial": """
        Shows the function computed by a radial basis network with two Gaussian
        hidden neurons, and how the centers, widths and output weights shape
        it. Compare it with the sigmoid network of Chapter 11.""",
    "book1/ch17/pattern_classification": """
        Shows a radial basis network as a two-dimensional classifier. Each
        hidden neuron carves out a region around its center, visible in the 3D
        response surface.""",
    "book1/ch17/linear_least_squares": """
        Trains only the output layer of a radial basis network with linear
        least squares, after fixing the centers and spreads of the basis
        functions.""",
    "book1/ch17/orthogonal_least_squares": """
        Shows orthogonal least squares, which picks the most useful basis
        centers from the training points one at a time, growing the hidden
        layer until the error is small enough.""",
    "book1/ch17/non_linear_optimization": """
        Trains all parameters of a radial basis network, including centers and
        widths, with gradient-based nonlinear optimization, as is done for
        multilayer networks.""",

    # Chapter 18: Grossberg Network
    "book1/ch18/leaky_integrator": """
        The leaky integrator, the basic building block of Grossberg's networks:
        its output follows the input with a time constant, modelling the
        continuous-time response of a neuron.""",
    "book1/ch18/shunting_network": """
        The shunting model, in which excitatory and inhibitory inputs multiply
        the state, so the response stays bounded no matter how large the inputs
        become.""",
    "book1/ch18/grossberg_layer_1": """
        Layer 1 of the Grossberg network. Its on-center/off-surround
        connections normalize the input pattern, so the layer responds to
        relative rather than absolute intensities.""",
    "book1/ch18/grossberg_layer_2": """
        Layer 2 of the Grossberg network, a competitive layer with lateral
        feedback that enhances contrast and settles into a winner-take-all
        response.""",
    "book1/ch18/adaptive_weights": """
        Shows instar learning on the adaptive weights between the two layers:
        the weights learn the normalized input prototypes presented to the
        network.""",

    # Chapter 19: Adaptive Resonance Theory
    "book1/ch19/art1_layer_1": """
        Layer 1 of ART1, which compares the input pattern with the expectation
        sent down from Layer 2 and passes on only where the two agree.""",
    "book1/ch19/art1_layer_2": """
        Layer 2 of ART1, the competitive layer that selects the best-matching
        prototype and sends the corresponding expectation back to Layer 1.""",
    "book1/ch19/orienting_subsystem": """
        The orienting subsystem, which measures the degree of match between
        input and expectation and resets Layer 2 when the match falls below the
        vigilance level.""",
    "book1/ch19/art1_algorithm": """
        Runs the complete ART1 algorithm on binary patterns, showing how
        prototypes are learned and how the vigilance parameter controls how
        many categories are formed.""",

    # Chapter 20: Stability
    "book1/ch20/dynamical_system": """
        A pendulum as a nonlinear dynamical system. Its trajectories in the
        phase plane illustrate equilibrium points, stability and Lyapunov's
        method, the background needed for recurrent networks.""",

    # Chapter 21: Hopfield Network
    "book1/ch21/hopfield_network": """
        The continuous Hopfield network as a dynamical system: the state moves
        downhill on the Lyapunov (energy) function toward stable points that
        act as stored memories.""",

    # ------------------------------------------------------------------ Book 2
    # Chapter 2: Multilayer Networks
    "book2/ch02/poslin_network_function": """
        A multilayer network built from poslin (ReLU) neurons. Each hidden
        neuron contributes one piecewise-linear bend, and the demo shows how the
        weights and biases place those bends to build up the network
        function.""",
    "book2/ch02/poslin_decision_regions": """
        Shows how a poslin network with two inputs divides the input plane into
        regions, each hidden neuron adding one linear fold to the response.""",
    "book2/ch02/poslin_decision_regions_2d": """
        The same decision regions viewed from above as a contour map, which
        makes the straight boundaries between regions easy to see.""",
    "book2/ch02/poslin_decision_regions_3d": """
        The same network drawn as a 3D response surface, showing the planar
        facets that the poslin neurons create.""",
    "book2/ch02/cascaded_function": """
        Shows what happens when layers are cascaded: the output of one hidden
        function becomes the input of the next, so a few layers of a simple
        function compose into a much more complex one.""",

    # Chapter 3: Multilayer Network Training
    "book2/ch03/gradient_descent": """
        Gradient descent on an error surface, drawn on the surface itself.
        Revisits steepest descent from Chapter 9 of the first book as the
        starting point for training deep networks.""",
    "book2/ch03/stochastic_gradient_descent": """
        Stochastic gradient descent estimates the gradient from a minibatch, so
        each step is cheaper but noisier. The trajectory shows the random walk
        around the path of full gradient descent.""",
    "book2/ch03/gradient_descent_vs_adam": """
        Compares plain gradient descent with Adam, which adapts the step size
        of every parameter from running estimates of the gradient's mean and
        variance, on the same training task.""",

    # Chapter 4: Supplemental Training Methods
    "book2/ch04/scaling": """
        Shows how input scaling and batch normalization change the
        distributions of inputs, net inputs and outputs inside a deep network,
        and why keeping those distributions well behaved makes training
        easier.""",
    "book2/ch04/initialization_effect": """
        Shows how the weight initialization scheme affects the activations as
        the network gets deeper, with distributions that vanish or saturate
        when the initialization is poorly matched to the transfer function.""",
    "book2/ch04/dropout": """
        Dropout as a regularizer: randomly disabling hidden neurons during
        training on noisy data, compared with the early-stopping and
        regularization demos of the first book.""",

    # Chapter 5: Python Introduction
    "book2/ch05/cheatsheet": """
        A set of quick-reference sheets for the Python features used in the
        book: data structures, functions, classes, NumPy, Matplotlib and
        more.""",
    "book2/ch05/learning_scheduler": """
        A learning rate schedule with linear warm-up, a plateau and geometric
        decay, applied to gradient descent on a linear regression problem, to
        show how the schedule changes convergence.""",

    # Chapter 6: TensorFlow Introduction
    "book2/ch06/cheatsheet": """
        A quick-reference sheet for TensorFlow 2 and Keras.""",
    "book2/ch06/tensor_shapes": """
        Explains tensor shapes for the data types used in the book, vectors,
        time series, images and video, and what each axis means when the tensor
        is handed to a model.""",
    "book2/ch06/model_builder": """
        Builds a network interactively and shows its diagram, parameter count,
        model summary and the equivalent Keras and PyTorch code, connecting the
        book's notation to the frameworks.""",
    "book2/ch06/xor_training": """
        Trains a small network on the XOR problem, the classic example a single
        layer cannot solve, while showing the loss curve and the decision
        boundary evolve for different optimizers and settings.""",
    "book2/ch06/dataset_pipeline": """
        Shows how a tf.data pipeline turns a CSV file into shuffled minibatches,
        and exactly which rows the network receives in each step.""",

    # Chapter 7: General Network Notation
    "book2/ch07/lffn": """
        Forward propagation through a layered feed-forward network written in
        the book's general notation, where every layer has its own weight
        function, net input function and transfer function.""",
    "book2/ch07/glddn": """
        Forward propagation through a general layered delayed dynamic network:
        two layers with tapped delays and feedback, showing how signals move
        through the network over time.""",

    # Chapter 8: Convolution Networks
    "book2/ch08/convolution_networks_demo": """
        Shows how a convolution layer slides a kernel over its input, and how
        input size, kernel size, stride and padding determine the size of the
        output feature map.""",

    # Chapter 9: Post-Training Analysis
    "book2/ch09/linearized_network_response": """
        Post-training analysis of a trained network: linearizing the response
        around a chosen input point shows the local gradient, or sensitivity,
        and the tangent-plane approximation at that point.""",

    # Chapter 10: PyTorch Introduction
    "book2/ch10/cheatsheet": """
        A set of quick-reference sheets for PyTorch.""",
    "book2/ch10/training_loop": """
        Steps through a PyTorch training loop one line at a time, showing which
        numbers each line changes and how the gradient moves a single
        neuron.""",
    "book2/ch10/dataset_dataloader": """
        Shows how a DataLoader feeds a Dataset to training: the effect of batch
        size, shuffling and dropping the last partial batch, one epoch at a
        time.""",

    # Chapter 11: Linear Sequence Processing
    "book2/ch11/sequence_averaging_network": """
        A linear network with a tapped delay line that averages consecutive
        inputs, the simplest example of processing a sequence one time step at
        a time.""",
    "book2/ch11/sequence_smoothing_response": """
        A linear network with feedback that smooths an input sequence. The
        layer weight sets how much of the past the network remembers.""",
    "book2/ch11/impulse_response": """
        Shows how the pole locations of a linear recurrent network determine
        its impulse response: decaying, oscillating or unstable.""",

    # Chapter 12: Nonlinear Sequence Processing
    "book2/ch12/parity_network": """
        A nonlinear recurrent network that computes the parity of a bit
        sequence, a task that requires remembering state from earlier time
        steps.""",
    "book2/ch12/constant_error_carousel": """
        The constant error carousel at the heart of the LSTM: a linear unit
        with unit self-feedback keeps information, and gradients, from fading
        over long sequences.""",
    "book2/ch12/gated_cec": """
        Adds input and feedback gates to the constant error carousel, showing
        how the gates decide what enters the memory and how much of it is
        kept.""",

    # Chapter 13: Sequence-to-Sequence and Attention
    "book2/ch13/self_attention_2d": """
        The geometry of self-attention in two dimensions: the query, key and
        value projections of two input vectors and the attention outputs they
        produce.""",
    "book2/ch13/self_attention_3d": """
        Shows the attention output as a surface over one input vector, to see
        how the output responds as that input moves while the other is
        fixed.""",
}
