clear ; close all; clc

input_layer_size  = 900;  
hidden_layer1_size = 630;
hidden_layer2_size = 360;  
num_labels = 92;             


fprintf('Loading and Visualizing Data ...\n')
X = dlmread('X.mat');
y = dlmread('Y.mat');
samples =size(X,1);
options = optimset('MaxIter', 50);
lambda = 1;
                          
%% =========== Part 1: Loading and Visualizing Data =============


X = X(1:samples,:);
y = y(1:samples,:);
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 2: Initializing Pameters ================

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInit(input_layer_size, hidden_layer1_size);
initial_Theta2 = randInit(hidden_layer1_size, hidden_layer2_size);
initial_Theta3 = randInit(hidden_layer2_size, num_labels);

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:)];

%% =================== Part 3: Training NN ===================

fprintf('\nTraining Neural Network... \n');

costFunction1 = @(p) nnCostFunction1(p, ...
                                   input_layer_size, ...
                                   hidden_layer1_size, ...
                                   hidden_layer2_size, ...
                                   num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction1, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer1_size * (input_layer_size + 1)), ...
                 hidden_layer1_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer1_size * (input_layer_size + 1))):hidden_layer2_size * (hidden_layer1_size + 1) + hidden_layer1_size  * ( input_layer_size + 1 )), ...
                 hidden_layer2_size, (hidden_layer1_size + 1));

Theta3 = reshape(nn_params((1 + (hidden_layer2_size * (hidden_layer1_size + 1)) + hidden_layer1_size * ( input_layer_size + 1)):end), ...
				num_labels, (hidden_layer2_size + 1));


dlmwrite('Theta32.mat',Theta3);
dlmwrite('Theta22.mat',Theta2);
dlmwrite('Theta12.mat',Theta1);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================= Part 4: Visualize Weights =================

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ================= Part 5: Implement Predict =================

pred = predict(X, Theta1, Theta2 , Theta3);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


