function [J ,grad] = nnCostFunction1(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer1_size, ...
                                   hidden_layer2_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer1_size * (input_layer_size + 1)), ...
                 hidden_layer1_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer1_size * (input_layer_size + 1))):hidden_layer2_size * (hidden_layer1_size + 1) + hidden_layer1_size  * ( input_layer_size + 1 )), ...
                 hidden_layer2_size, (hidden_layer1_size + 1));

Theta3 = reshape(nn_params((1 + (hidden_layer2_size * (hidden_layer1_size + 1)) + hidden_layer1_size * ( input_layer_size + 1)):end), ...
				num_labels, (hidden_layer2_size + 1));

m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));


h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
h3 = sigmoid([ones(m, 1) h2] * Theta3');

%'

K = num_labels;
y_k = eye(K);
cost = zeros(K,1);

for i=1:m
	value = -y_k(:,y(i)) .* log(h3(i,:)') - (1 - y_k(:,y(i))) .* log(1 - h3(i,:)') ;
	cost = cost + value ;
end

J = sum(cost) / m;
regularizationTerm = sum(sum(Theta1(:,2:end) .^2)) + sum(sum(Theta2(:,2:end).^2)) + sum(sum(Theta3(:,2:end).^2));
J = J + regularizationTerm * lambda / (2 * m);

Delta3 = zeros(size(Theta3));
Delta2 = zeros(size(Theta2));
Delta1 = zeros(size(Theta1));

for t=1:m

	a1 = X(t,:);
	z2 = [1 a1] * Theta1';
	a2 = sigmoid(z2);
	z3 = [1 a2] * Theta2';
    a3 = sigmoid(z3);
    z4 = [1 a3] * Theta3';
    a4 = sigmoid(z4);

   	delta4 = a4' - y_k(:,y(t));
	delta3 = (Theta3' * delta4 ) .* [0 ; sigmoidGradient(z3)'];
	delta3 = delta3(2:end,1);
	delta2 = (Theta2' * delta3 ) .* [0 ; sigmoidGradient(z2)'];
	delta2 = delta2(2:end,1);
	
	Delta3 = Delta3 + delta4 * [1 a3];
	Delta2 = Delta2 + delta3 * [1 a2];   
	Delta1 = Delta1 + delta2 * [1 a1];

end



Theta1_grad = Delta1 /m + ( lambda / m ) * [zeros(size(Theta1,1),1) , Theta1(:,2:end)]  ;
Theta2_grad = Delta2 /m + ( lambda / m ) * [zeros(size(Theta2,1),1) , Theta2(:,2:end)]  ; 
Theta3_grad = Delta3 /m + ( lambda / m ) * [zeros(size(Theta3,1),1) , Theta3(:,2:end)]  ; 
grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];

end
