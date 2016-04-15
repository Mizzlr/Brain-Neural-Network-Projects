function p = predict(X, Theta1, Theta2, Theta3, Theta4)

m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
[dummy, p] = max(h2, [], 2);

if exist('Theta3','var')
	h3 = sigmoid([ones(m, 1) h2] * Theta3');
	[dummy, p] = max(h3, [], 2);
end

if exist('Theta4','var')
	h4 = sigmoid([ones(m, 1) h3] * Theta4');
	[dummy, p] = max(h4, [], 2);
end


end
