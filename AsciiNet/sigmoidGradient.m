function g = sigmoidGradient(z)

g = zeros(size(z));

g1 = 1 ./ (1 + exp(-z));
g = g1 .* (1 - g1 ) ;

end
