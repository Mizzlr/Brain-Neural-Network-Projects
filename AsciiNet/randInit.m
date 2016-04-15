function W = randInit(fan_in, fan_out)
 
W = zeros(fan_out, 1 + fan_in);
epsilon = sqrt(6/(fan_out + fan_in))
W = rand(fan_out, 1 + fan_in) * 2 * epsilon - epsilon;

end
