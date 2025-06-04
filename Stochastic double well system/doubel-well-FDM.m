clear

tic
r = 1; sigma = 1; J = 20000; 
eps = 1;              % Diffusivity
f = @(x) x-x.^3;           % Drift function (can modify as needed)

Jt = 2 * J - 1;       
h = 1 / J;             
x = -2:h:2;

% Coefficients for Brownian motion
Chh = (sigma^2) / (2 * r^2) / h^2; 


b = zeros(Jt, 1); % Coefficient of U_j
a = zeros(Jt, 1); % Coefficient of U_(j-1)
c = zeros(Jt, 1); % Coefficient of U_(j+1)

% Non-integral part
b(2:Jt-1) = -2 * Chh; % Central difference for internal points
b(1)  = -2 * Chh - 3 * f(r * x(J+2)) / (2 * h * r); % One-sided near left boundary
b(Jt) = -2 * Chh + 3 * f(r * x(3 * J)) / (2 * h * r); % One-sided near right boundary

a(2:Jt-1) = (Chh - f(r * x(J+2:3*J-2)) / (2 * h * r))'; % Coefficient of U_(j-1)
c(2:Jt-1) = (Chh + f(r * x(J+2:3*J-2)) / (2 * h * r))'; % Coefficient of U_(j+1)

c(1)  = -5 * Chh + 4 * f(r * x(J+2)) / (2 * h * r); % One-sided diff
a(Jt) = -5 * Chh - 4 * f(r * x(3 * J)) / (2 * h * r); % One-sided diff
vp2 = zeros(Jt, 1); vp2(3) = 4 * Chh - f(r * x(3)) / (2 * h * r);  % one-sided diff
vp3 = zeros(Jt, 1); vp3(4) = -Chh; % one-sided diff 
vm2 = zeros(Jt, 1); vm2(Jt-2) = 4 * Chh + f(r * x(Jt-2)) / (2 * h * r);  % one-sided diff
vm3 = zeros(Jt, 1); vm3(Jt-3) = -Chh; % one-sided diff 

A = spdiags([vm3 vm2 [a(2:end); 0] b ...
           [0; c(1:end-1)] vp2 vp3], -3:3, Jt, Jt);

U = A \ (-1 * ones(Jt, 1)); 
U=[0; U; 0];


X = -r:(h * r):r;

hold on;
plot(X, U, 'b--');
xlabel('x');
ylabel('u(x)');


