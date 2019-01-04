%% Warm Up
rng(19875) 
X = randi([0, 1], [1000 5000]);
X_train = X(1:900,:);
X_test = X(901:1000,:);

%% 1)
% data preparation:
rng(19875) 
w = zeros(5000,1);
w(1:15,:) = 1;
y = X*w+randi([-1,1],1000,1);
y_train = y(1:900,:);
y_test = y(901:1000,:);

%%
...
e= (1/100)*norm(X_test*weight-y_test)^2
...
e= (1/100)*norm(X_test*weight-y_test)^2

%% 2)
% data preparation:
rng(19875) 
w = randi([-1,1],5000,1);
w(1:500,:) = 0;
y = X*w+randi([-1,1],1000,1);
y_train = y(1:900,:);
y_test = y(901:1000,:);
%%
...
e= (1/100)*norm(X_test*weight-y_test)^2 % MSE for LASSO
...
e= (1/100)*norm(X_test*weight-y_test)^2 % MSE for Ridge
%% Main Activity 
% read in data from csv. 
% X stores data from 72 patients with 3571 features classified as two
% classes. Among which, we use 1 to represent calss "ALL" and -1 tp
% represent "AML"
data = csvread('leukemia_small.csv', 0, 0);
y =data(1,:)';
X =data(2:3572,:)';

%% 6)  
%
 [E,EFitInfo] = lasso(X,y,'Alpha',0.001);
 [M,I] = min(EFitInfo.MSE);
 C =  corrcoef(X);
 temp = C(1:12,1:12); 
 weight = E(:,I);
 