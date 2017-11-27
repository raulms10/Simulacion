function W = regresionLogistica(X,Y,eta)

[N,D]=size(X);
W = zeros(D,1);

for iter = 1:1000
    %%% Completar el c�digo %%% 
    %%% Completar el c�digo %%% 
    for j = 1:D
        W(j) = W(j) - eta*(sigmoide(X*W) - Y)'*X(:,j)/N;
    end
    %%% Fin de la modificaci�n %%%
    %%% Fin de la modificaci�n %%%
end

end