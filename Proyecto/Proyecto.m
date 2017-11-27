clear all
close all
clc

disp('Modelos');
disp('1 -> Funciones Discriminantes Gausianas');
disp('2 -> K vecinos mas cercanos');
disp('3 -> Redes Neuronales Artificiales');
disp('4 -> Random Forest');
disp('5 -> Maquinas de Soporte Vectorial');

rng('default');
%tipo = input('Ingrese opcion');
tipo = 1;

load('sat.mat');

X = data(:, 1:end-1);
Y = data(:, end:end);
Y(Y == 7) = 6;

nroClases = 7;
for c=1:nroClases
    s = sum(Y == c);
    Texto = ['De la clase ', num2str(c), ' hay ', num2str(s), ' muestras'];
    disp(Texto);
end

if tipo == 1
    grado = 2; % Grado del polinomio
    eta = 0.1; % Tasa de aprendizaje
    
    X = potenciaPolinomio(X,grado);
    
    nroMuestras = size(X,1);
    nroClases = 6;
    
    rng('default');
    ind=randperm(nroMuestras); %%% Se seleccionan los indices de forma aleatoria
    
    porcentaje = ceil(0.7*nroMuestras);
    
    copiaY = Y;
    
%     for c=1:nroClases
%         s = sum(Y == c);
%         Texto = ['De la clase ', num2str(c), ' se tomaron ', num2str(s), ' muestras'];
%         disp(Texto);
%     end
    
    for c=1:nroClases
        Y = copiaY;
        Y(Y == c) = 1;
        Y(Y ~= 1) = 0;
        
%         Y = copiaY;
%         Y(:, :) = 0;
%         Y(copiaY == c) = 1;
        
        Xtrain = X(ind(1:porcentaje),:);
        Xtest = X(ind(porcentaje+1:end),:);
        Ytrain = Y(ind(1:porcentaje),:);
        Ytest = Y(ind(porcentaje+1:end),:);
        
        Texto = ['Para la clase ',num2str(c),' para entrenamiento hay ',num2str(sum(Ytrain)),' - len: ',num2str(size(Ytrain, 1)), ' para prueba hay ', num2str(sum(Ytest)), ' en total ', num2str(sum(Y))];
        disp(Texto);
                
        %%% Normalizacion %%%
        
        %[Xtrain,mu,sigma]=zscore(Xtrain);
        %Xtest=normalizar(Xtest,mu,sigma);
        
        %%% Se extienden las matrices %%%
        
        Xtrain=[Xtrain,ones(porcentaje,1)];
        Xtest=[Xtest,ones(nroMuestras - porcentaje, 1)];
        
        %%% Se aplica la regresion logistica %%%
        
        W=regresionLogistica(Xtrain,Ytrain,eta); %%% Se optienen los W coeficientes del polinomio
        
        Yesti=(W'*Xtrain')';
        Yesti(Yesti>=0)=1;
        Yesti(Yesti<0)=0;
        
        Eficiencia=(sum(Yesti==Ytrain))/length(Ytrain);
        Texto=['     Eficiencia en entrenamiento: ',num2str(Eficiencia), ' -> ', num2str(sum(Yesti==Ytrain)), ' de ', num2str(length(Ytrain))];
        disp(Texto);
        
        Yesti=(W'*Xtest')';
        Yesti(Yesti>=0)=1;
        Yesti(Yesti<0)=0;
        
        Eficiencia=(sum(Yesti==Ytest))/length(Ytest);
        Texto=['     Eficiencia en prueba: ',num2str(Eficiencia), ' -> ', num2str(sum(Yesti==Ytest)), ' de ', num2str(length(Ytest))];
        disp(Texto);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
elseif tipo == 2
    Xclas=X;
    Yclas=Y;
    %%% Se hace la particion entre los conjuntos de entrenamiento y prueba.
    %%% Esta particion se hace forma aletoria %%%
    
    N=size(Xclas,1);

    porcentaje=N*0.7;
    rng('default');
    ind=randperm(N); %%% Se seleccionan los indices de forma aleatoria

    Xtrain=Xclas(ind(1:porcentaje),:);
    Xtest=Xclas(ind(porcentaje+1:end),:);
    Ytrain=Yclas(ind(1:porcentaje),:);
    Ytest=Yclas(ind(porcentaje+1:end),:);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%% Normalizacion %%%
    
    [Xtrain,mu,sigma]=zscore(Xtrain);
    Xtest=normalizar(Xtest,mu,sigma);
    
    %%%%%%%%%%%%%%%%%%%%%

    %%% Se aplica la clasificacion con KNN %%%
    
    k=10;
    Yesti=vecinosCercanos(Xtest,Xtrain,Ytrain,k,'class'); 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%% Se encuentra la eficiencia y el error de clasificacion %%%
    nroClases = 7;
    for c=1:nroClases
        s = sum(Y == c);
        Texto = ['De la clase ', num2str(c), ' se tomaron ', num2str(s), ' muestras'];
        disp(Texto);
    end
    
%     a = 0;
%     b = 0;
%     c = 0;
%     for i=1:size(Yclas)
%         if(Yclas(i) == 1)
%            a=a+1; 
%         end
%         if(Yclas(i) == 2)
%            b=b+1; 
%         end
%         if(Yclas(i) == 3)
%            c = c+1; 
%         end
%     end
%     Texto=strcat('clase 1: ',{' '},num2str(a));
%     disp(Texto);
%     Texto=strcat('clase 2: ',{' '},num2str(b));
%     disp(Texto);
%     Texto=strcat('clase 3: ',{' '},num2str(c));
%     disp(Texto);
        
    
    Eficiencia=(sum(Yesti==Ytest))/length(Ytest);
    Error=1-Eficiencia;
    
    Texto=strcat('La eficiencia en prueba es: ',{' '},num2str(Eficiencia));
    disp(Texto);
    Texto=strcat('El error de clasificacion en prueba es: ',{' '},num2str(Error));
    disp(Texto);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


elseif tipo == 3

elseif tipo == 4

elseif tipo == 5
 
else
    disp('Opcion no valida');
end