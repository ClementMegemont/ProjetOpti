@doc doc"""

#### Objet

Minimise une fonction de ``\mathbb{R}^{n}`` à valeurs dans ``\mathbb{R}`` en utilisant l'algorithme des régions de confiance. 

La solution approchées des sous-problèmes quadratiques est calculé 
par le pas de Cauchy ou le pas issu de l'algorithme du gradient conjugue tronqué

#### Syntaxe
```julia
xmin, fxmin, flag, nb_iters = Regions_De_Confiance(algo,f,gradf,hessf,x0,option)
```

#### Entrées :

   - algo        : (String) string indicant la méthode à utiliser pour calculer le pas
        - "gct"   : pour l'algorithme du gradient conjugué tronqué
        - "cauchy": pour le pas de Cauchy
   - f           : (Function) la fonction à minimiser
   - gradf       : (Function) le gradient de la fonction f
   - hessf       : (Function) la hessiene de la fonction à minimiser
   - x0          : (Array{Float,1}) point de départ
   - options     : (Array{Float,1})
     - deltaMax       : utile pour les m-à-j de la région de confiance
                      ``R_{k}=\left\{x_{k}+s ;\|s\| \leq \Delta_{k}\right\}``
     - gamma1, gamma2 : ``0 < \gamma_{1} < 1 < \gamma_{2}`` pour les m-à-j de ``R_{k}``
     - eta1, eta2     : ``0 < \eta_{1} < \eta_{2} < 1`` pour les m-à-j de ``R_{k}``
     - delta0         : le rayon de départ de la région de confiance
     - max_iter       : le nombre maximale d'iterations
     - Tol_abs        : la tolérence absolue
     - Tol_rel        : la tolérence relative
     - epsilon       : epsilon pour les tests de stagnation

#### Sorties:

   - xmin    : (Array{Float,1}) une approximation de la solution du problème : 
               ``\min_{x \in \mathbb{R}^{n}} f(x)``
   - fxmin   : (Float) ``f(x_{min})``
   - flag    : (Integer) un entier indiquant le critère sur lequel le programme s'est arrêté (en respectant cet ordre de priorité si plusieurs critères sont vérifiés)
      - 0    : CN1
      - 1    : stagnation du ``x``
      - 2    : stagnation du ``f``
      - 3    : nombre maximal d'itération dépassé
   - nb_iters : (Integer)le nombre d'iteration qu'à fait le programme

#### Exemple d'appel
```julia
algo="gct"
f(x)=100*(x[2]-x[1]^2)^2+(1-x[1])^2
gradf(x)=[-400*x[1]*(x[2]-x[1]^2)-2*(1-x[1]) ; 200*(x[2]-x[1]^2)]
hessf(x)=[-400*(x[2]-3*x[1]^2)+2  -400*x[1];-400*x[1]  200]
x0 = [1; 0]
options = []
xmin, fxmin, flag, nb_iters = Regions_De_Confiance(algo,f,gradf,hessf,x0,options)
```
"""
function Regions_De_Confiance(algo,f::Function,gradf::Function,hessf::Function,x0,options)

    if options == []
        deltaMax = 10
        gamma1 = 0.5
        gamma2 = 2.00
        eta1 = 0.25
        eta2 = 0.75
        delta0 = 2
        max_iter = 1000
        Tol_abs = sqrt(eps())
        Tol_rel = 1e-15
        epsilon = 1.e-2
    else
        deltaMax = options[1]
        gamma1 = options[2]
        gamma2 = options[3]
        eta1 = options[4]
        eta2 = options[5]
        delta0 = options[6]
        max_iter = options[7]
        Tol_abs = options[8]
        Tol_rel = options[9]
        epsilon = options[10]
    end

    n = length(x0)
    xmin = zeros(n)
    fxmin = f(xmin)
    flag = 0
    nb_iters = 0
    delta_k = delta0
    xmin = x0
    crit1 = false
    crit2 = false
    Non_arret = (nb_iters + 1 <= max_iter) & ~(norm(gradf(x0)) <= Tol_abs)
    while Non_arret
        if algo == "cauchy"
            s_k, e = Pas_De_Cauchy(gradf(xmin), hessf(xmin), delta_k)
        elseif algo == "gct"
            s_k = Gradient_Conjugue_Tronque(gradf(xmin), hessf(xmin), [delta_k; max_iter; Tol_rel])
        end
        rho_k = (f(xmin) - f(xmin+s_k))/(-gradf(xmin)'*s_k - (1/2)*s_k'*hessf(xmin)*s_k)
        if rho_k >= eta1
            crit1 = norm(s_k) <= epsilon*max(Tol_rel*norm(xmin), Tol_abs)
            crit2 = norm(f(xmin + s_k) - f(xmin)) <= epsilon*max(Tol_rel*abs(f(xmin)), Tol_abs)
            xmin = xmin + s_k
        end
        if rho_k >= eta2
            delta_k = min(gamma2*delta_k, deltaMax)
        elseif rho_k >= eta1
            delta_k = delta_k
        else 
            delta_k = gamma1*delta_k
        end
        crit0 = norm(gradf(xmin)) <= max(Tol_rel*norm(gradf(x0)), Tol_abs)
        nb_iters = nb_iters + 1
        crit3 = nb_iters < max_iter
        if crit0
            flag = 0
        elseif crit1
            flag = 1
        elseif crit2
            flag = 2
        elseif ~crit3
            flag = 3
        end
        Non_arret = ~crit0 & ~crit1 & ~crit2 & crit3
    end
    fxmin = f(xmin)
    return xmin, fxmin, flag, nb_iters
end
