@doc doc"""
#### Objet
Cette fonction calcule une solution approchée du problème

```math
\min_{||s||< \Delta}  q(s) = s^{t} g + \frac{1}{2} s^{t}Hs
```

par l'algorithme du gradient conjugué tronqué

#### Syntaxe
```julia
s = Gradient_Conjugue_Tronque(g,H,option)
```

#### Entrées :   
   - g : (Array{Float,1}) un vecteur de ``\mathbb{R}^n``
   - H : (Array{Float,2}) une matrice symétrique de ``\mathbb{R}^{n\times n}``
   - options          : (Array{Float,1})
      - delta    : le rayon de la région de confiance
      - max_iter : le nombre maximal d'iterations
      - tol      : la tolérance pour la condition d'arrêt sur le gradient

#### Sorties:
   - s : (Array{Float,1}) le pas s qui approche la solution du problème : ``min_{||s||< \Delta} q(s)``

#### Exemple d'appel:
```julia
gradf(x)=[-400*x[1]*(x[2]-x[1]^2)-2*(1-x[1]) ; 200*(x[2]-x[1]^2)]
hessf(x)=[-400*(x[2]-3*x[1]^2)+2  -400*x[1];-400*x[1]  200]
xk = [1; 0]
options = []
s = Gradient_Conjugue_Tronque(gradf(xk),hessf(xk),options)
```
"""
function Gradient_Conjugue_Tronque(g,H,options)

    "# Si option est vide on initialise les 3 paramètres par défaut"
    if options == []
        delta = 2
        max_iter = 100
        tol = 1e-6
    else
        delta = options[1]
        max_iter = options[2]
        tol = options[3]
    end

    n = length(g)
    s = zeros(n)
    j = 0
    g_k = g
    p_k = -g
    while (j < 2*n) & (norm(g_k) > max(norm(g)*tol, tol ))
        k = p_k'*H*p_k
        if k <= 0
            sigma = Resoudre(s, p_k, delta, g, H, true)
            return s + sigma*p_k
        end
        alpha = (g_k'*g_k)/k
        if norm(s + alpha*p_k) >= delta
            sigma = Resoudre(s, p_k, delta, g, H, false)
            return s + sigma*p_k
        end
        s = s + alpha*p_k
        mem = g_k'*g_k
        g_k = g_k + alpha*H*p_k
        beta = (g_k'*g_k)/mem
        p_k = -g_k + beta*p_k
        j = j + 1
    end
    return s
end


function Resoudre(s, p, delta, g, H, cas)
    a = norm(p)^2
    b = 2*p'*s
    c = norm(s)^2 - delta^2
    sig1 = (-b - sqrt(b^2 - 4*a*c))/(2*a)
    sig2 = (-b + sqrt(b^2 - 4*a*c))/(2*a)
    x1 = s + sig1*p
    x2 = s + sig2*p
    q1 = 2*g'*x1 + x1'*H*x1
    q2 = 2*g'*x2 + x2'*H*x2
    if cas
        if q1 < q2
            return sig1
        else
            return sig2
        end
    else
        return max(sig1, sig2)
    end
end 
