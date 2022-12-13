@doc doc"""

#### Objet
Cette fonction calcule une solution approchée du problème
```math
\min_{||s||< \Delta} s^{t}g + \frac{1}{2}s^{t}Hs
```
par le calcul du pas de Cauchy.

#### Syntaxe
```julia
s, e = Pas_De_Cauchy(g,H,delta)
```

#### Entrées
 - g : (Array{Float,1}) un vecteur de ``\mathbb{R}^n``
 - H : (Array{Float,2}) une matrice symétrique de ``\mathbb{R}^{n\times n}``
 - delta  : (Float) le rayon de la région de confiance

#### Sorties
 - s : (Array{Float,1}) une approximation de la solution du sous-problème
 - e : (Integer) indice indiquant l'état de sortie:
        si g != 0
            si on ne sature pas la boule
              e <- 1
            sinon
              e <- -1
        sinon
            e <- 0

#### Exemple d'appel
```julia
g = [0; 0]
H = [7 0 ; 0 2]
delta = 1
s, e = Pas_De_Cauchy(g,H,delta)
```
"""
function Pas_De_Cauchy(g,H,delta)

    e = 0
    n = length(g)
    s = zeros(n)
    a = g'*H*g
    b = -norm(g)^2
    if norm(g) == 0
        s = 0
        e = 0
    else
        if a <= 0
            t_min = delta/norm(g)
            s = -t_min*g
            e = -1
        else
            t_min = -b/a
            if t_min < delta/norm(g)
                s = -t_min*g
                e = 1
            else
                t_min = delta/norm(g)
                s = -t_min*g
                e = -1
            end
        end
    end 
    return s, e
end
