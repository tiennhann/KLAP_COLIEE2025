
%%% naive approach, ignore relationship between abstract factor
query.
% query link activates bfactor
bfactor(X) :- entail((query, bfactor(X))).

% baselevel factor activate abstract factor
sup(afactor(X)) :- sup(bfactor(Y), afactor(X)), bfactor(Y).
atk(afactor(X)) :- atk(bfactor(Y), afactor(X)), bfactor(Y).
afactor(X) :- sup(afactor(X)), not atk(afactor(X)).

% abstract factor activates link to root
sup_root(Y) :- afactor(X), sup(afactor(X), Y).
atk_root(Y) :- afactor(X), atk(afactor(X), Y).

yes :- sup_root(Y),  not atk_root(X), afactor(X).

no :- not yes.

#show yes/0.
#show no/0.